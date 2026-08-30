"""Run the remaining RadioUNet/Restormer revision training jobs in order.

The queue is intended to be launched while the existing RadioUNet Physics-L1
20-epoch job is still running.  It waits for that launcher PID, validates the
resulting checkpoint, and then executes these stages sequentially:

1. RadioUNet Physics-L1: resume epoch 20 and finish epoch 50 (seed 42).
2. Restormer L1: train the requested new seeds.
3. Restormer Physics-L1: warm-start each matching L1 run for 20 epochs.
4. Restormer Physics-L1: fully resume each run from epoch 20 to epoch 50.

Every Restormer seed uses the original configs, fixed split seed, global batch,
optimizer, scheduler, and four-process DDP protocol.  Explicit training seeds
only change initialization/data order and isolate outputs below ``seed_<N>``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RADIOUNET_STAGE20_CHECKPOINT = Path(
    "outputs/radiounet_c/physics_l1/checkpoints/final_model.pth"
)

_GUARDED_FILES = (
    "configs/config.yaml",
    "configs/config_ablation.yaml",
    "configs/config_ablation_50ep.yaml",
    "configs/config_radiounet_ablation_50ep.yaml",
    "datasets/radiomapseer_dataset.py",
    "losses/loss.py",
    "model/factory.py",
    "model/radio_map_model.py",
    "model/radiounet.py",
    "scripts/run_multi_seed.py",
    "training/train.py",
    "training/train_physics.py",
)


@dataclass(frozen=True)
class QueueStep:
    """One foreground command in the fail-fast training queue."""

    name: str
    command: tuple[str, ...]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wait for RadioUNet epoch 20, then run the revision training queue"
    )
    parser.add_argument(
        "--wait-pid",
        type=int,
        default=None,
        help=(
            "PID of the currently running RadioUNet torchrun launcher. If it has "
            "already exited, checkpoint validation starts immediately."
        ),
    )
    parser.add_argument(
        "--restormer-seeds",
        type=int,
        nargs="+",
        default=[123, 2016],
        help="New Restormer training seeds; seed 42 is reused from the completed run",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed map split shared by seed 42 and every new training seed",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--allow-existing-seed-outputs",
        action="store_true",
        help="Allow new seeded runs to reuse/overwrite non-empty seed directories",
    )
    parser.add_argument(
        "--allow-source-changes",
        action="store_true",
        help="Do not stop if guarded configs or training sources change while queued",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the protocol and print commands without waiting or training",
    )
    return parser.parse_args()


def validate_args(args):
    seeds = [int(seed) for seed in args.restormer_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--restormer-seeds must contain unique values")
    if any(seed < 0 for seed in seeds):
        raise ValueError("training seeds must be non-negative")
    if 42 in seeds:
        raise ValueError(
            "Do not include seed 42: the completed legacy Restormer run is reused"
        )
    if args.split_seed != 42:
        raise ValueError(
            "This controlled comparison must keep split_seed=42 to match the existing run"
        )
    if args.wait_pid is not None and args.wait_pid <= 0:
        raise ValueError("--wait-pid must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be positive")

    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_ids) < args.nproc_per_node:
        raise ValueError(
            f"{args.nproc_per_node} DDP processes requested, but only "
            f"{len(gpu_ids)} GPU IDs were provided"
        )


def _load_yaml(relative_path):
    path = _PROJECT_ROOT / relative_path
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def _assert_equal(label, values):
    first_name, first_value = values[0]
    for name, value in values[1:]:
        if value != first_value:
            raise ValueError(
                f"Protocol mismatch for {label}: "
                f"{first_name}={first_value!r}, {name}={value!r}"
            )


def validate_protocol(split_seed):
    """Reject config drift that would confound the Restormer seed comparison."""
    configs = {
        "l1": _load_yaml("configs/config.yaml"),
        "physics20": _load_yaml("configs/config_ablation.yaml"),
        "physics50": _load_yaml("configs/config_ablation_50ep.yaml"),
        "radio50": _load_yaml("configs/config_radiounet_ablation_50ep.yaml"),
    }

    for name in ("l1", "physics20", "physics50"):
        config = configs[name]
        if config["model"].get("name", "restormer").lower() != "restormer":
            raise ValueError(f"{name} config is not a Restormer config")
        effective_split = int(
            config["data"].get("split_seed", config["training"]["seed"])
        )
        if effective_split != split_seed:
            raise ValueError(
                f"{name} uses split_seed={effective_split}, expected {split_seed}"
            )
        if int(config["training"]["seed"]) != 42:
            raise ValueError(
                f"{name} must retain legacy config seed=42; CLI overrides only new runs"
            )

    rest_names = ("l1", "physics20", "physics50")
    for section in ("data", "model"):
        _assert_equal(
            f"Restormer {section}",
            [(name, configs[name][section]) for name in rest_names],
        )

    shared_training_keys = (
        "batch_size",
        "lr",
        "weight_decay",
        "scheduler",
        "T_max",
        "eta_min",
        "grad_clip",
        "grad_accum_steps",
        "amp_init_scale",
    )
    for key in shared_training_keys:
        _assert_equal(
            f"Restormer training.{key}",
            [
                (name, configs[name]["training"].get(key))
                for name in rest_names
            ],
        )

    if int(configs["l1"]["training"]["epochs"]) != 50:
        raise ValueError("Restormer L1 must remain a 50-epoch baseline")
    if configs["l1"]["loss"].get("primary") != "l1":
        raise ValueError("Restormer baseline must use L1")
    if int(configs["physics20"]["training"]["epochs"]) != 20:
        raise ValueError("Restormer Physics-L1 warm-start stage must use 20 epochs")
    if bool(configs["physics20"]["training"].get("full_resume", False)):
        raise ValueError("Restormer Physics-L1 epoch-20 stage must be a warm start")
    if int(configs["physics50"]["training"]["epochs"]) != 50:
        raise ValueError("Restormer Physics-L1 continuation must end at epoch 50")
    if not bool(configs["physics50"]["training"].get("full_resume", False)):
        raise ValueError("Restormer Physics-L1 epoch-50 config must fully resume")
    _assert_equal(
        "Restormer Physics-L1 loss",
        [
            (name, configs[name]["loss"])
            for name in ("physics20", "physics50")
        ],
    )

    radio = configs["radio50"]
    if radio["model"].get("name") != "radiounet_c":
        raise ValueError("RadioUNet continuation config must select radiounet_c")
    if int(radio["training"]["epochs"]) != 50:
        raise ValueError("RadioUNet continuation must end at epoch 50")
    if not bool(radio["training"].get("full_resume", False)):
        raise ValueError("RadioUNet continuation config must fully resume")
    if int(radio["data"].get("split_seed", radio["training"]["seed"])) != split_seed:
        raise ValueError("RadioUNet continuation split does not match split_seed=42")


def build_queue_steps(
    *,
    python_executable,
    restormer_seeds,
    split_seed,
    gpus,
    nproc_per_node,
):
    seeds = [str(seed) for seed in restormer_seeds]
    multi_seed_script = str(_PROJECT_ROOT / "scripts/run_multi_seed.py")

    radio_command = (
        python_executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "training/train_physics.py",
        "--config",
        "configs/config_radiounet_ablation_50ep.yaml",
        "--resume",
        os.fspath(_RADIOUNET_STAGE20_CHECKPOINT),
        "--split-seed",
        str(split_seed),
        "--full-resume",
    )
    common_multi_seed = (
        "--seeds",
        *seeds,
        "--split-seed",
        str(split_seed),
        "--nproc-per-node",
        str(nproc_per_node),
        "--gpus",
        gpus,
    )
    rest_l1_command = (
        python_executable,
        multi_seed_script,
        "--trainer",
        "l1",
        "--config",
        "configs/config.yaml",
        *common_multi_seed,
    )
    rest_physics20_command = (
        python_executable,
        multi_seed_script,
        "--trainer",
        "physics",
        "--config",
        "configs/config_ablation.yaml",
        *common_multi_seed,
        "--resume-template",
        "outputs/checkpoints/seed_{seed}/best_model.pth",
    )
    rest_physics50_command = (
        python_executable,
        multi_seed_script,
        "--trainer",
        "physics",
        "--config",
        "configs/config_ablation_50ep.yaml",
        *common_multi_seed,
        "--resume-template",
        "outputs/improved_checkpoints/seed_{seed}/final_model.pth",
        "--full-resume",
    )
    return [
        QueueStep("RadioUNet Physics-L1: epoch 20 -> 50 (seed 42)", radio_command),
        QueueStep("Restormer L1: new training seeds", rest_l1_command),
        QueueStep("Restormer Physics-L1: warm start through epoch 20", rest_physics20_command),
        QueueStep("Restormer Physics-L1: epoch 20 -> 50", rest_physics50_command),
    ]


def _process_identity(pid):
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        fields = stat_path.read_text().split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if len(fields) <= 21:
        return None
    return fields[21]


def _process_command(pid):
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        return path.read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except (FileNotFoundError, ProcessLookupError):
        return ""


def wait_for_current_radiounet(pid, poll_seconds):
    if pid is None:
        print("No --wait-pid supplied; validating the epoch-20 checkpoint now", flush=True)
        return False

    identity = _process_identity(pid)
    if identity is None:
        print(f"PID {pid} has already exited; continuing with checkpoint validation", flush=True)
        return False

    command = _process_command(pid)
    required_fragments = (
        "training/train_physics.py",
        "configs/config_radiounet_ablation.yaml",
    )
    if not all(fragment in command for fragment in required_fragments):
        raise RuntimeError(
            f"PID {pid} is not the expected RadioUNet epoch-20 launcher: {command}"
        )

    print(f"Waiting for current RadioUNet launcher PID {pid}", flush=True)
    wait_start = time.monotonic()
    next_status = wait_start
    while _process_identity(pid) == identity:
        now = time.monotonic()
        if now >= next_status:
            elapsed_minutes = (now - wait_start) / 60.0
            print(f"  still running; waited {elapsed_minutes:.1f} min", flush=True)
            next_status = now + 300.0
        time.sleep(poll_seconds)

    print(f"RadioUNet epoch-20 launcher PID {pid} has exited", flush=True)
    return True


def validate_radiounet_checkpoint(relative_path, expected_epoch, split_seed):
    import torch

    path = _PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required RadioUNet checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    required_keys = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "amp_scaler_state_dict",
    )
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise ValueError(f"{path} cannot fully resume; missing keys: {missing}")
    if int(checkpoint.get("epoch", -1)) != expected_epoch:
        raise ValueError(
            f"{path} records epoch={checkpoint.get('epoch')}, expected {expected_epoch}"
        )
    if checkpoint.get("training_variant") != "physics_weighted_l1":
        raise ValueError(f"{path} is not a Physics-L1 checkpoint")
    model_name = checkpoint.get("model_name")
    if model_name is not None and model_name != "radiounet_c":
        raise ValueError(f"{path} records model_name={model_name!r}, expected radiounet_c")
    training_seed = checkpoint.get("training_seed")
    if training_seed is not None and int(training_seed) != 42:
        raise ValueError(f"{path} records training_seed={training_seed}, expected 42")
    recorded_split = checkpoint.get("split_seed")
    if recorded_split is not None and int(recorded_split) != split_seed:
        raise ValueError(
            f"{path} records split_seed={recorded_split}, expected {split_seed}"
        )
    print(
        f"Validated {path}: completed epoch {expected_epoch + 1}, "
        "Physics-L1, full optimizer/scheduler/AMP state",
        flush=True,
    )


def ensure_seed_outputs_are_fresh(seeds, allow_existing):
    if allow_existing:
        return
    conflicts = []
    for seed in seeds:
        tag = f"seed_{seed}"
        for parent in ("outputs/checkpoints", "outputs/improved_checkpoints"):
            path = _PROJECT_ROOT / parent / tag
            if path.is_dir() and any(path.iterdir()):
                conflicts.append(path)
    if conflicts:
        paths = ", ".join(os.fspath(path) for path in conflicts)
        raise FileExistsError(
            "Refusing to overwrite existing seeded outputs: "
            f"{paths}. Inspect them or pass --allow-existing-seed-outputs explicitly."
        )


def source_fingerprints():
    fingerprints = {}
    for relative_path in _GUARDED_FILES:
        path = _PROJECT_ROOT / relative_path
        fingerprints[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints


def assert_sources_unchanged(expected):
    current = source_fingerprints()
    changed = [path for path in expected if current[path] != expected[path]]
    if changed:
        raise RuntimeError(
            "Guarded training/config files changed while the queue was running: "
            + ", ".join(changed)
        )


def run_step(step, gpus):
    print(f"\nStarting: {step.name}", flush=True)
    print(shlex.join(step.command), flush=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpus
    subprocess.run(
        step.command,
        cwd=_PROJECT_ROOT,
        env=environment,
        check=True,
    )
    print(f"Completed: {step.name}", flush=True)


def main():
    args = parse_args()
    validate_args(args)
    validate_protocol(args.split_seed)
    ensure_seed_outputs_are_fresh(
        args.restormer_seeds,
        allow_existing=args.allow_existing_seed_outputs,
    )
    guarded_sources = source_fingerprints()
    steps = build_queue_steps(
        python_executable=sys.executable,
        restormer_seeds=args.restormer_seeds,
        split_seed=args.split_seed,
        gpus=args.gpus,
        nproc_per_node=args.nproc_per_node,
    )

    print("Revision training queue", flush=True)
    print(f"  Restormer training seeds: {args.restormer_seeds}", flush=True)
    print(f"  Fixed split seed: {args.split_seed}", flush=True)
    print(f"  GPUs: {args.gpus}; DDP processes: {args.nproc_per_node}", flush=True)
    for index, step in enumerate(steps, start=1):
        print(f"  {index}. {step.name}", flush=True)
        print(f"     {shlex.join(step.command)}", flush=True)

    if args.dry_run:
        print("Dry run complete; no process was awaited or started", flush=True)
        return

    checkpoint_path = _PROJECT_ROOT / _RADIOUNET_STAGE20_CHECKPOINT
    checkpoint_signature_before = None
    if checkpoint_path.exists():
        stat = checkpoint_path.stat()
        checkpoint_signature_before = (stat.st_mtime_ns, stat.st_size)

    waited = wait_for_current_radiounet(args.wait_pid, args.poll_seconds)
    if waited and checkpoint_signature_before is not None:
        stat = checkpoint_path.stat() if checkpoint_path.exists() else None
        signature_after = None if stat is None else (stat.st_mtime_ns, stat.st_size)
        if signature_after == checkpoint_signature_before:
            raise RuntimeError(
                "The current process exited without updating the RadioUNet "
                "epoch-20 final checkpoint"
            )

    validate_radiounet_checkpoint(
        _RADIOUNET_STAGE20_CHECKPOINT,
        expected_epoch=19,
        split_seed=args.split_seed,
    )

    for index, step in enumerate(steps):
        if not args.allow_source_changes:
            assert_sources_unchanged(guarded_sources)
        run_step(step, args.gpus)
        if index == 0:
            validate_radiounet_checkpoint(
                _RADIOUNET_STAGE20_CHECKPOINT,
                expected_epoch=49,
                split_seed=args.split_seed,
            )

    print("\nAll queued training stages completed successfully", flush=True)


if __name__ == "__main__":
    main()
