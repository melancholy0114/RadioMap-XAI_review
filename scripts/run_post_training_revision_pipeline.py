"""Automate post-queue evaluation and the matched-budget L1 control.

The pipeline is deliberately fail-fast and resumable.  It waits for the active
revision queue, validates every required epoch-50 checkpoint, evaluates the
existing Restormer and RadioUNet checkpoints, creates weights-only Restormer
warm starts, trains one matched-budget L1 continuation per seed, evaluates the
three Restormer arms, and finally writes an evidence-linked decision report.

Optional RadioUNet multi-seed or external-dataset training is never launched by
this script; the final report recommends whether those expansions are needed.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import yaml

from model import checkpoint_model_name


_SOURCE_FILES = (
    "configs/config.yaml",
    "configs/config_ablation_50ep.yaml",
    "configs/config_radiounet.yaml",
    "configs/config_radiounet_ablation_50ep.yaml",
    "configs/config_l1_continuation.yaml",
    "datasets/radiomapseer_dataset.py",
    "explanation/__init__.py",
    "explanation/integrated_gradients.py",
    "losses/loss.py",
    "metrics/__init__.py",
    "metrics/physical_alignment_score.py",
    "model/__init__.py",
    "model/factory.py",
    "model/radio_map_model.py",
    "model/radiounet.py",
    "priors/los_mask.py",
    "priors/obstruction_mask.py",
    "priors/directional_mask.py",
    "priors/__init__.py",
    "training/train.py",
    "training/validate.py",
    "utils/__init__.py",
    "utils/reproducibility.py",
    "analysis/evaluate_multiseed.py",
    "analysis/compare_multiseed.py",
    "analysis/evaluate_multiseed_xai.py",
    "analysis/compare_multiseed_xai.py",
    "analysis/build_existing_checkpoint_report.py",
    "analysis/build_revision_decision_report.py",
    "scripts/prepare_l1_continuation_warmstarts.py",
    "scripts/run_post_training_revision_pipeline.py",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the automated post-training major-revision pipeline"
    )
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2016])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--xai-samples", type=int, default=50)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument(
        "--output-root",
        default="outputs/revision_pipeline",
    )
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprints():
    fingerprints = {}
    for relative in _SOURCE_FILES:
        path = _PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required pipeline source is missing: {path}")
        fingerprints[relative] = _sha256(path)
    return fingerprints


def process_identity(pid):
    if pid is None:
        return None
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().split()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return fields[21] if len(fields) > 21 else None


def wait_for_process(pid, expected_identity, poll_seconds):
    if pid is None or expected_identity is None:
        print("No active queue PID to wait for", flush=True)
        return
    print(f"Waiting for training queue PID {pid}", flush=True)
    while process_identity(pid) == expected_identity:
        time.sleep(poll_seconds)
    print(f"Training queue PID {pid} has exited", flush=True)


def query_compute_processes():
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        cwd=_PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def wait_for_idle_gpus(poll_seconds, consecutive=2):
    idle_observations = 0
    while idle_observations < consecutive:
        processes = query_compute_processes()
        if processes:
            idle_observations = 0
            print(
                "Waiting for GPUs to become idle; compute processes: "
                + " | ".join(processes),
                flush=True,
            )
        else:
            idle_observations += 1
            print(
                f"GPU idle observation {idle_observations}/{consecutive}",
                flush=True,
            )
        if idle_observations < consecutive:
            time.sleep(poll_seconds)


def checkpoint_paths(parent, seeds, filename):
    paths = []
    for seed in seeds:
        directory = Path(parent) if int(seed) == 42 else Path(parent) / f"seed_{seed}"
        paths.append(directory / filename)
    return paths


def validate_checkpoint(path, seed, split_seed, expected_epoch, expected_model, variant=None):
    path = _PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {path}")
    checkpoint = _torch_load(path)
    epoch = int(checkpoint.get("epoch", -1))
    if epoch != int(expected_epoch):
        raise ValueError(f"{path} records epoch={epoch}, expected {expected_epoch}")
    model_name = checkpoint_model_name(checkpoint)
    if model_name != expected_model:
        raise ValueError(f"{path} records model={model_name}, expected {expected_model}")
    recorded_seed = checkpoint.get("training_seed")
    if recorded_seed is not None and int(recorded_seed) != int(seed):
        raise ValueError(f"{path} records training_seed={recorded_seed}, expected {seed}")
    recorded_split = checkpoint.get("split_seed")
    if recorded_split is not None and int(recorded_split) != int(split_seed):
        raise ValueError(f"{path} records split_seed={recorded_split}, expected {split_seed}")
    if variant is not None and checkpoint.get("training_variant") != variant:
        raise ValueError(
            f"{path} records training_variant={checkpoint.get('training_variant')!r}, "
            f"expected {variant!r}"
        )
    del checkpoint


def validate_existing_training(seeds, split_seed):
    groups = (
        ("outputs/checkpoints", "restormer", None),
        ("outputs/improved_checkpoints", "restormer", "physics_weighted_l1"),
    )
    for parent, model_name, variant in groups:
        for seed, path in zip(
            seeds,
            checkpoint_paths(parent, seeds, "final_model.pth"),
        ):
            validate_checkpoint(
                path,
                seed,
                split_seed,
                expected_epoch=49,
                expected_model=model_name,
                variant=variant,
            )
        for path in checkpoint_paths(parent, seeds, "best_model.pth"):
            if not (_PROJECT_ROOT / path).is_file():
                raise FileNotFoundError(f"Best-validation checkpoint not found: {path}")

    validate_checkpoint(
        Path("outputs/radiounet_c/l1/checkpoints/final_model.pth"),
        42,
        split_seed,
        expected_epoch=49,
        expected_model="radiounet_c",
    )
    validate_checkpoint(
        Path("outputs/radiounet_c/physics_l1/checkpoints/final_model.pth"),
        42,
        split_seed,
        expected_epoch=49,
        expected_model="radiounet_c",
        variant="physics_weighted_l1",
    )


def validate_control_protocol(split_seed):
    with open(_PROJECT_ROOT / "configs/config.yaml", "r") as handle:
        baseline = yaml.safe_load(handle)
    with open(_PROJECT_ROOT / "configs/config_ablation_50ep.yaml", "r") as handle:
        physics = yaml.safe_load(handle)
    with open(_PROJECT_ROOT / "configs/config_l1_continuation.yaml", "r") as handle:
        control = yaml.safe_load(handle)

    if control["loss"].get("primary") != "l1":
        raise ValueError("L1 continuation config must use plain L1")
    if int(control["data"].get("split_seed")) != int(split_seed):
        raise ValueError("L1 continuation split seed does not match the pipeline")
    if control["model"] != baseline["model"] or control["model"] != physics["model"]:
        raise ValueError("Baseline, Physics-L1, and L1 continuation models differ")

    data_keys = ("root_dir", "gain_method", "img_size", "train_ratio", "val_ratio", "split_seed")
    for key in data_keys:
        values = {baseline["data"][key], physics["data"][key], control["data"][key]}
        if len(values) != 1:
            raise ValueError(f"Data protocol differs for {key}: {values}")

    training_keys = (
        "batch_size",
        "epochs",
        "lr",
        "weight_decay",
        "scheduler",
        "T_max",
        "eta_min",
        "grad_clip",
        "grad_accum_steps",
        "amp_init_scale",
    )
    for key in training_keys:
        values = {baseline["training"][key], physics["training"][key], control["training"][key]}
        if len(values) != 1:
            raise ValueError(f"Training protocol differs for {key}: {values}")


def validate_json(path):
    path = _PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Expected JSON output not found: {path}")
    with open(path, "r") as handle:
        json.load(handle)


class PipelineState:
    def __init__(self, path, metadata):
        self.path = Path(path)
        if self.path.exists():
            with open(self.path, "r") as handle:
                self.payload = json.load(handle)
        else:
            self.payload = {
                "created_at": now_iso(),
                "status": "initializing",
                "metadata": metadata,
                "stages": {},
            }
            self.save()

    def save(self):
        self.payload["updated_at"] = now_iso()
        _atomic_json_dump(self.payload, self.path)

    def mark_pipeline(self, status, error=None):
        self.payload["status"] = status
        if error is not None:
            self.payload["error"] = str(error)
        self.save()

    def stage_completed(self, name):
        return self.payload.get("stages", {}).get(name, {}).get("status") == "completed"

    def mark_stage(self, name, status, command=None, outputs=None, error=None):
        stage = self.payload.setdefault("stages", {}).setdefault(name, {})
        stage["status"] = status
        if status == "running":
            stage["started_at"] = now_iso()
        if status in ("completed", "failed"):
            stage["finished_at"] = now_iso()
        if command is not None:
            stage["command"] = command
        if outputs is not None:
            stage["outputs"] = [os.fspath(path) for path in outputs]
        if error is not None:
            stage["error"] = str(error)
        self.save()


def run_stage(state, name, command, outputs, fingerprints, environment=None, validator=None):
    if source_fingerprints() != fingerprints:
        raise RuntimeError(
            "A guarded post-training protocol source changed after pipeline launch"
        )
    if state.stage_completed(name):
        for output in outputs:
            if not (_PROJECT_ROOT / output).exists():
                raise FileNotFoundError(
                    f"Stage {name} was marked complete but output is missing: {output}"
                )
        if validator is not None:
            validator()
        print(f"Skipping completed stage: {name}", flush=True)
        return

    if outputs and all((_PROJECT_ROOT / output).exists() for output in outputs):
        try:
            if validator is not None:
                validator()
        except Exception:
            pass
        else:
            printable = shlex.join([os.fspath(value) for value in command])
            state.mark_stage(
                name,
                "completed",
                command=printable,
                outputs=outputs,
            )
            state.payload["stages"][name]["recovered_existing_outputs"] = True
            state.save()
            print(f"Recovered already-complete stage: {name}", flush=True)
            return

    printable = shlex.join([os.fspath(value) for value in command])
    print(f"\nStarting stage: {name}\n{printable}", flush=True)
    state.mark_stage(name, "running", command=printable, outputs=outputs)
    try:
        subprocess.run(
            [os.fspath(value) for value in command],
            cwd=_PROJECT_ROOT,
            env=environment,
            check=True,
        )
        for output in outputs:
            if not (_PROJECT_ROOT / output).exists():
                raise FileNotFoundError(f"Stage {name} did not create {output}")
        if validator is not None:
            validator()
    except Exception as exc:
        state.mark_stage(name, "failed", error=exc)
        raise
    state.mark_stage(name, "completed")
    print(f"Completed stage: {name}", flush=True)


def prediction_command(python, config, seeds, checkpoints, output):
    return [
        python,
        "analysis/evaluate_multiseed.py",
        "--config",
        config,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--checkpoints",
        *[os.fspath(path) for path in checkpoints],
        "--output",
        os.fspath(output),
        "--device",
        "cuda:0",
    ]


def xai_command(python, config, seeds, checkpoints, output, xai_samples, ig_steps):
    return [
        python,
        "analysis/evaluate_multiseed_xai.py",
        "--config",
        config,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--checkpoints",
        *[os.fspath(path) for path in checkpoints],
        "--output",
        os.fspath(output),
        "--methods",
        "integrated_gradients",
        "--n-samples",
        str(xai_samples),
        "--ig-steps",
        str(ig_steps),
        "--device",
        "cuda:0",
    ]


def compare_prediction_command(python, baseline, candidate, output, baseline_label, candidate_label):
    return [
        python,
        "analysis/compare_multiseed.py",
        "--baseline",
        os.fspath(baseline),
        "--candidate",
        os.fspath(candidate),
        "--output",
        os.fspath(output),
        "--baseline-label",
        baseline_label,
        "--candidate-label",
        candidate_label,
    ]


def compare_xai_command(python, baseline, candidate, output, baseline_label, candidate_label):
    return [
        python,
        "analysis/compare_multiseed_xai.py",
        "--baseline",
        os.fspath(baseline),
        "--candidate",
        os.fspath(candidate),
        "--output",
        os.fspath(output),
        "--baseline-label",
        baseline_label,
        "--candidate-label",
        candidate_label,
    ]


def launch_detached(args):
    output_root = _PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "pipeline.log"
    command = [sys.executable, "-u", os.fspath(Path(__file__).resolve())]
    original = sys.argv[1:]
    command.extend(argument for argument in original if argument != "--detach")
    with open(log_path, "a") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=_PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        recorded_log_path = log_path.relative_to(_PROJECT_ROOT)
    except ValueError:
        recorded_log_path = log_path
    launch_record = {
        "launched_at": now_iso(),
        "pid": process.pid,
        "command": command,
        "log": os.fspath(recorded_log_path),
    }
    _atomic_json_dump(launch_record, output_root / "launch.json")
    print(f"Detached pipeline PID: {process.pid}")
    print(f"Log: {log_path}")


def main():
    args = parse_args()
    if args.detach:
        launch_detached(args)
        return
    if len(set(args.seeds)) != len(args.seeds) or sorted(args.seeds) != [42, 123, 2016]:
        raise ValueError("Formal revision protocol requires seeds 42, 123, and 2016")
    if args.nproc_per_node < 1 or args.poll_seconds <= 0:
        raise ValueError("nproc and poll interval must be positive")
    if args.xai_samples < 1 or args.ig_steps < 1:
        raise ValueError("XAI sample and IG step counts must be positive")
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_ids) < args.nproc_per_node:
        raise ValueError("Not enough GPU IDs for the requested DDP processes")

    output_root = Path(args.output_root)
    absolute_output_root = _PROJECT_ROOT / output_root
    absolute_output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = open(absolute_output_root / "pipeline.lock", "a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another post-training revision pipeline is already running") from exc

    fingerprints = source_fingerprints()
    wait_identity = process_identity(args.wait_pid)
    metadata = {
        "pipeline_pid": os.getpid(),
        "python": sys.executable,
        "project_root": os.fspath(_PROJECT_ROOT),
        "wait_pid": args.wait_pid,
        "wait_pid_identity": wait_identity,
        "training_seeds": args.seeds,
        "split_seed": args.split_seed,
        "gpus": gpu_ids,
        "nproc_per_node": args.nproc_per_node,
        "xai_samples": args.xai_samples,
        "ig_steps": args.ig_steps,
        "source_sha256": fingerprints,
    }
    state = PipelineState(absolute_output_root / "status.json", metadata)
    recorded_fingerprints = state.payload.get("metadata", {}).get("source_sha256")
    if recorded_fingerprints is not None and recorded_fingerprints != fingerprints:
        raise RuntimeError(
            "The output root belongs to a different source fingerprint set; "
            "use a new output root rather than mixing protocols"
        )
    # A waiting or failed pipeline may be safely relaunched against the same
    # fingerprint set. Keep runtime identity/arguments current while preserving
    # completed stage records and the original creation time.
    state.payload["metadata"].update(metadata)
    state.save()

    python = sys.executable
    eval_environment = os.environ.copy()
    eval_environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
    training_environment = os.environ.copy()
    training_environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    baseline_checkpoints = checkpoint_paths(
        "outputs/checkpoints", args.seeds, "best_model.pth"
    )
    physics_checkpoints = checkpoint_paths(
        "outputs/improved_checkpoints", args.seeds, "best_model.pth"
    )
    control_checkpoints = [
        Path("outputs/l1_continuation/checkpoints") / f"seed_{seed}" / "best_model.pth"
        for seed in args.seeds
    ]
    radio_baseline_checkpoint = Path("outputs/radiounet_c/l1/checkpoints/best_model.pth")
    radio_physics_checkpoint = Path("outputs/radiounet_c/physics_l1/checkpoints/best_model.pth")

    prediction_dir = output_root / "prediction"
    xai_dir = output_root / "xai"
    comparison_dir = output_root / "comparisons"
    report_dir = output_root / "reports"

    baseline_prediction = prediction_dir / "restormer_baseline_3seed.json"
    physics_prediction = prediction_dir / "restormer_physics_l1_3seed.json"
    control_prediction = prediction_dir / "restormer_l1_continuation_3seed.json"
    radio_baseline_prediction = prediction_dir / "radiounet_baseline_seed42.json"
    radio_physics_prediction = prediction_dir / "radiounet_physics_l1_seed42.json"

    baseline_xai = xai_dir / "restormer_baseline_3seed_ig.json"
    physics_xai = xai_dir / "restormer_physics_l1_3seed_ig.json"
    control_xai = xai_dir / "restormer_l1_continuation_3seed_ig.json"
    radio_baseline_xai = xai_dir / "radiounet_baseline_seed42_ig.json"
    radio_physics_xai = xai_dir / "radiounet_physics_l1_seed42_ig.json"

    pred_bp = comparison_dir / "restormer_baseline_vs_physics_prediction.json"
    pred_bc = comparison_dir / "restormer_baseline_vs_l1_continuation_prediction.json"
    pred_cp = comparison_dir / "restormer_l1_continuation_vs_physics_prediction.json"
    xai_bp = comparison_dir / "restormer_baseline_vs_physics_xai.json"
    xai_bc = comparison_dir / "restormer_baseline_vs_l1_continuation_xai.json"
    xai_cp = comparison_dir / "restormer_l1_continuation_vs_physics_xai.json"
    xai_radio = comparison_dir / "radiounet_baseline_vs_physics_xai.json"

    if args.dry_run:
        commands = [
            prediction_command(python, "configs/config.yaml", args.seeds, baseline_checkpoints, baseline_prediction),
            prediction_command(python, "configs/config_ablation_50ep.yaml", args.seeds, physics_checkpoints, physics_prediction),
            xai_command(python, "configs/config.yaml", args.seeds, baseline_checkpoints, baseline_xai, args.xai_samples, args.ig_steps),
            xai_command(python, "configs/config_ablation_50ep.yaml", args.seeds, physics_checkpoints, physics_xai, args.xai_samples, args.ig_steps),
        ]
        for command in commands:
            print(shlex.join([os.fspath(value) for value in command]))
        print("Dry run only; no process was awaited, evaluated, or trained")
        return

    try:
        state.mark_pipeline("waiting_for_training_queue")
        wait_for_process(args.wait_pid, wait_identity, args.poll_seconds)
        validate_control_protocol(args.split_seed)
        validate_existing_training(args.seeds, args.split_seed)
        if source_fingerprints() != fingerprints:
            raise RuntimeError("Post-training protocol sources changed while waiting")
        wait_for_idle_gpus(args.poll_seconds)
        state.mark_pipeline("running")

        # Existing Restormer prediction and Table-II-style IG evaluation.
        run_stage(
            state,
            "evaluate_existing_restormer_baseline_prediction",
            prediction_command(python, "configs/config.yaml", args.seeds, baseline_checkpoints, baseline_prediction),
            [baseline_prediction],
            fingerprints,
            eval_environment,
            lambda: validate_json(baseline_prediction),
        )
        run_stage(
            state,
            "evaluate_existing_restormer_physics_prediction",
            prediction_command(python, "configs/config_ablation_50ep.yaml", args.seeds, physics_checkpoints, physics_prediction),
            [physics_prediction],
            fingerprints,
            eval_environment,
            lambda: validate_json(physics_prediction),
        )
        run_stage(
            state,
            "compare_existing_restormer_prediction",
            compare_prediction_command(python, baseline_prediction, physics_prediction, pred_bp, "L1", "Physics-L1"),
            [pred_bp],
            fingerprints,
            validator=lambda: validate_json(pred_bp),
        )
        run_stage(
            state,
            "evaluate_existing_restormer_baseline_xai",
            xai_command(python, "configs/config.yaml", args.seeds, baseline_checkpoints, baseline_xai, args.xai_samples, args.ig_steps),
            [baseline_xai],
            fingerprints,
            eval_environment,
            lambda: validate_json(baseline_xai),
        )
        run_stage(
            state,
            "evaluate_existing_restormer_physics_xai",
            xai_command(python, "configs/config_ablation_50ep.yaml", args.seeds, physics_checkpoints, physics_xai, args.xai_samples, args.ig_steps),
            [physics_xai],
            fingerprints,
            eval_environment,
            lambda: validate_json(physics_xai),
        )
        run_stage(
            state,
            "compare_existing_restormer_xai",
            compare_xai_command(python, baseline_xai, physics_xai, xai_bp, "L1", "Physics-L1"),
            [xai_bp],
            fingerprints,
            validator=lambda: validate_json(xai_bp),
        )

        # Existing single-seed RadioUNet evidence for the later expansion decision.
        for name, config, checkpoint, output in (
            ("baseline", "configs/config_radiounet.yaml", radio_baseline_checkpoint, radio_baseline_prediction),
            ("physics", "configs/config_radiounet_ablation_50ep.yaml", radio_physics_checkpoint, radio_physics_prediction),
        ):
            run_stage(
                state,
                f"evaluate_existing_radiounet_{name}_prediction",
                prediction_command(python, config, [42], [checkpoint], output),
                [output],
                fingerprints,
                eval_environment,
                lambda output=output: validate_json(output),
            )
        for name, config, checkpoint, output in (
            ("baseline", "configs/config_radiounet.yaml", radio_baseline_checkpoint, radio_baseline_xai),
            ("physics", "configs/config_radiounet_ablation_50ep.yaml", radio_physics_checkpoint, radio_physics_xai),
        ):
            run_stage(
                state,
                f"evaluate_existing_radiounet_{name}_xai",
                xai_command(python, config, [42], [checkpoint], output, args.xai_samples, args.ig_steps),
                [output],
                fingerprints,
                eval_environment,
                lambda output=output: validate_json(output),
            )
        run_stage(
            state,
            "compare_existing_radiounet_xai",
            compare_xai_command(python, radio_baseline_xai, radio_physics_xai, xai_radio, "RadioUNet L1", "RadioUNet Physics-L1"),
            [xai_radio],
            fingerprints,
            validator=lambda: validate_json(xai_radio),
        )

        # Freeze a human-readable and machine-readable record, including hashes
        # of every evaluated checkpoint, before any continuation training starts.
        existing_report_markdown = report_dir / "existing_checkpoint_evaluation.md"
        existing_report_json = report_dir / "existing_checkpoint_evaluation.json"
        existing_report_command = [
            python,
            "analysis/build_existing_checkpoint_report.py",
            "--baseline-prediction", os.fspath(baseline_prediction),
            "--physics-prediction", os.fspath(physics_prediction),
            "--prediction-comparison", os.fspath(pred_bp),
            "--baseline-xai", os.fspath(baseline_xai),
            "--physics-xai", os.fspath(physics_xai),
            "--xai-comparison", os.fspath(xai_bp),
            "--radiounet-baseline-prediction", os.fspath(radio_baseline_prediction),
            "--radiounet-physics-prediction", os.fspath(radio_physics_prediction),
            "--radiounet-baseline-xai", os.fspath(radio_baseline_xai),
            "--radiounet-physics-xai", os.fspath(radio_physics_xai),
            "--output", os.fspath(existing_report_markdown),
            "--json-output", os.fspath(existing_report_json),
        ]
        run_stage(
            state,
            "record_existing_checkpoint_evaluation",
            existing_report_command,
            [existing_report_markdown, existing_report_json],
            fingerprints,
            validator=lambda: validate_json(existing_report_json),
        )

        # Prepare weights-only baseline starts after all existing checkpoints are evaluated.
        warm_manifest = output_root / "l1_continuation_warmstarts.json"
        warm_template = "outputs/l1_continuation/warm_starts/seed_{seed}.pth"
        warm_paths = [Path(warm_template.format(seed=seed)) for seed in args.seeds]
        prepare_command = [
            python,
            "scripts/prepare_l1_continuation_warmstarts.py",
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--source-checkpoints",
            *[os.fspath(path) for path in baseline_checkpoints],
            "--output-template",
            warm_template,
            "--split-seed",
            str(args.split_seed),
            "--manifest",
            os.fspath(warm_manifest),
            "--allow-existing",
        ]
        run_stage(
            state,
            "prepare_l1_continuation_warmstarts",
            prepare_command,
            [warm_manifest, *warm_paths],
            fingerprints,
            validator=lambda: validate_json(warm_manifest),
        )

        # One resumable stage per seed prevents a completed seed from being repeated.
        for seed, warm_path in zip(args.seeds, warm_paths):
            final_path = Path("outputs/l1_continuation/checkpoints") / f"seed_{seed}" / "final_model.pth"
            training_command = [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.nproc_per_node}",
                "training/train.py",
                "--config",
                "configs/config_l1_continuation.yaml",
                "--seed",
                str(seed),
                "--split-seed",
                str(args.split_seed),
                "--resume",
                os.fspath(warm_path),
            ]
            run_stage(
                state,
                f"train_l1_continuation_seed_{seed}",
                training_command,
                [final_path],
                fingerprints,
                training_environment,
                lambda final_path=final_path, seed=seed: validate_checkpoint(
                    final_path,
                    seed,
                    args.split_seed,
                    expected_epoch=49,
                    expected_model="restormer",
                ),
            )

        # Evaluate and compare the matched-budget control.
        run_stage(
            state,
            "evaluate_l1_continuation_prediction",
            prediction_command(python, "configs/config_l1_continuation.yaml", args.seeds, control_checkpoints, control_prediction),
            [control_prediction],
            fingerprints,
            eval_environment,
            lambda: validate_json(control_prediction),
        )
        for name, baseline_file, candidate_file, output, baseline_label, candidate_label in (
            ("baseline_vs_continuation", baseline_prediction, control_prediction, pred_bc, "L1", "L1 continuation"),
            ("continuation_vs_physics", control_prediction, physics_prediction, pred_cp, "L1 continuation", "Physics-L1"),
        ):
            run_stage(
                state,
                f"compare_prediction_{name}",
                compare_prediction_command(python, baseline_file, candidate_file, output, baseline_label, candidate_label),
                [output],
                fingerprints,
                validator=lambda output=output: validate_json(output),
            )
        run_stage(
            state,
            "evaluate_l1_continuation_xai",
            xai_command(python, "configs/config_l1_continuation.yaml", args.seeds, control_checkpoints, control_xai, args.xai_samples, args.ig_steps),
            [control_xai],
            fingerprints,
            eval_environment,
            lambda: validate_json(control_xai),
        )
        for name, baseline_file, candidate_file, output, baseline_label, candidate_label in (
            ("baseline_vs_continuation", baseline_xai, control_xai, xai_bc, "L1", "L1 continuation"),
            ("continuation_vs_physics", control_xai, physics_xai, xai_cp, "L1 continuation", "Physics-L1"),
        ):
            run_stage(
                state,
                f"compare_xai_{name}",
                compare_xai_command(python, baseline_file, candidate_file, output, baseline_label, candidate_label),
                [output],
                fingerprints,
                validator=lambda output=output: validate_json(output),
            )

        decision_markdown = report_dir / "follow_up_decision.md"
        decision_json = report_dir / "follow_up_decision.json"
        decision_command = [
            python,
            "analysis/build_revision_decision_report.py",
            "--baseline-prediction", os.fspath(baseline_prediction),
            "--physics-prediction", os.fspath(physics_prediction),
            "--continuation-prediction", os.fspath(control_prediction),
            "--baseline-vs-physics", os.fspath(pred_bp),
            "--baseline-vs-continuation", os.fspath(pred_bc),
            "--continuation-vs-physics", os.fspath(pred_cp),
            "--baseline-xai", os.fspath(baseline_xai),
            "--physics-xai", os.fspath(physics_xai),
            "--continuation-xai", os.fspath(control_xai),
            "--baseline-vs-physics-xai", os.fspath(xai_bp),
            "--baseline-vs-continuation-xai", os.fspath(xai_bc),
            "--continuation-vs-physics-xai", os.fspath(xai_cp),
            "--radiounet-baseline-prediction", os.fspath(radio_baseline_prediction),
            "--radiounet-physics-prediction", os.fspath(radio_physics_prediction),
            "--radiounet-baseline-xai", os.fspath(radio_baseline_xai),
            "--radiounet-physics-xai", os.fspath(radio_physics_xai),
            "--output", os.fspath(decision_markdown),
            "--json-output", os.fspath(decision_json),
        ]
        run_stage(
            state,
            "build_follow_up_decision_report",
            decision_command,
            [decision_markdown, decision_json],
            fingerprints,
            validator=lambda: validate_json(decision_json),
        )
        state.mark_pipeline("completed")
        print("\nPost-training revision pipeline completed successfully", flush=True)
    except Exception as exc:
        state.mark_pipeline("failed", error=exc)
        print(f"Pipeline failed: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        lock_handle.close()


if __name__ == "__main__":
    main()
