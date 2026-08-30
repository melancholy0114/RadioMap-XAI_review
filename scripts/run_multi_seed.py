"""Run independent seeded training jobs sequentially.

Each child process receives ``--seed``, so the training entry point stores its
outputs under a separate ``seed_<N>`` directory. The same launcher works for
Restormer and RadioUNet because the backbone remains config-selected.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a controlled multi-seed experiment")
    parser.add_argument("--trainer", choices=("l1", "physics"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Fixed data split seed; defaults to data.split_seed in the config",
    )
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated physical GPU IDs exposed to every sequential run",
    )
    parser.add_argument(
        "--resume-template",
        default=None,
        help="Optional checkpoint path containing {seed}, e.g. outputs/checkpoints/seed_{seed}/best_model.pth",
    )
    parser.add_argument(
        "--full-resume",
        action="store_true",
        help="Restore full Physics-L1 state (valid only with --trainer physics)",
    )
    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--smoke-test-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without starting training",
    )
    return parser.parse_args()


def validate_args(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be positive")
    if not 0 < args.subset <= 1:
        raise ValueError("--subset must be in (0, 1]")
    if args.full_resume and args.trainer != "physics":
        raise ValueError("--full-resume is only valid with --trainer physics")
    if args.gpus is not None:
        gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
        if len(gpu_ids) < args.nproc_per_node:
            raise ValueError(
                f"{args.nproc_per_node} processes requested, but --gpus exposes only "
                f"{len(gpu_ids)} device(s)"
            )


def resolve_resume_path(template, seed):
    if template is None:
        return None
    try:
        return template.format(seed=seed)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError("--resume-template may only use the {seed} field") from exc


def build_command(args, seed):
    training_script = (
        "training/train.py" if args.trainer == "l1" else "training/train_physics.py"
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        training_script,
        "--config",
        args.config,
        "--seed",
        str(seed),
    ]
    if args.split_seed is not None:
        command.extend(["--split-seed", str(args.split_seed)])
    if args.subset != 1.0:
        command.extend(["--subset", str(args.subset)])
    if args.smoke_test_batches is not None:
        command.extend(["--smoke-test-batches", str(args.smoke_test_batches)])
    if args.log_interval is not None:
        command.extend(["--log-interval", str(args.log_interval)])
    resume_path = resolve_resume_path(args.resume_template, seed)
    if resume_path is not None:
        command.extend(["--resume", resume_path])
    if args.full_resume:
        command.append("--full-resume")
    return command


def main():
    args = parse_args()
    validate_args(args)

    environment = os.environ.copy()
    if args.gpus is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpus

    print(
        f"Multi-seed protocol: training seeds={args.seeds}, "
        f"split seed={args.split_seed if args.split_seed is not None else 'config'}",
        flush=True,
    )
    for index, seed in enumerate(args.seeds, start=1):
        command = build_command(args, seed)
        print(f"\n[{index}/{len(args.seeds)}] seed={seed}", flush=True)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(
                command,
                cwd=_PROJECT_ROOT,
                env=environment,
                check=True,
            )


if __name__ == "__main__":
    main()
