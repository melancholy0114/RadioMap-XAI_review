"""Create weights-only warm starts for the matched-budget L1 control.

The ordinary L1 trainer interprets a checkpoint containing optimizer state as
a full resume.  Physics-L1, however, loads only baseline model weights and
starts a fresh optimizer at refinement epoch zero.  This utility removes the
optimizer/scheduler state from each selected baseline checkpoint so the L1
control follows the same warm-start semantics without changing the guarded
training entry point while the existing queue is active.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from model import checkpoint_model_name


_TRAINING_VARIANT = "l1_matched_budget_warm_start"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare weights-only warm starts for L1 continuation"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--source-checkpoints",
        nargs="+",
        required=True,
        help="One baseline checkpoint per seed, in the same order as --seeds",
    )
    parser.add_argument(
        "--output-template",
        required=True,
        help="Destination containing {seed}, for example outputs/.../seed_{seed}.pth",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_source_checkpoint(checkpoint, source_path, seed, split_seed):
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{source_path} does not contain model_state_dict")

    recorded_seed = checkpoint.get("training_seed")
    if recorded_seed is not None and int(recorded_seed) != int(seed):
        raise ValueError(
            f"{source_path} records training_seed={recorded_seed}, expected {seed}"
        )
    recorded_split = checkpoint.get("split_seed")
    if recorded_split is not None and int(recorded_split) != int(split_seed):
        raise ValueError(
            f"{source_path} records split_seed={recorded_split}, expected {split_seed}"
        )
    if checkpoint_model_name(checkpoint) != "restormer":
        raise ValueError(f"{source_path} is not a Restormer checkpoint")


def build_warm_start(checkpoint, source_path, source_sha256, seed, split_seed):
    warm_start = {
        "epoch": -1,
        "model_state_dict": checkpoint["model_state_dict"],
        "training_seed": int(seed),
        "split_seed": int(split_seed),
        "model_name": checkpoint_model_name(checkpoint),
        "training_variant": _TRAINING_VARIANT,
        "warm_start_source": os.fspath(source_path),
        "warm_start_source_sha256": source_sha256,
        "warm_start_source_epoch": int(checkpoint.get("epoch", -1)),
        "warm_start_source_best_val_loss": (
            float(checkpoint["best_val_loss"])
            if checkpoint.get("best_val_loss") is not None
            else None
        ),
    }
    if checkpoint.get("model_config") is not None:
        warm_start["model_config"] = checkpoint["model_config"]
    return warm_start


def prepare_one(source_path, destination, seed, split_seed, allow_existing=False):
    source_path = Path(source_path)
    destination = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {source_path}")

    source_sha256 = sha256_file(source_path)
    if destination.exists():
        existing = _torch_load(destination)
        matches = (
            existing.get("training_variant") == _TRAINING_VARIANT
            and int(existing.get("training_seed", -1)) == int(seed)
            and int(existing.get("split_seed", -1)) == int(split_seed)
            and existing.get("warm_start_source_sha256") == source_sha256
            and "optimizer_state_dict" not in existing
            and "scheduler_state_dict" not in existing
        )
        if matches and allow_existing:
            return {
                "seed": int(seed),
                "source": os.fspath(source_path),
                "source_sha256": source_sha256,
                "destination": os.fspath(destination),
                "status": "reused",
            }
        raise FileExistsError(
            f"Refusing to overwrite warm-start checkpoint: {destination}"
        )

    checkpoint = _torch_load(source_path)
    validate_source_checkpoint(checkpoint, source_path, seed, split_seed)
    warm_start = build_warm_start(
        checkpoint,
        source_path,
        source_sha256,
        seed,
        split_seed,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(warm_start, temporary)
    os.replace(temporary, destination)
    del checkpoint, warm_start

    return {
        "seed": int(seed),
        "source": os.fspath(source_path),
        "source_sha256": source_sha256,
        "destination": os.fspath(destination),
        "destination_sha256": sha256_file(destination),
        "status": "created",
    }


def main():
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if len(args.seeds) != len(args.source_checkpoints):
        raise ValueError("--source-checkpoints must contain one path per seed")
    if "{seed}" not in args.output_template and len(args.seeds) > 1:
        raise ValueError("--output-template must contain {seed}")

    records = []
    for seed, source_path in zip(args.seeds, args.source_checkpoints):
        destination = args.output_template.format(seed=seed)
        print(f"Preparing seed={seed}: {source_path} -> {destination}", flush=True)
        records.append(
            prepare_one(
                source_path,
                destination,
                seed,
                args.split_seed,
                allow_existing=args.allow_existing,
            )
        )

    manifest = {
        "protocol": {
            "training_variant": _TRAINING_VARIANT,
            "semantics": (
                "load model weights only; fresh optimizer and scheduler; "
                "refinement epoch starts at zero"
            ),
            "split_seed": int(args.split_seed),
        },
        "checkpoints": records,
    }
    _atomic_json_dump(manifest, args.manifest)
    print(f"Saved warm-start manifest to {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
