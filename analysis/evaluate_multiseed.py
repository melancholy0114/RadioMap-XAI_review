"""Evaluate seeded checkpoints on one fixed test split.

The output contains run-level prediction metrics, per-map metrics for paired
analysis, and across-seed mean, sample standard deviation, and 95% Student-t
confidence intervals. Metrics are computed on normalized [0, 1] radio maps.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from scipy import stats
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
import yaml

from datasets.radiomapseer_dataset import RadioMapSeerDataset
from model import (
    build_model,
    get_model_name,
    normalize_state_dict,
    validate_checkpoint_model,
)
from utils import get_split_seed


METRIC_KEYS = (
    "global_rmse",
    "global_mae",
    "mean_sample_rmse",
    "mean_sample_mae",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple training seeds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint-template",
        help="Checkpoint path containing {seed}",
    )
    checkpoint_group.add_argument(
        "--checkpoints",
        nargs="+",
        help="Explicit checkpoint paths in the same order as --seeds",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="For example cuda:0 or cpu")
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable CUDA automatic mixed precision during evaluation",
    )
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Smoke-test only; omit for formal full-test-set results",
    )
    return parser.parse_args()


def summary_statistics(values, confidence=0.95):
    """Summarize independent seed-level values with a small-sample t CI."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")

    result = {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_std": None,
        "ci95_low": None,
        "ci95_high": None,
    }
    if array.size < 2:
        return result

    sample_std = float(array.std(ddof=1))
    standard_error = sample_std / math.sqrt(array.size)
    critical_value = float(stats.t.ppf((1.0 + confidence) / 2.0, array.size - 1))
    margin = critical_value * standard_error
    result.update(
        {
            "sample_std": sample_std,
            "ci95_low": float(result["mean"] - margin),
            "ci95_high": float(result["mean"] + margin),
        }
    )
    return result


def resolve_checkpoint_paths(seeds, template=None, checkpoints=None):
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates")
    if template is not None:
        if "{seed}" not in template and len(seeds) > 1:
            raise ValueError("--checkpoint-template must contain {seed}")
        try:
            paths = [template.format(seed=seed) for seed in seeds]
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError("--checkpoint-template may only use the {seed} field") from exc
    else:
        if checkpoints is None or len(checkpoints) != len(seeds):
            raise ValueError("--checkpoints must contain exactly one path per seed")
        paths = checkpoints
    if len(paths) > 1 and len(set(paths)) != len(paths):
        raise ValueError("Each seed must use a distinct checkpoint path")
    return paths


def evaluate_model(model, loader, device, amp_enabled):
    model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_pixels = 0
    sample_rmses = []
    sample_maes = []
    map_totals = defaultdict(lambda: {"squared_error": 0.0, "absolute_error": 0.0, "pixels": 0, "samples": 0})

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with autocast("cuda", enabled=amp_enabled):
                outputs = model(inputs)

            errors = outputs.float() - targets.float()
            per_sample_squared = errors.square().flatten(1).sum(dim=1).double().cpu().numpy()
            per_sample_absolute = errors.abs().flatten(1).sum(dim=1).double().cpu().numpy()
            pixels_per_sample = int(errors[0].numel())

            for map_id, squared_error, absolute_error in zip(
                batch["map_id"], per_sample_squared, per_sample_absolute
            ):
                squared_error = float(squared_error)
                absolute_error = float(absolute_error)
                total_squared_error += squared_error
                total_absolute_error += absolute_error
                total_pixels += pixels_per_sample
                sample_rmses.append(math.sqrt(squared_error / pixels_per_sample))
                sample_maes.append(absolute_error / pixels_per_sample)

                map_total = map_totals[str(map_id)]
                map_total["squared_error"] += squared_error
                map_total["absolute_error"] += absolute_error
                map_total["pixels"] += pixels_per_sample
                map_total["samples"] += 1

            if batch_index % 250 == 0:
                print(f"  evaluated {len(sample_rmses)}/{len(loader.dataset)} samples", flush=True)

    per_map = []
    if total_pixels == 0:
        raise ValueError("The evaluation dataset is empty")
    for map_id in sorted(map_totals, key=lambda value: (len(value), value)):
        totals = map_totals[map_id]
        per_map.append(
            {
                "map_id": map_id,
                "n_samples": int(totals["samples"]),
                "rmse": float(math.sqrt(totals["squared_error"] / totals["pixels"])),
                "mae": float(totals["absolute_error"] / totals["pixels"]),
            }
        )

    return {
        "metrics": {
            "n_samples": len(sample_rmses),
            "n_maps": len(per_map),
            "global_rmse": float(math.sqrt(total_squared_error / total_pixels)),
            "global_mae": float(total_absolute_error / total_pixels),
            "mean_sample_rmse": float(np.mean(sample_rmses)),
            "mean_sample_mae": float(np.mean(sample_maes)),
        },
        "per_map": per_map,
    }


def load_and_evaluate(config, checkpoint_path, requested_seed, split_seed, loader, device, amp_enabled):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    validate_checkpoint_model(checkpoint, model)

    checkpoint_seed = checkpoint.get("training_seed")
    checkpoint_split_seed = checkpoint.get("split_seed")
    if checkpoint_seed is not None and int(checkpoint_seed) != requested_seed:
        raise ValueError(
            f"{checkpoint_path} records training_seed={checkpoint_seed}, "
            f"but it was assigned to seed {requested_seed}"
        )
    if checkpoint_split_seed is not None and int(checkpoint_split_seed) != split_seed:
        raise ValueError(
            f"{checkpoint_path} records split_seed={checkpoint_split_seed}, "
            f"but evaluation uses split_seed={split_seed}"
        )

    model.load_state_dict(normalize_state_dict(checkpoint["model_state_dict"]))
    model_name = get_model_name(model)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    best_val_loss = checkpoint.get("best_val_loss")
    del checkpoint

    evaluated = evaluate_model(model, loader, device, amp_enabled)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "training_seed": int(requested_seed),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "model_name": model_name,
        **evaluated,
    }


def summarize_runs(runs):
    return {
        metric: summary_statistics([run["metrics"][metric] for run in runs])
        for metric in METRIC_KEYS
    }


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and num workers non-negative")
    if args.limit_samples is not None and args.limit_samples < 1:
        raise ValueError("--limit-samples must be positive")

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    split_seed = get_split_seed(config)
    checkpoint_paths = resolve_checkpoint_paths(
        args.seeds,
        template=args.checkpoint_template,
        checkpoints=args.checkpoints,
    )
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    amp_enabled = bool(args.amp and device.type == "cuda")

    data_config = config["data"]
    test_dataset = RadioMapSeerDataset(
        root_dir=data_config["root_dir"],
        gain_method=data_config["gain_method"],
        img_size=data_config["img_size"],
        split="test",
        train_ratio=data_config["train_ratio"],
        val_ratio=data_config["val_ratio"],
        seed=split_seed,
    )
    formal_full_test_set = args.limit_samples is None
    if args.limit_samples is not None:
        test_dataset = Subset(test_dataset, range(min(args.limit_samples, len(test_dataset))))

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    runs = []
    for index, (seed, checkpoint_path) in enumerate(
        zip(args.seeds, checkpoint_paths), start=1
    ):
        print(f"[{index}/{len(args.seeds)}] Evaluating seed={seed}: {checkpoint_path}", flush=True)
        runs.append(
            load_and_evaluate(
                config,
                checkpoint_path,
                seed,
                split_seed,
                loader,
                device,
                amp_enabled,
            )
        )
        print(
            f"  global RMSE={runs[-1]['metrics']['global_rmse']:.6f}, "
            f"MAE={runs[-1]['metrics']['global_mae']:.6f}",
            flush=True,
        )

    result = {
        "protocol": {
            "config": args.config,
            "training_seeds": [int(seed) for seed in args.seeds],
            "split_seed": split_seed,
            "test_split": "test",
            "full_test_set": formal_full_test_set,
            "n_test_samples": len(test_dataset),
            "amp": amp_enabled,
            "metric_definitions": {
                "global_rmse": "sqrt(sum squared pixel error / number of pixels)",
                "global_mae": "sum absolute pixel error / number of pixels",
                "mean_sample_rmse": "arithmetic mean of per-sample pixel RMSE",
                "mean_sample_mae": "arithmetic mean of per-sample pixel MAE",
                "confidence_interval": "two-sided 95% Student-t interval across training seeds",
            },
        },
        "runs": runs,
        "summary": summarize_runs(runs),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved multi-seed evaluation to {output_path}", flush=True)


if __name__ == "__main__":
    main()
