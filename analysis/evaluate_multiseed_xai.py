"""Evaluate physical-alignment diagnostics on fixed samples across seeds.

Unlike the legacy figure script, this evaluator records every selected sample,
checkpoint, attribution method, prior, and metric.  The same deterministic test
samples are reused for every training seed so model variants can be compared by
seed and map rather than only through aggregate bar heights.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import yaml

from analysis.evaluate_multiseed import resolve_checkpoint_paths, summary_statistics
from datasets.radiomapseer_dataset import RadioMapSeerDataset
from explanation import GradCAM, IntegratedGradients, OcclusionSensitivity
from metrics import PhysicalAlignmentScore
from model import (
    build_model,
    get_model_name,
    normalize_state_dict,
    validate_checkpoint_model,
)
from priors import (
    compute_directional_mask,
    compute_los_mask_fast,
    compute_obstruction_mask,
)
from utils import get_evaluation_seed, get_split_seed


METHODS = ("integrated_gradients", "grad_cam", "occlusion_sensitivity")
PRIORS = ("los", "obstruction", "directional")
ALIGNMENT_METRICS = (
    "iou",
    "soft_iou",
    "pearson_corr",
    "spearman_corr",
    "precision",
    "recall",
    "center_of_mass_distance",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-sample XAI diagnostics across training seeds"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint-template")
    checkpoint_group.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["integrated_gradients"])
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--top-k-percent", type=float, default=20.0)
    parser.add_argument("--ig-steps", type=int, default=None)
    parser.add_argument("--occlusion-window", type=int, default=None)
    parser.add_argument("--occlusion-stride", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def select_sample_indices(dataset_size, n_samples, evaluation_seed):
    if dataset_size < 1:
        raise ValueError("test dataset is empty")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    n_samples = min(int(n_samples), int(dataset_size))
    rng = np.random.default_rng(int(evaluation_seed))
    return sorted(int(index) for index in rng.choice(dataset_size, size=n_samples, replace=False))


def build_explainers(model, device, methods):
    explainers = {}
    if "integrated_gradients" in methods:
        explainers["integrated_gradients"] = IntegratedGradients(model, device)
    if "grad_cam" in methods:
        explainers["grad_cam"] = GradCAM(model, device=device)
    if "occlusion_sensitivity" in methods:
        explainers["occlusion_sensitivity"] = OcclusionSensitivity(model, device)
    return explainers


def explain_sample(method, explainer, inputs, settings):
    if method == "integrated_gradients":
        return explainer.explain_sample(inputs, n_steps=settings["ig_steps"])
    if method == "grad_cam":
        return explainer.explain_sample(inputs)
    if method == "occlusion_sensitivity":
        return explainer.explain_sample(
            inputs,
            window_size=settings["occlusion_window"],
            stride=settings["occlusion_stride"],
        )
    raise ValueError(f"Unsupported attribution method: {method}")


def summarize_samples(samples, methods):
    summary = {}
    for method in methods:
        summary[method] = {}
        for prior in PRIORS:
            summary[method][prior] = {
                metric: summary_statistics(
                    [
                        sample["alignment"][method][prior][metric]
                        for sample in samples
                    ]
                )
                for metric in ALIGNMENT_METRICS
            }
    return summary


def summarize_across_seeds(runs, methods):
    summary = {}
    for method in methods:
        summary[method] = {}
        for prior in PRIORS:
            summary[method][prior] = {}
            for metric in ALIGNMENT_METRICS:
                seed_means = [
                    run["summary"][method][prior][metric]["mean"]
                    for run in runs
                ]
                summary[method][prior][metric] = summary_statistics(seed_means)
    return summary


def evaluate_checkpoint(
    config,
    checkpoint_path,
    requested_seed,
    split_seed,
    dataset,
    sample_indices,
    methods,
    settings,
    device,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(config["model"]).to(device)
    checkpoint = _torch_load(checkpoint_path, device)
    validate_checkpoint_model(checkpoint, model)
    recorded_seed = checkpoint.get("training_seed")
    if recorded_seed is not None and int(recorded_seed) != int(requested_seed):
        raise ValueError(
            f"{checkpoint_path} records training_seed={recorded_seed}, "
            f"expected {requested_seed}"
        )
    recorded_split = checkpoint.get("split_seed")
    if recorded_split is not None and int(recorded_split) != int(split_seed):
        raise ValueError(
            f"{checkpoint_path} records split_seed={recorded_split}, expected {split_seed}"
        )
    model.load_state_dict(normalize_state_dict(checkpoint["model_state_dict"]))
    model.eval()
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    best_val_loss = checkpoint.get("best_val_loss")
    del checkpoint

    explainers = build_explainers(model, device, methods)
    pas = PhysicalAlignmentScore()
    records = []
    for position, dataset_index in enumerate(sample_indices, start=1):
        sample = dataset[dataset_index]
        inputs = sample["input"].unsqueeze(0).to(device)
        target = sample["target"][0].numpy()
        building = sample["building"].numpy()
        tx_position = sample["tx_position"].numpy()

        with torch.no_grad():
            prediction = model(inputs)[0, 0].float().cpu().numpy()
        error = prediction - target
        priors = {
            "los": compute_los_mask_fast(building, tx_position),
            "obstruction": compute_obstruction_mask(building, tx_position),
            "directional": compute_directional_mask(
                tx_position,
                img_size=building.shape[0],
            ),
        }

        alignment = {}
        for method in methods:
            model.zero_grad(set_to_none=True)
            explanation = explain_sample(method, explainers[method], inputs, settings)
            alignment[method] = pas.compute_multi_prior_extended(
                explanation,
                priors,
                top_k_percent=settings["top_k_percent"],
            )

        records.append(
            {
                "dataset_index": int(dataset_index),
                "map_id": str(sample["map_id"]),
                "tx_idx": int(sample["tx_idx"]),
                "prediction_rmse": float(np.sqrt(np.mean(error ** 2))),
                "prediction_mae": float(np.mean(np.abs(error))),
                "alignment": alignment,
            }
        )
        print(
            f"  seed={requested_seed}: explained {position}/{len(sample_indices)} "
            f"(map={sample['map_id']}, tx={int(sample['tx_idx'])})",
            flush=True,
        )

    run = {
        "training_seed": int(requested_seed),
        "checkpoint": os.fspath(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "model_name": get_model_name(model),
        "samples": records,
        "summary": summarize_samples(records, methods),
    }
    del explainers, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run


def main():
    args = parse_args()
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods must not contain duplicates")
    if not 0 < args.top_k_percent <= 100:
        raise ValueError("--top-k-percent must be in (0, 100]")

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    checkpoint_paths = resolve_checkpoint_paths(
        args.seeds,
        template=args.checkpoint_template,
        checkpoints=args.checkpoints,
    )
    split_seed = get_split_seed(config)
    evaluation_seed = (
        get_evaluation_seed(config)
        if args.evaluation_seed is None
        else int(args.evaluation_seed)
    )
    explain_config = config["explainability"]
    settings = {
        "ig_steps": int(args.ig_steps or explain_config["ig_steps"]),
        "occlusion_window": int(
            args.occlusion_window or explain_config["occlusion_window"]
        ),
        "occlusion_stride": int(
            args.occlusion_stride or explain_config["occlusion_stride"]
        ),
        "top_k_percent": float(args.top_k_percent),
    }
    if min(
        settings["ig_steps"],
        settings["occlusion_window"],
        settings["occlusion_stride"],
    ) < 1:
        raise ValueError("attribution step/window/stride settings must be positive")

    data = config["data"]
    dataset = RadioMapSeerDataset(
        root_dir=data["root_dir"],
        gain_method=data["gain_method"],
        img_size=data["img_size"],
        split="test",
        train_ratio=data["train_ratio"],
        val_ratio=data["val_ratio"],
        seed=split_seed,
    )
    n_samples = int(args.n_samples or explain_config["num_samples"])
    sample_indices = select_sample_indices(len(dataset), n_samples, evaluation_seed)
    sample_identities = [
        {
            "dataset_index": index,
            "map_id": str(dataset[index]["map_id"]),
            "tx_idx": int(dataset[index]["tx_idx"]),
        }
        for index in sample_indices
    ]
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    runs = []
    for index, (seed, checkpoint_path) in enumerate(
        zip(args.seeds, checkpoint_paths), start=1
    ):
        print(
            f"[{index}/{len(args.seeds)}] XAI evaluation seed={seed}: {checkpoint_path}",
            flush=True,
        )
        runs.append(
            evaluate_checkpoint(
                config,
                checkpoint_path,
                seed,
                split_seed,
                dataset,
                sample_indices,
                args.methods,
                settings,
                device,
            )
        )

    result = {
        "protocol": {
            "config": args.config,
            "training_seeds": [int(seed) for seed in args.seeds],
            "split_seed": int(split_seed),
            "evaluation_seed": int(evaluation_seed),
            "test_split": "test",
            "n_test_samples_available": len(dataset),
            "n_explanation_samples": len(sample_indices),
            "sample_selection": "without replacement using numpy PCG64; sorted indices",
            "sample_identities": sample_identities,
            "methods": list(args.methods),
            "settings": settings,
            "scalar_attribution_target": "sum of all output pixels in output channel 0",
            "integrated_gradients_channel_aggregation": "sum of absolute attribution over input channels",
            "alignment_summary": (
                "sample means within each seed, followed by mean/sample std/95% "
                "Student-t CI across training-seed means"
            ),
        },
        "runs": runs,
        "across_seed_summary": summarize_across_seeds(runs, args.methods),
    }
    _atomic_json_dump(result, args.output)
    print(json.dumps(result["across_seed_summary"], indent=2), flush=True)
    print(f"Saved multi-seed XAI evaluation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
