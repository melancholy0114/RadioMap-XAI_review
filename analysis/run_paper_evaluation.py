"""
Reproducible subset evaluation for the paper draft.

This script computes:
  - ID vs OOD prediction metrics on random subsets
  - PAS against physical priors on OOD explanation samples
  - Faithfulness deletion scores and AUC
  - Stability under input perturbations
  - In-domain / cross-domain explanation consistency
  - Explanation drift vs prediction error

The quantitative explanation metrics are reported for Integrated Gradients.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from torch.amp import autocast

from datasets.radiomapseer_dataset import RadioMapSeerDataset
from inference.infer import load_model
from explanation import IntegratedGradients
from metrics import Faithfulness, PhysicalAlignmentScore, Stability, Consistency
from priors import compute_los_mask_fast, compute_obstruction_mask, compute_directional_mask
from utils import get_evaluation_seed, get_split_seed, get_training_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper evaluation subsets")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--save_json", type=str, default="outputs/logs/paper_eval_results.json")
    parser.add_argument("--id_samples", type=int, default=1000)
    parser.add_argument("--ood_samples", type=int, default=1000)
    parser.add_argument("--explain_samples", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ig_steps", type=int, default=None)
    parser.add_argument("--stability_perturbations", type=int, default=None)
    parser.add_argument("--faithfulness_auc_steps", type=int, default=20)
    return parser.parse_args()


def compute_rmse(prediction, target):
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def compute_prediction_metrics(model, dataset, indices, device, batch_size=8, num_workers=0):
    subset = Subset(dataset, [int(i) for i in indices])
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    rmses = []
    maes = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            with autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(inputs)

            preds = outputs[:, 0].detach().cpu().numpy()
            gts = targets[:, 0].detach().cpu().numpy()

            for pred, gt in zip(preds, gts):
                rmses.append(float(np.sqrt(np.mean((pred - gt) ** 2))))
                maes.append(float(np.mean(np.abs(pred - gt))))

    return {
        "n_samples": len(indices),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
    }


def explain_and_score_sample(
    model,
    explainer,
    faithfulness,
    stability,
    pas_evaluator,
    sample,
    config,
    device,
    ig_steps,
    stability_perturbations,
    faithfulness_auc_steps,
):
    inputs = sample["input"].unsqueeze(0).to(device)
    target = sample["target"][0].numpy()
    building = sample["building"].numpy()
    tx_pos = sample["tx_position"].numpy()

    with torch.no_grad(), autocast("cuda", enabled=device.type == "cuda"):
        prediction = model(inputs)[0, 0].detach().cpu().numpy()

    rmse = compute_rmse(prediction, target)

    explanation = explainer.explain_sample(inputs, n_steps=ig_steps)

    priors = {
        "los": compute_los_mask_fast(building, tx_pos),
        "obstruction": compute_obstruction_mask(building, tx_pos),
        "directional": compute_directional_mask(tx_pos, img_size=building.shape[0]),
    }
    pas_scores = pas_evaluator.compute_multi_prior(explanation, priors, top_k_percent=20)

    faithfulness_scores = faithfulness.compute(
        inputs,
        explanation,
        top_k_percentages=config["metrics"]["faithfulness_top_k"],
    )
    faithfulness_auc, _ = faithfulness.compute_auc(
        inputs,
        explanation,
        n_steps=faithfulness_auc_steps,
    )

    stability_score, stability_details = stability.compute(
        sample["input"],
        n_perturbations=stability_perturbations,
        noise_std=config["metrics"]["stability_noise_std"],
    )
    ssim_score, _ = stability.compute_ssim_stability(
        sample["input"],
        n_perturbations=stability_perturbations,
        noise_std=config["metrics"]["stability_noise_std"],
    )

    return {
        "map_id": sample["map_id"],
        "tx_idx": int(sample["tx_idx"]),
        "rmse": rmse,
        "building": building,
        "explanation": explanation,
        "pas": {k: float(v["pas"]) for k, v in pas_scores.items()},
        "faithfulness": {str(k): float(v) for k, v in faithfulness_scores.items()},
        "faithfulness_auc": float(faithfulness_auc),
        "stability_score": float(stability_score),
        "stability_l2_mean": float(stability_details["mean_l2_distance"]),
        "stability_ssim": float(ssim_score),
    }


def summarize_explanation_results(records):
    top_ks = sorted(records[0]["faithfulness"].keys(), key=lambda x: int(x))

    summary = {
        "n_samples": len(records),
        "rmse_mean": float(np.mean([r["rmse"] for r in records])),
        "rmse_std": float(np.std([r["rmse"] for r in records])),
        "pas_mean": {},
        "pas_std": {},
        "faithfulness_mean": {},
        "faithfulness_std": {},
        "faithfulness_auc_mean": float(np.mean([r["faithfulness_auc"] for r in records])),
        "faithfulness_auc_std": float(np.std([r["faithfulness_auc"] for r in records])),
        "stability_mean": float(np.mean([r["stability_score"] for r in records])),
        "stability_std": float(np.std([r["stability_score"] for r in records])),
        "stability_l2_mean": float(np.mean([r["stability_l2_mean"] for r in records])),
        "stability_ssim_mean": float(np.mean([r["stability_ssim"] for r in records])),
    }

    for prior_name in records[0]["pas"]:
        values = [r["pas"][prior_name] for r in records]
        summary["pas_mean"][prior_name] = float(np.mean(values))
        summary["pas_std"][prior_name] = float(np.std(values))

    for k in top_ks:
        values = [r["faithfulness"][k] for r in records]
        summary["faithfulness_mean"][k] = float(np.mean(values))
        summary["faithfulness_std"][k] = float(np.std(values))

    return summary


def compute_within_domain_consistency(consistency, records):
    explanation_maps = [r["explanation"] for r in records]
    pairwise_corr, pairwise_details = consistency.compute_pairwise(explanation_maps)
    spatial_consistency = consistency.compute_spatial_consistency(explanation_maps)

    return {
        "pairwise_correlation": float(pairwise_corr),
        "pairwise_std": float(pairwise_details["std_correlation"]),
        "spatial_consistency": float(spatial_consistency),
    }


def compute_cross_domain_results(consistency, id_records, ood_records):
    id_explanations = [r["explanation"] for r in id_records]
    ood_explanations = [r["explanation"] for r in ood_records]
    drift, details = consistency.compute_cross_domain_consistency(
        id_explanations,
        ood_explanations,
    )

    return {
        "l2_drift": float(drift),
        "cosine_similarity": float(details["cosine_similarity"]),
        "correlation": float(details["correlation"]),
        "n_id": int(details["n_id"]),
        "n_ood": int(details["n_ood"]),
    }


def compute_drift_vs_error(id_records, ood_records):
    drifts = []
    errors = []

    for ood in ood_records:
        best_similarity = -2.0
        best_id_explanation = None
        ood_building_flat = ood["building"].flatten()

        for id_record in id_records:
            sim = np.corrcoef(ood_building_flat, id_record["building"].flatten())[0, 1]
            if np.isnan(sim):
                sim = -1.0
            if sim > best_similarity:
                best_similarity = sim
                best_id_explanation = id_record["explanation"]

        drift = float(np.sqrt(np.mean((best_id_explanation - ood["explanation"]) ** 2)))
        drifts.append(drift)
        errors.append(ood["rmse"])

    correlation = float(np.corrcoef(drifts, errors)[0, 1]) if len(drifts) > 1 else 0.0
    return {
        "correlation": correlation,
        "mean_drift": float(np.mean(drifts)),
        "std_drift": float(np.std(drifts)),
        "mean_ood_rmse": float(np.mean(errors)),
    }


def strip_arrays(records):
    stripped = []
    for record in records:
        stripped.append({
            "map_id": record["map_id"],
            "tx_idx": record["tx_idx"],
            "rmse": record["rmse"],
            "pas": record["pas"],
            "faithfulness": record["faithfulness"],
            "faithfulness_auc": record["faithfulness_auc"],
            "stability_score": record["stability_score"],
            "stability_l2_mean": record["stability_l2_mean"],
            "stability_ssim": record["stability_ssim"],
        })
    return stripped


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_seed = get_evaluation_seed(config)
    split_seed = get_split_seed(config)
    rng = np.random.default_rng(evaluation_seed)

    model = load_model(config, args.checkpoint, device)
    ig_steps = args.ig_steps or config["explainability"]["ig_steps"]
    stability_perturbations = (
        args.stability_perturbations or config["metrics"]["stability_num_samples"]
    )

    train_dataset = RadioMapSeerDataset(
        root_dir=config["data"]["root_dir"],
        gain_method=config["data"]["gain_method"],
        split="train",
        seed=split_seed,
    )
    test_dataset = RadioMapSeerDataset(
        root_dir=config["data"]["root_dir"],
        gain_method=config["data"]["gain_method"],
        split="test",
        seed=split_seed,
    )

    id_metric_indices = rng.choice(len(train_dataset), size=min(args.id_samples, len(train_dataset)), replace=False)
    ood_metric_indices = rng.choice(len(test_dataset), size=min(args.ood_samples, len(test_dataset)), replace=False)
    id_explain_indices = rng.choice(len(train_dataset), size=min(args.explain_samples, len(train_dataset)), replace=False)
    ood_explain_indices = rng.choice(len(test_dataset), size=min(args.explain_samples, len(test_dataset)), replace=False)

    print("Running ID/OOD prediction metrics...")
    id_metrics = compute_prediction_metrics(
        model,
        train_dataset,
        id_metric_indices,
        device,
        args.batch_size,
        args.num_workers,
    )
    ood_metrics = compute_prediction_metrics(
        model,
        test_dataset,
        ood_metric_indices,
        device,
        args.batch_size,
        args.num_workers,
    )

    print("Running explanation-centric metrics on subsets...")
    explainer = IntegratedGradients(model, device)
    faithfulness = Faithfulness(model, device)
    stability = Stability(explainer, model, device)
    pas_evaluator = PhysicalAlignmentScore()
    consistency = Consistency()

    id_records = []
    for idx in id_explain_indices:
        sample = train_dataset[int(idx)]
        id_records.append(
            explain_and_score_sample(
                model,
                explainer,
                faithfulness,
                stability,
                pas_evaluator,
                sample,
                config,
                device,
                ig_steps,
                stability_perturbations,
                args.faithfulness_auc_steps,
            )
        )

    ood_records = []
    for idx in ood_explain_indices:
        sample = test_dataset[int(idx)]
        ood_records.append(
            explain_and_score_sample(
                model,
                explainer,
                faithfulness,
                stability,
                pas_evaluator,
                sample,
                config,
                device,
                ig_steps,
                stability_perturbations,
                args.faithfulness_auc_steps,
            )
        )

    print("Aggregating results...")
    results = {
        "protocol": {
            "training_seed": get_training_seed(config),
            "split_seed": split_seed,
            "evaluation_seed": evaluation_seed,
            "id_samples_prediction": int(len(id_metric_indices)),
            "ood_samples_prediction": int(len(ood_metric_indices)),
            "id_samples_explanations": int(len(id_explain_indices)),
            "ood_samples_explanations": int(len(ood_explain_indices)),
            "explainer": "IntegratedGradients",
            "ig_steps": int(ig_steps),
            "stability_perturbations": int(stability_perturbations),
            "faithfulness_auc_steps": int(args.faithfulness_auc_steps),
        },
        "prediction": {
            "id": id_metrics,
            "ood": ood_metrics,
            "ood_minus_id_rmse": float(ood_metrics["rmse_mean"] - id_metrics["rmse_mean"]),
            "ood_minus_id_mae": float(ood_metrics["mae_mean"] - id_metrics["mae_mean"]),
        },
        "explanations": {
            "id_summary": summarize_explanation_results(id_records),
            "ood_summary": summarize_explanation_results(ood_records),
            "id_consistency": compute_within_domain_consistency(consistency, id_records),
            "ood_consistency": compute_within_domain_consistency(consistency, ood_records),
            "cross_domain": compute_cross_domain_results(consistency, id_records, ood_records),
            "drift_vs_error": compute_drift_vs_error(id_records, ood_records),
            "id_samples": strip_arrays(id_records),
            "ood_samples": strip_arrays(ood_records),
        },
    }

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({
        "id_rmse": results["prediction"]["id"]["rmse_mean"],
        "ood_rmse": results["prediction"]["ood"]["rmse_mean"],
        "ood_pas_obstruction": results["explanations"]["ood_summary"]["pas_mean"]["obstruction"],
        "ood_stability": results["explanations"]["ood_summary"]["stability_mean"],
        "cross_domain_drift": results["explanations"]["cross_domain"]["l2_drift"],
        "drift_error_corr": results["explanations"]["drift_vs_error"]["correlation"],
    }, indent=2))
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    main()
