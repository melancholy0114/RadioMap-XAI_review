"""
Enhanced XAI Evaluation: runs all new analyses on baseline and improved models.

Analyses:
  1. Full faithfulness protocol (deletion + insertion curves)
  2. Extended physical alignment (multi-metric PAS)
  3. Cross-method agreement (IG vs Grad-CAM vs Occlusion)
  4. Threshold sensitivity
  5. Range-conditioned XAI
  6. LoS/NLoS explanation split

Usage:
    conda run -n pytorch python3 scripts/run_xai_enhanced_evaluation.py
    conda run -n pytorch python3 scripts/run_xai_enhanced_evaluation.py --checkpoint outputs/improved_checkpoints/best_model.pth --tag improved
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from datasets.radiomapseer_dataset import RadioMapSeerDataset
from inference.infer import load_model
from explanation import IntegratedGradients, GradCAM, OcclusionSensitivity
from metrics import Faithfulness, PhysicalAlignmentScore
from priors import compute_los_mask_fast, compute_obstruction_mask, compute_directional_mask
from analysis.cross_method_agreement import CrossMethodAgreement
from analysis.threshold_sensitivity import ThresholdSensitivity
from analysis.range_conditioned_xai import RangeConditionedXAI
from analysis.los_nlos_explanation_split import LoSNLoSExplanationSplit
from utils import get_evaluation_seed, get_split_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced XAI evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--tag", type=str, default="baseline",
                        help="Tag for output files (baseline or improved)")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of test samples to evaluate")
    parser.add_argument("--ig_steps", type=int, default=50)
    parser.add_argument("--faithfulness_steps", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="outputs/enhanced_xai")
    return parser.parse_args()


def get_sample_data(model, sample, device):
    """Get prediction and RMSE for a sample."""
    inputs = sample["input"].unsqueeze(0).to(device)
    target = sample["target"][0].numpy()
    building = sample["building"].numpy()
    tx_pos = sample["tx_position"].numpy()

    with torch.no_grad():
        pred = model(inputs)[0, 0].cpu().numpy()

    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    return inputs, pred, target, building, tx_pos, rmse


def get_priors(building, tx_pos, img_size=256):
    """Compute physical priors."""
    return {
        "los": compute_los_mask_fast(building, tx_pos),
        "obstruction": compute_obstruction_mask(building, tx_pos),
        "directional": compute_directional_mask(tx_pos, img_size=img_size),
    }


def run_analysis(model, dataset, device, args, config):
    """Run all enhanced XAI analyses."""
    rng = np.random.default_rng(get_evaluation_seed(config))
    n = min(args.n_samples, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False)

    # Initialize explainers
    ig = IntegratedGradients(model, device)
    grad_cam = GradCAM(model, device=device)
    occlusion = OcclusionSensitivity(model, device)
    faithfulness = Faithfulness(model, device)
    pas_evaluator = PhysicalAlignmentScore()
    cross_method = CrossMethodAgreement()
    threshold_sens = ThresholdSensitivity()
    range_xai = RangeConditionedXAI()
    los_nlos = LoSNLoSExplanationSplit(model, device)

    # Storage
    sample_results = []
    all_deletion_curves = []
    all_insertion_curves = []
    all_cross_method = []
    all_threshold_pas = []
    all_extended_pas = []
    all_range_analysis = []
    all_los_nlos = []
    rmses = []

    print(f"Running enhanced XAI evaluation on {n} samples (tag={args.tag})...")

    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]
        inputs, pred, target, building, tx_pos, rmse = get_sample_data(model, sample, device)
        rmses.append(rmse)

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{n}] RMSE={rmse:.4f}")

        # 1. Generate all explanation methods
        ig_map = ig.explain_sample(inputs, n_steps=args.ig_steps)
        cam_map = grad_cam.explain_sample(inputs)
        occ_map = occlusion.explain_sample(inputs, window_size=16, stride=8)

        explanations = {"ig": ig_map, "grad_cam": cam_map, "occlusion": occ_map}
        priors = get_priors(building, tx_pos)

        # 2. Faithfulness protocol (deletion + insertion)
        faith_result = faithfulness.compute_full_protocol(
            inputs, ig_map, n_steps=args.faithfulness_steps
        )
        all_deletion_curves.append(faith_result["deletion_curve"])
        all_insertion_curves.append(faith_result["insertion_curve"])

        # 3. Extended PAS
        ext_pas = pas_evaluator.compute_multi_prior_extended(ig_map, priors, top_k_percent=20)
        all_extended_pas.append(ext_pas)

        # 4. Cross-method agreement
        cross_result = cross_method.compute_all_pairs(explanations, top_k_percent=20)
        all_cross_method.append(cross_result)

        # 5. Threshold sensitivity
        thresh_result = threshold_sens.compute_multi_prior_at_thresholds(
            ig_map, priors, thresholds=[5, 10, 20, 30, 40]
        )
        all_threshold_pas.append(thresh_result)

        # 6. Range-conditioned analysis
        range_result = range_xai.compute_full_analysis(ig_map, priors, tx_pos, img_size=256)
        all_range_analysis.append(range_result)

        # 7. LoS/NLoS split
        los_nlos_result = los_nlos.compute_full_analysis(inputs, ig_map, priors["los"], building)
        all_los_nlos.append(los_nlos_result)

        sample_results.append({
            "map_id": sample["map_id"],
            "tx_idx": int(sample["tx_idx"]),
            "rmse": rmse,
        })

    return {
        "sample_results": sample_results,
        "rmses": rmses,
        "deletion_curves": all_deletion_curves,
        "insertion_curves": all_insertion_curves,
        "extended_pas": all_extended_pas,
        "cross_method": all_cross_method,
        "threshold_pas": all_threshold_pas,
        "range_analysis": all_range_analysis,
        "los_nlos": all_los_nlos,
    }


def aggregate_results(raw):
    """Aggregate raw results into summary statistics."""
    rmses = raw["rmses"]
    n = len(rmses)

    # 1. Faithfulness summary
    deletion_aucs = []
    insertion_aucs = []
    deletion_monos = []
    insertion_monos = []
    for del_curve, ins_curve in zip(raw["deletion_curves"], raw["insertion_curves"]):
        del_errors = [e for _, e in del_curve]
        ins_errors = [e for _, e in ins_curve]
        deletion_aucs.append(float(np.trapz(del_errors, dx=1.0 / max(len(del_errors) - 1, 1))))
        insertion_aucs.append(float(np.trapz(ins_errors, dx=1.0 / max(len(ins_errors) - 1, 1))))
        del_mono = sum(1 for i in range(1, len(del_errors)) if del_errors[i] >= del_errors[i-1])
        deletion_monos.append(del_mono / (len(del_errors) - 1))
        ins_mono = sum(1 for i in range(1, len(ins_errors)) if ins_errors[i] <= ins_errors[i-1])
        insertion_monos.append(ins_mono / (len(ins_errors) - 1))

    faithfulness_summary = {
        "deletion_auc": {"mean": float(np.mean(deletion_aucs)), "std": float(np.std(deletion_aucs))},
        "insertion_auc": {"mean": float(np.mean(insertion_aucs)), "std": float(np.std(insertion_aucs))},
        "deletion_monotonicity": {"mean": float(np.mean(deletion_monos)), "std": float(np.std(deletion_monos))},
        "insertion_monotonicity": {"mean": float(np.mean(insertion_monos)), "std": float(np.std(insertion_monos))},
    }

    # 2. Extended PAS summary
    pas_metrics = ["iou", "soft_iou", "pearson_corr", "spearman_corr", "precision", "recall", "center_of_mass_distance"]
    priors = list(raw["extended_pas"][0].keys())
    ext_pas_summary = {}
    for prior in priors:
        ext_pas_summary[prior] = {}
        for metric in pas_metrics:
            values = [r[prior][metric] for r in raw["extended_pas"]]
            ext_pas_summary[prior][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    # 3. Cross-method summary
    cross_summary = CrossMethodAgreement().compute_summary(raw["cross_method"])
    cross_grouped = CrossMethodAgreement().compute_grouped_summary(
        raw["cross_method"], rmses
    )

    # 4. Threshold sensitivity
    thresh_robustness = ThresholdSensitivity().compute_robustness_summary(raw["threshold_pas"])

    # 5. Range-conditioned summary
    range_summary = {"intensity": {}, "pas_per_range": {}}
    range_names = list(raw["range_analysis"][0]["intensity"].keys())
    for rn in range_names:
        range_summary["intensity"][rn] = {
            "mean_intensity": {
                "mean": float(np.mean([r["intensity"][rn]["mean_intensity"] for r in raw["range_analysis"]])),
                "std": float(np.std([r["intensity"][rn]["mean_intensity"] for r in raw["range_analysis"]])),
            },
            "intensity_fraction": {
                "mean": float(np.mean([r["intensity"][rn]["intensity_fraction"] for r in raw["range_analysis"]])),
                "std": float(np.std([r["intensity"][rn]["intensity_fraction"] for r in raw["range_analysis"]])),
            },
        }

    for prior in priors:
        range_summary["pas_per_range"][prior] = {}
        for rn in range_names:
            prec_vals = [r["pas_per_range"][prior][rn]["precision"] for r in raw["range_analysis"]]
            rec_vals = [r["pas_per_range"][prior][rn]["recall"] for r in raw["range_analysis"]]
            range_summary["pas_per_range"][prior][rn] = {
                "precision": {"mean": float(np.mean(prec_vals)), "std": float(np.std(prec_vals))},
                "recall": {"mean": float(np.mean(rec_vals)), "std": float(np.std(rec_vals))},
            }

    # 6. LoS/NLoS summary
    los_nlos_summary = {
        "los_mass_fraction": {
            "mean": float(np.mean([r["mass_split"]["los_mass_fraction"] for r in raw["los_nlos"]])),
            "std": float(np.std([r["mass_split"]["los_mass_fraction"] for r in raw["los_nlos"]])),
        },
        "nlos_mass_fraction": {
            "mean": float(np.mean([r["mass_split"]["nlos_mass_fraction"] for r in raw["los_nlos"]])),
            "std": float(np.std([r["mass_split"]["nlos_mass_fraction"] for r in raw["los_nlos"]])),
        },
        "deletion_impact": {},
    }
    for region in ["los", "nlos", "boundary", "building", "open_nlos"]:
        vals = [r["deletion_impact"][region]["deletion_error"] for r in raw["los_nlos"]]
        los_nlos_summary["deletion_impact"][region] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    return {
        "n_samples": n,
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "faithfulness": faithfulness_summary,
        "extended_pas": ext_pas_summary,
        "cross_method_agreement": cross_summary,
        "cross_method_grouped": cross_grouped,
        "threshold_sensitivity": thresh_robustness,
        "range_conditioned": range_summary,
        "los_nlos": los_nlos_summary,
    }


def generate_figures(summary, raw, save_dir, tag):
    """Generate all enhanced XAI figures."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Faithfulness curves
    plot_faithfulness_curves(raw, save_dir, tag)

    # 2. Extended PAS bar chart
    plot_extended_pas(summary, save_dir, tag)

    # 3. Cross-method agreement
    plot_cross_method(summary, save_dir, tag)

    # 4. Threshold sensitivity
    plot_threshold_sensitivity(summary, save_dir, tag)

    # 5. Range-conditioned analysis
    plot_range_conditioned(summary, save_dir, tag)

    # 6. LoS/NLoS split
    plot_los_nlos(summary, save_dir, tag)


def plot_faithfulness_curves(raw, save_dir, tag):
    """Plot average deletion and insertion curves."""
    n_steps = len(raw["deletion_curves"][0])
    fractions = [i / (n_steps - 1) for i in range(n_steps)]

    # Average curves
    del_mean = np.mean([[e for _, e in curve] for curve in raw["deletion_curves"]], axis=0)
    ins_mean = np.mean([[e for _, e in curve] for curve in raw["insertion_curves"]], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fractions, del_mean, "b-o", markersize=3, label="Deletion")
    axes[0].set_xlabel("Fraction deleted")
    axes[0].set_ylabel("MSE from baseline")
    axes[0].set_title("Deletion Curve (avg)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(fractions, ins_mean, "r-o", markersize=3, label="Insertion")
    axes[1].set_xlabel("Fraction inserted")
    axes[1].set_ylabel("MSE from full")
    axes[1].set_title("Insertion Curve (avg)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Faithfulness Curves — {tag}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_faithfulness_curves_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig_faithfulness_curves_{tag}.png")


def plot_extended_pas(summary, save_dir, tag):
    """Plot extended PAS metrics as grouped bar chart."""
    pas = summary["extended_pas"]
    priors = list(pas.keys())
    metrics = ["iou", "soft_iou", "pearson_corr", "spearman_corr"]
    metric_labels = ["IoU", "Soft-IoU", "Pearson", "Spearman"]

    x = np.arange(len(priors))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [pas[p][metric]["mean"] for p in priors]
        stds = [pas[p][metric]["std"] for p in priors]
        ax.bar(x + i * width, vals, width, yerr=stds, label=label, capsize=3)

    ax.set_ylabel("Score")
    ax.set_title(f"Extended Physical Alignment — {tag}")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(priors)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_extended_pas_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig_extended_pas_{tag}.png")


def plot_cross_method(summary, save_dir, tag):
    """Plot cross-method agreement."""
    cross = summary["cross_method_agreement"]
    if not cross:
        return

    pairs = list(cross.keys())
    metrics = ["topk_overlap", "spearman", "cosine"]
    metric_labels = ["Top-k Overlap", "Spearman", "Cosine"]

    x = np.arange(len(pairs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [cross[p][metric]["mean"] for p in pairs]
        stds = [cross[p][metric]["std"] for p in pairs]
        ax.bar(x + i * width, vals, width, yerr=stds, label=label, capsize=3)

    ax.set_ylabel("Score")
    ax.set_title(f"Cross-Method Agreement — {tag}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(pairs, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_cross_method_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig_cross_method_{tag}.png")

    # Grouped by error if available
    grouped = summary.get("cross_method_grouped", {})
    if grouped and grouped.get("low_error") and grouped.get("high_error"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax_i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            low_vals = [grouped["low_error"][p][metric]["mean"] for p in pairs]
            high_vals = [grouped["high_error"][p][metric]["mean"] for p in pairs]

            x_pos = np.arange(len(pairs))
            axes[ax_i].bar(x_pos - 0.15, low_vals, 0.3, label="Low Error", color="steelblue")
            axes[ax_i].bar(x_pos + 0.15, high_vals, 0.3, label="High Error", color="coral")
            axes[ax_i].set_title(label)
            axes[ax_i].set_xticks(x_pos)
            axes[ax_i].set_xticklabels(pairs, rotation=15, ha="right")
            axes[ax_i].legend()
            axes[ax_i].grid(True, alpha=0.3, axis="y")

        plt.suptitle(f"Cross-Method Agreement by Error Group — {tag}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"fig_cross_method_grouped_{tag}.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved fig_cross_method_grouped_{tag}.png")


def plot_threshold_sensitivity(summary, save_dir, tag):
    """Plot PAS at different thresholds."""
    ts = summary["threshold_sensitivity"]
    per_prior = ts.get("per_prior", {})
    if not per_prior:
        return

    priors = list(per_prior.keys())
    thresholds = sorted(per_prior[priors[0]].keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    for prior in priors:
        vals = [per_prior[prior][t] for t in thresholds]
        ax.plot(thresholds, vals, "o-", label=prior, markersize=6)

    ax.set_xlabel("Top-k Threshold (%)")
    ax.set_ylabel("PAS (IoU)")
    ax.set_title(f"Threshold Sensitivity — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    robust = ts.get("conclusion_robust", False)
    ax.annotate(f"Conclusion robust: {robust}",
                xy=(0.02, 0.98), xycoords="axes fraction", va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_threshold_sensitivity_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig_threshold_sensitivity_{tag}.png")


def plot_range_conditioned(summary, save_dir, tag):
    """Plot range-conditioned explanation analysis."""
    rc = summary["range_conditioned"]
    intensity = rc.get("intensity", {})
    if not intensity:
        return

    ranges = list(intensity.keys())
    x = np.arange(len(ranges))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Intensity fraction
    fracs = [intensity[r]["intensity_fraction"]["mean"] for r in ranges]
    frac_stds = [intensity[r]["intensity_fraction"]["std"] for r in ranges]
    axes[0].bar(x, fracs, yerr=frac_stds, capsize=5, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ranges)
    axes[0].set_ylabel("Explanation Mass Fraction")
    axes[0].set_title("Explanation Distribution by Distance")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Mean intensity
    means = [intensity[r]["mean_intensity"]["mean"] for r in ranges]
    mean_stds = [intensity[r]["mean_intensity"]["std"] for r in ranges]
    axes[1].bar(x, means, yerr=mean_stds, capsize=5, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ranges)
    axes[1].set_ylabel("Mean Explanation Intensity")
    axes[1].set_title("Mean Intensity by Distance")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Range-Conditioned XAI — {tag}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_range_conditioned_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig_range_conditioned_{tag}.png")


def plot_los_nlos(summary, save_dir, tag):
    """Plot LoS/NLoS split analysis."""
    ln = summary["los_nlos"]
    mass = ln.get("los_mass_fraction", {})
    impact = ln.get("deletion_impact", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mass fraction
    regions = ["LoS", "NLoS"]
    fracs = [mass.get("mean", 0), ln.get("nlos_mass_fraction", {}).get("mean", 0)]
    stds = [mass.get("std", 0), ln.get("nlos_mass_fraction", {}).get("std", 0)]
    axes[0].bar(regions, fracs, yerr=stds, capsize=5, color=["steelblue", "coral"])
    axes[0].set_ylabel("Explanation Mass Fraction")
    axes[0].set_title("Explanation Mass: LoS vs NLoS")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Deletion impact
    if impact:
        imp_regions = list(impact.keys())
        imp_vals = [impact[r]["mean"] for r in imp_regions]
        imp_stds = [impact[r]["std"] for r in imp_regions]
        axes[1].bar(imp_regions, imp_vals, yerr=imp_stds, capsize=5, color="salmon")
        axes[1].set_ylabel("MSE increase after deletion")
        axes[1].set_title("Region Deletion Impact")
        axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"LoS/NLoS Analysis — {tag}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_los_nlos_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig_los_nlos_{tag}.png")


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device)

    test_dataset = RadioMapSeerDataset(
        root_dir=config["data"]["root_dir"],
        gain_method=config["data"]["gain_method"],
        split="test",
        seed=get_split_seed(config),
    )

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Run analysis
    raw = run_analysis(model, test_dataset, device, args, config)

    # Aggregate
    summary = aggregate_results(raw)
    summary["tag"] = args.tag

    # Save results
    json_path = os.path.join(save_dir, f"enhanced_xai_summary_{args.tag}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {json_path}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(summary, raw, save_dir, args.tag)

    # Print key results
    print(f"\n{'='*60}")
    print(f"Enhanced XAI Results — {args.tag}")
    print(f"{'='*60}")
    print(f"Samples: {summary['n_samples']}, RMSE: {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")

    faith = summary["faithfulness"]
    print(f"\nFaithfulness:")
    print(f"  Deletion AUC: {faith['deletion_auc']['mean']:.4f} ± {faith['deletion_auc']['std']:.4f}")
    print(f"  Insertion AUC: {faith['insertion_auc']['mean']:.4f} ± {faith['insertion_auc']['std']:.4f}")
    print(f"  Deletion Monotonicity: {faith['deletion_monotonicity']['mean']:.3f}")
    print(f"  Insertion Monotonicity: {faith['insertion_monotonicity']['mean']:.3f}")

    print(f"\nExtended PAS (IoU):")
    for prior in summary["extended_pas"]:
        iou = summary["extended_pas"][prior]["iou"]["mean"]
        soft = summary["extended_pas"][prior]["soft_iou"]["mean"]
        pearson = summary["extended_pas"][prior]["pearson_corr"]["mean"]
        print(f"  {prior}: IoU={iou:.4f}, Soft-IoU={soft:.4f}, Pearson={pearson:.4f}")

    if summary["cross_method_agreement"]:
        print(f"\nCross-Method Agreement:")
        for pair, metrics in summary["cross_method_agreement"].items():
            print(f"  {pair}: overlap={metrics['topk_overlap']['mean']:.3f}, "
                  f"spearman={metrics['spearman']['mean']:.3f}, "
                  f"cosine={metrics['cosine']['mean']:.3f}")

    print(f"\nLoS/NLoS Split:")
    ln = summary["los_nlos"]
    print(f"  LoS mass: {ln['los_mass_fraction']['mean']:.3f} ± {ln['los_mass_fraction']['std']:.3f}")
    print(f"  NLoS mass: {ln['nlos_mass_fraction']['mean']:.3f} ± {ln['nlos_mass_fraction']['std']:.3f}")

    print(f"\nRange-Conditioned Intensity:")
    for rn, stats in summary["range_conditioned"]["intensity"].items():
        print(f"  {rn}: fraction={stats['intensity_fraction']['mean']:.3f}")


if __name__ == "__main__":
    main()
