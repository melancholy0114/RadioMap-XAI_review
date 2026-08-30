"""
Run all post-training analysis experiments:
  1. ID vs OOD analysis (full train + test set)
  2. Failure case mining (full test set)
  3. Explanation drift vs error (50 train + 50 test samples)

Usage:
    conda run -n pytorch python3 scripts/run_all_analysis.py --config configs/config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import torch
import json

from inference.infer import load_model
from analysis.id_vs_ood_analysis import analyze_id_vs_ood
from analysis.failure_case_mining import mine_failure_cases
from analysis.explanation_drift_vs_error import analyze_drift_vs_error
from explanation.integrated_gradients import IntegratedGradients
from utils import get_evaluation_seed, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(get_evaluation_seed(config))

    print("Loading model...")
    model = load_model(config, args.checkpoint, device)

    save_dir = config["output"]["figure_dir"]
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    # 1. ID vs OOD
    print("\n" + "=" * 60)
    print("1. ID vs OOD Analysis")
    print("=" * 60)
    id_ood_results = analyze_id_vs_ood(model, config, device, save_dir=save_dir)
    all_results["id_vs_ood"] = id_ood_results
    print(f"  ID RMSE: {id_ood_results['id_metrics']['rmse_mean']:.4f} ({id_ood_results['id_metrics']['rmse_mean']*139:.2f} dB)")
    print(f"  OOD RMSE: {id_ood_results['ood_metrics']['rmse_mean']:.4f} ({id_ood_results['ood_metrics']['rmse_mean']*139:.2f} dB)")

    # 2. Failure case mining
    print("\n" + "=" * 60)
    print("2. Failure Case Mining")
    print("=" * 60)
    failure_results = mine_failure_cases(model, config, device, top_k=20, save_dir=save_dir)
    all_results["failure_cases"] = {
        "mean_rmse": failure_results["mean_rmse"],
        "median_rmse": failure_results["median_rmse"],
        "p95_rmse": failure_results["p95_rmse"],
        "worst_rmse": failure_results["worst_cases"][0]["rmse"],
        "best_rmse": failure_results["best_cases"][0]["rmse"],
    }
    print(f"  Mean RMSE: {failure_results['mean_rmse']:.4f} ({failure_results['mean_rmse']*139:.2f} dB)")
    print(f"  Median RMSE: {failure_results['median_rmse']:.4f}")
    print(f"  P95 RMSE: {failure_results['p95_rmse']:.4f}")
    print(f"  Worst: {failure_results['worst_cases'][0]['rmse']:.4f}")

    # 3. Explanation drift vs error
    print("\n" + "=" * 60)
    print("3. Explanation Drift vs Error")
    print("=" * 60)
    explainer = IntegratedGradients(model, device)
    drift_results = analyze_drift_vs_error(
        model, explainer, config, device, n_samples=50, save_dir=save_dir
    )
    all_results["drift_vs_error"] = drift_results

    # Save all results
    results_path = os.path.join(config["output"]["log_dir"], "analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
