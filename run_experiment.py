"""
Main experiment runner for the radio map XAI pipeline.

Steps:
  1. Train a baseline model if no checkpoint is available.
  2. Run inference on test samples.
  3. Generate explanation maps.
  4. Build physical priors.
  5. Evaluate explanations.
  6. Analyze generalization behavior.
  7. Export figures.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import argparse
import torch
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.radiomapseer_dataset import RadioMapSeerDataset
from torch.utils.data import DataLoader
from inference.infer import load_model
from explanation import IntegratedGradients, GradCAM, OcclusionSensitivity
from visualization.plot_explanations import plot_method_comparison, plot_single_explanation
from priors import compute_los_mask_fast, compute_obstruction_mask, compute_directional_mask
from metrics import Faithfulness, PhysicalAlignmentScore, Stability, Consistency
from utils import get_evaluation_seed, get_split_seed, get_training_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full radio map XAI pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--num-samples", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_seed = get_training_seed(config)
    evaluation_seed = get_evaluation_seed(config)
    torch.manual_seed(training_seed)
    np.random.seed(evaluation_seed)

    figure_dir = config["output"]["figure_dir"]
    os.makedirs(figure_dir, exist_ok=True)

    # Step 1: Load model
    ckpt_dir = config["output"]["checkpoint_dir"]
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(ckpt_dir, "best_model.pth")

    if not os.path.exists(ckpt_path):
        if args.skip_training:
            print(f"Checkpoint not found at {ckpt_path}")
            print("Train a model first or provide --checkpoint with a valid path.")
            return
        else:
            print("No checkpoint found. Training from scratch...")
            from training.train import train
            train(config)
            ckpt_path = os.path.join(ckpt_dir, "best_model.pth")

    print(f"Loading model from {ckpt_path}")
    model = load_model(config, ckpt_path, device)

    # Step 2: Load test dataset
    test_dataset = RadioMapSeerDataset(
        root_dir=config["data"]["root_dir"],
        gain_method=config["data"]["gain_method"],
        split="test",
        seed=get_split_seed(config),
    )
    n_samples = min(args.num_samples, len(test_dataset))
    indices = np.random.choice(len(test_dataset), n_samples, replace=False)

    # Step 3: Initialize explainers and metrics
    explainer_ig = IntegratedGradients(model, device)
    explainer_cam = GradCAM(model, device=device)
    explainer_occ = OcclusionSensitivity(model, device)

    pas_evaluator = PhysicalAlignmentScore()

    # Step 4: Process samples
    all_results = []

    for sample_idx, idx in enumerate(indices):
        sample = test_dataset[int(idx)]
        inputs = sample["input"].unsqueeze(0).to(device)
        building = sample["building"].numpy()
        tx_pos = sample["tx_position"].numpy()
        target = sample["target"][0].numpy()
        map_id = sample["map_id"]
        tx_idx = sample["tx_idx"]

        print(f"\n[{sample_idx+1}/{n_samples}] Map {map_id}, Tx {tx_idx}")

        # Prediction
        with torch.no_grad(), torch.amp.autocast("cuda"):
            pred = model(inputs)
        prediction = pred[0, 0].cpu().numpy()
        rmse = np.sqrt(np.mean((prediction - target) ** 2))
        print(f"  RMSE: {rmse:.4f}")

        # Explanations
        print("  Computing explanations...")
        ig_map = explainer_ig.explain_sample(inputs, n_steps=config["explainability"]["ig_steps"])
        cam_map = explainer_cam.explain_sample(inputs)
        occ_map = explainer_occ.explain_sample(
            inputs,
            window_size=config["explainability"]["occlusion_window"],
            stride=config["explainability"]["occlusion_stride"],
        )

        # Physical priors
        print("  Computing physical priors...")
        los = compute_los_mask_fast(building, tx_pos)
        obstruction = compute_obstruction_mask(building, tx_pos)
        directional = compute_directional_mask(tx_pos, img_size=256)

        prior_masks = {"los": los, "obstruction": obstruction, "directional": directional}

        # PAS evaluation
        print("  Evaluating PAS...")
        pas_scores = pas_evaluator.compute_multi_prior(ig_map, prior_masks, top_k_percent=20)

        result = {
            "map_id": map_id,
            "tx_idx": tx_idx,
            "rmse": float(rmse),
            "pas_scores": {k: v["pas"] for k, v in pas_scores.items()},
            "ig_pas": pas_scores,
        }
        all_results.append(result)

        # Save visualization
        save_dir = os.path.join(config["output"]["explanation_dir"], f"{map_id}_{tx_idx}")
        os.makedirs(save_dir, exist_ok=True)

        explanations = {
            "IntegratedGradients": ig_map,
            "GradCAM": cam_map,
            "Occlusion": occ_map,
        }
        plot_method_comparison(
            building, tx_pos, prediction, target,
            explanations,
            os.path.join(save_dir, "comparison.png"),
        )

        # Save priors visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(building, cmap="gray")
        axes[0, 0].set_title("Building Map")
        axes[0, 1].imshow(los, cmap="hot")
        axes[0, 1].set_title("LoS Mask")
        axes[0, 2].imshow(obstruction, cmap="hot")
        axes[0, 2].set_title("Obstruction Mask")
        axes[1, 0].imshow(directional, cmap="hot")
        axes[1, 0].set_title("Directional Mask")
        axes[1, 1].imshow(prediction, cmap="jet")
        axes[1, 1].set_title("Prediction")
        axes[1, 2].imshow(ig_map, cmap="hot")
        axes[1, 2].set_title("Explanation (IG)")
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "priors.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Step 5: Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    rmses = [r["rmse"] for r in all_results]
    print(f"Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    for prior_name in ["los", "obstruction", "directional"]:
        pas_vals = [r["pas_scores"][prior_name] for r in all_results if prior_name in r["pas_scores"]]
        if pas_vals:
            print(f"PAS ({prior_name}): {np.mean(pas_vals):.4f} ± {np.std(pas_vals):.4f}")

    # Correlation: RMSE vs PAS
    for prior_name in ["los", "obstruction", "directional"]:
        pas_vals = [r["pas_scores"].get(prior_name, 0) for r in all_results]
        if len(pas_vals) == len(rmses) and np.std(pas_vals) > 1e-8:
            corr = np.corrcoef(rmses, pas_vals)[0, 1]
            print(f"Correlation (RMSE vs PAS-{prior_name}): {corr:.4f}")

    # Save results
    results_path = os.path.join(config["output"]["log_dir"], "experiment_results.json")
    os.makedirs(config["output"]["log_dir"], exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
