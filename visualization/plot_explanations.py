"""
Visualization for explanation results.

Creates overlay plots of:
  - Original input (building map)
  - Model prediction
  - Explanation map overlay
  - Comparison across methods

Usage:
    conda run -n pytorch python3 visualization/plot_explanations.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best_model.pth
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from datasets.radiomapseer_dataset import RadioMapSeerDataset
from explanation import IntegratedGradients, GradCAM, OcclusionSensitivity
from inference.infer import load_model
from utils import get_split_seed


def plot_single_explanation(
    building, tx_pos, prediction, target, explanation, method_name, save_path
):
    """Plot a single explanation result."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Building map
    axes[0].imshow(building, cmap="gray")
    axes[0].set_title("Building Map")
    axes[0].axis("off")

    # GT Radio Map
    axes[1].imshow(target, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("GT Radio Map")
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(prediction, cmap="jet", vmin=0, vmax=1)
    rmse = np.sqrt(np.mean((prediction - target) ** 2))
    axes[2].set_title(f"Prediction (RMSE={rmse:.4f})")
    axes[2].axis("off")

    # Explanation map
    im = axes[3].imshow(explanation, cmap="hot", vmin=0, vmax=explanation.max() + 1e-8)
    axes[3].set_title(f"Explanation ({method_name})")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    # Overlay: building + explanation
    axes[4].imshow(building, cmap="gray", alpha=0.5)
    axes[4].imshow(explanation, cmap="hot", alpha=0.5)
    axes[4].plot(tx_pos[0], tx_pos[1], "b*", markersize=15)
    axes[4].set_title("Overlay")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_method_comparison(
    building, tx_pos, prediction, target, explanations, save_path
):
    """Compare explanations from different methods side by side."""
    n_methods = len(explanations)
    fig, axes = plt.subplots(2, n_methods + 2, figsize=(4 * (n_methods + 2), 8))

    # Top row: input, prediction, explanations
    axes[0, 0].imshow(building, cmap="gray")
    axes[0, 0].set_title("Building Map")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(prediction, cmap="jet", vmin=0, vmax=1)
    axes[0, 1].set_title("Prediction")
    axes[0, 1].axis("off")

    for i, (method_name, expl) in enumerate(explanations.items()):
        axes[0, i + 2].imshow(expl, cmap="hot")
        axes[0, i + 2].set_title(method_name)
        axes[0, i + 2].axis("off")

    # Bottom row: overlays
    axes[1, 0].imshow(target, cmap="jet", vmin=0, vmax=1)
    axes[1, 0].set_title("GT Radio Map")
    axes[1, 0].axis("off")

    error = np.abs(prediction - target)
    axes[1, 1].imshow(error, cmap="hot", vmin=0, vmax=max(0.5, error.max()))
    axes[1, 1].set_title("Prediction Error")
    axes[1, 1].axis("off")

    for i, (method_name, expl) in enumerate(explanations.items()):
        axes[1, i + 2].imshow(building, cmap="gray", alpha=0.5)
        axes[1, i + 2].imshow(expl, cmap="hot", alpha=0.5)
        axes[1, i + 2].plot(tx_pos[0], tx_pos[1], "b*", markersize=12)
        axes[1, i + 2].set_title(f"{method_name} (overlay)")
        axes[1, i + 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_visualization(config, checkpoint_path, num_samples=10, save_dir="outputs/explanations"):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, checkpoint_path, device)

    dataset = RadioMapSeerDataset(
        root_dir=config["data"]["root_dir"],
        gain_method=config["data"]["gain_method"],
        split="test",
        seed=get_split_seed(config),
    )

    # Initialize explainers
    expl_ig = IntegratedGradients(model, device)
    expl_cam = GradCAM(model, device=device)
    expl_occ = OcclusionSensitivity(model, device)

    ig_steps = config["explainability"]["ig_steps"]
    occ_stride = config["explainability"]["occlusion_stride"]
    occ_window = config["explainability"]["occlusion_window"]

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    for sample_idx, idx in enumerate(indices):
        sample = dataset[int(idx)]
        inputs = sample["input"].unsqueeze(0).to(device)
        building = sample["building"].numpy()
        tx_pos = sample["tx_position"].numpy()
        target = sample["target"][0].numpy()

        with torch.no_grad():
            pred = model(inputs)
        prediction = pred[0, 0].cpu().numpy()

        map_id = sample["map_id"]
        tx_idx = sample["tx_idx"]
        print(f"Sample {sample_idx+1}/{len(indices)}: Map {map_id}, Tx {tx_idx}")

        # Compute explanations
        explanations = {}

        print(f"  Computing Integrated Gradients ({ig_steps} steps)...")
        ig_map = expl_ig.explain_sample(inputs, n_steps=ig_steps)
        explanations["Integrated Gradients"] = ig_map

        print(f"  Computing Grad-CAM...")
        cam_map = expl_cam.explain_sample(inputs)
        explanations["Grad-CAM"] = cam_map

        print(f"  Computing Occlusion Sensitivity...")
        occ_map = expl_occ.explain_sample(inputs, window_size=occ_window, stride=occ_stride)
        explanations["Occlusion"] = occ_map

        # Save individual explanations
        for method_name, expl in explanations.items():
            save_path = os.path.join(
                save_dir, f"expl_{method_name}_{map_id}_{tx_idx}.png"
            )
            plot_single_explanation(
                building, tx_pos, prediction, target, expl, method_name, save_path
            )

        # Save comparison
        compare_path = os.path.join(save_dir, f"comparison_{map_id}_{tx_idx}.png")
        plot_method_comparison(
            building, tx_pos, prediction, target, explanations, compare_path
        )
        print(f"  Saved comparison to {compare_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="outputs/explanations")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_visualization(config, args.checkpoint, args.num_samples, args.save_dir)
