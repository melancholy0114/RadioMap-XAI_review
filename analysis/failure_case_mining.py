"""
Failure case mining: identify and analyze samples where the model performs poorly.

For each failure case:
  - Show prediction vs GT
  - Show explanation
  - Compare with physical prior
  - Identify potential causes of failure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.radiomapseer_dataset import RadioMapSeerDataset
from torch.utils.data import DataLoader
from torch.amp import autocast
from utils import get_split_seed


def mine_failure_cases(
    model,
    config,
    device,
    top_k=20,
    save_dir="outputs/figures",
):
    """
    Find the worst-performing test samples and analyze them.
    """
    os.makedirs(save_dir, exist_ok=True)

    test_dataset = RadioMapSeerDataset(
        root_dir=config["data"]["root_dir"],
        gain_method=config["data"]["gain_method"],
        split="test",
        seed=get_split_seed(config),
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Compute all errors
    print("Computing errors for all test samples...")
    sample_errors = []
    model.eval()

    for i, batch in enumerate(test_loader):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        with torch.no_grad(), autocast("cuda"):
            outputs = model(inputs)

        pred = outputs[0, 0].cpu().numpy()
        gt = targets[0, 0].cpu().numpy()
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        mae = np.mean(np.abs(pred - gt))

        sample_errors.append({
            "index": i,
            "map_id": batch["map_id"][0],
            "tx_idx": batch["tx_idx"][0].item(),
            "rmse": rmse,
            "mae": mae,
            "prediction": pred,
            "ground_truth": gt,
            "building": batch["building"][0].numpy(),
            "tx_pos": batch["tx_position"][0].numpy(),
        })

    # Sort by error
    sample_errors.sort(key=lambda x: x["rmse"], reverse=True)

    # Worst cases
    worst = sample_errors[:top_k]
    best = sample_errors[-top_k:]

    print(f"Top-{top_k} worst cases: RMSE range [{worst[-1]['rmse']:.4f}, {worst[0]['rmse']:.4f}]")
    print(f"Top-{top_k} best cases: RMSE range [{best[0]['rmse']:.4f}, {best[-1]['rmse']:.4f}]")

    # Visualize worst cases
    n_show = min(6, len(worst))
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        case = worst[i]
        axes[i, 0].imshow(case["building"], cmap="gray")
        axes[i, 0].set_title(f"Building (Map {case['map_id']})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(case["ground_truth"], cmap="jet", vmin=0, vmax=1)
        axes[i, 1].set_title("GT Radio Map")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(case["prediction"], cmap="jet", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Prediction (RMSE={case['rmse']:.4f})")
        axes[i, 2].axis("off")

        error = np.abs(case["prediction"] - case["ground_truth"])
        axes[i, 3].imshow(error, cmap="hot", vmin=0, vmax=0.5)
        axes[i, 3].set_title("Error")
        axes[i, 3].axis("off")

    plt.suptitle(f"Top-{n_show} Failure Cases", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "failure_cases.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save error distribution
    all_rmses = [s["rmse"] for s in sample_errors]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(all_rmses, bins=100, alpha=0.7)
    ax.axvline(np.percentile(all_rmses, 95), color="r", linestyle="--", label="95th percentile")
    ax.axvline(np.mean(all_rmses), color="g", linestyle="--", label=f"Mean ({np.mean(all_rmses):.4f})")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Count")
    ax.set_title("Test Set Error Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "worst_cases": worst[:top_k],
        "best_cases": best[:top_k],
        "mean_rmse": float(np.mean(all_rmses)),
        "median_rmse": float(np.median(all_rmses)),
        "p95_rmse": float(np.percentile(all_rmses, 95)),
    }
