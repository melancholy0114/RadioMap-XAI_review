"""
In-Distribution vs Out-of-Distribution Analysis.

Compares model performance and explanation patterns between
training domain (ID) and test domain (OOD) samples.
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


def compute_prediction_metrics(model, dataloader, device):
    """Compute RMSE, MAE for a dataset."""
    model.eval()
    rmses = []
    maes = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            with autocast("cuda"):
                outputs = model(inputs)

            for i in range(outputs.shape[0]):
                pred = outputs[i, 0].cpu().numpy()
                gt = targets[i, 0].cpu().numpy()
                rmses.append(np.sqrt(np.mean((pred - gt) ** 2)))
                maes.append(np.mean(np.abs(pred - gt)))

    return {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmses": rmses,
        "maes": maes,
    }


def analyze_id_vs_ood(
    model,
    config,
    device,
    explainer=None,
    n_explain_samples=50,
    save_dir="outputs/figures",
):
    """
    Compare ID (train domain) vs OOD (test domain) performance and explanations.

    ID: samples from maps in the training set
    OOD: samples from maps NOT in the training set
    """
    os.makedirs(save_dir, exist_ok=True)

    seed = get_split_seed(config)

    # Create datasets
    train_dataset = RadioMapSeerDataset(
        root_dir=config["data"]["root_dir"],
        gain_method=config["data"]["gain_method"],
        split="train", seed=seed,
    )
    test_dataset = RadioMapSeerDataset(
        root_dir=config["data"]["root_dir"],
        gain_method=config["data"]["gain_method"],
        split="test", seed=seed,
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

    # Compute metrics
    print("Computing ID metrics...")
    id_metrics = compute_prediction_metrics(model, train_loader, device)
    print(f"  ID RMSE: {id_metrics['rmse_mean']:.4f} ± {id_metrics['rmse_std']:.4f}")

    print("Computing OOD metrics...")
    ood_metrics = compute_prediction_metrics(model, test_loader, device)
    print(f"  OOD RMSE: {ood_metrics['rmse_mean']:.4f} ± {ood_metrics['rmse_std']:.4f}")

    # Plot RMSE distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(id_metrics["rmses"], bins=50, alpha=0.6, label="ID (Train)", density=True)
    axes[0].hist(ood_metrics["rmses"], bins=50, alpha=0.6, label="OOD (Test)", density=True)
    axes[0].set_xlabel("RMSE")
    axes[0].set_ylabel("Density")
    axes[0].set_title("RMSE Distribution: ID vs OOD")
    axes[0].legend()

    axes[1].hist(id_metrics["maes"], bins=50, alpha=0.6, label="ID (Train)", density=True)
    axes[1].hist(ood_metrics["maes"], bins=50, alpha=0.6, label="OOD (Test)", density=True)
    axes[1].set_xlabel("MAE")
    axes[1].set_ylabel("Density")
    axes[1].set_title("MAE Distribution: ID vs OOD")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "id_vs_ood_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "id_metrics": {k: v for k, v in id_metrics.items() if not isinstance(v, list)},
        "ood_metrics": {k: v for k, v in ood_metrics.items() if not isinstance(v, list)},
    }

    return results
