"""
Explanation Drift vs Prediction Error analysis.

Core analysis: Does explanation drift correlate with prediction error increase
when moving from ID to OOD samples?
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
from utils import get_evaluation_seed, get_split_seed


def compute_sample_error(model, sample, device):
    """Compute RMSE for a single sample."""
    model.eval()
    inputs = sample["input"].unsqueeze(0).to(device)
    targets = sample["target"].to(device)

    with torch.no_grad(), autocast("cuda"):
        outputs = model(inputs)

    pred = outputs[0, 0].cpu().numpy()
    gt = targets[0, 0].cpu().numpy()
    return np.sqrt(np.mean((pred - gt) ** 2))


def compute_explanation(explainer, sample, device):
    """Compute explanation for a single sample."""
    inputs = sample["input"].unsqueeze(0).to(device)
    return explainer.explain_sample(inputs)


def compute_drift(expl_id, expl_ood):
    """Compute explanation drift between two maps."""
    return np.sqrt(np.mean((expl_id - expl_ood) ** 2))


def analyze_drift_vs_error(
    model,
    explainer,
    config,
    device,
    n_samples=100,
    save_dir="outputs/figures",
):
    """
    Analyze correlation between explanation drift and prediction error.

    For each test sample:
    1. Find the most similar training sample (by building map)
    2. Compute prediction error
    3. Compute explanation
    4. Compute drift from the most similar training explanation
    5. Analyze correlation

    Simplified: compare train vs test sample distributions.
    """
    os.makedirs(save_dir, exist_ok=True)

    seed = get_split_seed(config)

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

    # Sample subset
    n_train = min(n_samples, len(train_dataset))
    n_test = min(n_samples, len(test_dataset))

    rng = np.random.default_rng(get_evaluation_seed(config))
    train_indices = rng.choice(len(train_dataset), n_train, replace=False)
    test_indices = rng.choice(len(test_dataset), n_test, replace=False)

    # Compute explanations and errors for train samples
    print("Computing train explanations and errors...")
    train_data = []
    for i, idx in enumerate(train_indices):
        sample = train_dataset[int(idx)]
        error = compute_sample_error(model, sample, device)
        expl = compute_explanation(explainer, sample, device)
        train_data.append({
            "error": error,
            "explanation": expl,
            "building": sample["building"].numpy(),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_train}")

    # Compute explanations and errors for test samples
    print("Computing test explanations and errors...")
    test_data = []
    for i, idx in enumerate(test_indices):
        sample = test_dataset[int(idx)]
        error = compute_sample_error(model, sample, device)
        expl = compute_explanation(explainer, sample, device)
        test_data.append({
            "error": error,
            "explanation": expl,
            "building": sample["building"].numpy(),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_test}")

    # For each test sample, find nearest train sample and compute drift
    print("Computing explanation drifts...")
    drifts = []
    errors = []

    for td in test_data:
        # Find most similar train sample by building map correlation
        best_sim = -1
        best_expl = None
        for trd in train_data:
            sim = np.corrcoef(td["building"].flatten(), trd["building"].flatten())[0, 1]
            if sim > best_sim:
                best_sim = sim
                best_expl = trd["explanation"]

        drift = compute_drift(best_expl, td["explanation"])
        drifts.append(drift)
        errors.append(td["error"])

    drifts = np.array(drifts)
    errors = np.array(errors)

    # Correlation analysis
    correlation = np.corrcoef(drifts, errors)[0, 1]

    # Plot drift vs error
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(drifts, errors, alpha=0.5, s=10)
    z = np.polyfit(drifts, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(drifts.min(), drifts.max(), 100)
    axes[0].plot(x_line, p(x_line), "r-", linewidth=2)
    axes[0].set_xlabel("Explanation Drift (L2)")
    axes[0].set_ylabel("Prediction Error (RMSE)")
    axes[0].set_title(f"Drift vs Error (r={correlation:.3f})")

    # Error distribution
    train_errors = [d["error"] for d in train_data]
    test_errors = [d["error"] for d in test_data]
    axes[1].hist(train_errors, bins=30, alpha=0.6, label="Train (ID)", density=True)
    axes[1].hist(test_errors, bins=30, alpha=0.6, label="Test (OOD)", density=True)
    axes[1].set_xlabel("RMSE")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Error Distribution")
    axes[1].legend()

    # Drift distribution
    axes[2].hist(drifts, bins=30, alpha=0.7)
    axes[2].set_xlabel("Explanation Drift")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Drift Distribution (mean={drifts.mean():.4f})")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "drift_vs_error.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nResults:")
    print(f"  Correlation (drift vs error): {correlation:.4f}")
    print(f"  Mean drift: {drifts.mean():.4f} ± {drifts.std():.4f}")
    print(f"  Mean train error: {np.mean(train_errors):.4f}")
    print(f"  Mean test error: {np.mean(test_errors):.4f}")

    return {
        "correlation": float(correlation),
        "mean_drift": float(drifts.mean()),
        "mean_train_error": float(np.mean(train_errors)),
        "mean_test_error": float(np.mean(test_errors)),
    }
