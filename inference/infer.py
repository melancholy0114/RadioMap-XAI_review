"""
Inference pipeline for radio map prediction.
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

from model import build_model, normalize_state_dict, validate_checkpoint_model
from datasets.radiomapseer_dataset import RadioMapSeerDataset
from torch.utils.data import DataLoader
from torch.amp import autocast


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for radio map prediction")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="outputs/predictions")
    return parser.parse_args()


def load_model(config, checkpoint_path, device):
    model = build_model(config["model"]).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    validate_checkpoint_model(ckpt, model)
    model.load_state_dict(normalize_state_dict(ckpt["model_state_dict"]))
    model.eval()
    return model


def run_inference(model, dataset, device, num_samples=10, save_dir="outputs/predictions"):
    os.makedirs(save_dir, exist_ok=True)

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    results = []
    for idx in indices:
        sample = dataset[int(idx)]
        inputs = sample["input"].unsqueeze(0).to(device)

        with torch.no_grad(), autocast("cuda", enabled=device.type == "cuda"):
            pred = model(inputs)

        pred_np = pred[0, 0].cpu().numpy()
        target_np = sample["target"][0].numpy()
        building_np = sample["building"].numpy()
        tx_pos = sample["tx_position"].numpy()

        # Compute metrics
        mse = np.mean((pred_np - target_np) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_np - target_np))

        result = {
            "map_id": sample["map_id"],
            "tx_idx": sample["tx_idx"],
            "tx_pos": tx_pos.tolist(),
            "rmse": float(rmse),
            "mae": float(mae),
            "mse": float(mse),
        }
        results.append(result)

        # Visualize
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(building_np, cmap="gray")
        axes[0].set_title("Building Map")
        axes[0].axis("off")

        axes[1].imshow(target_np, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("GT Radio Map")
        axes[1].axis("off")

        axes[2].imshow(pred_np, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title(f"Prediction (RMSE={rmse:.4f})")
        axes[2].axis("off")

        error = np.abs(pred_np - target_np)
        axes[3].imshow(error, cmap="hot", vmin=0, vmax=0.5)
        axes[3].set_title(f"Absolute Error (MAE={mae:.4f})")
        axes[3].axis("off")

        plt.suptitle(
            f"Map {sample['map_id']}, Tx {sample['tx_idx']}, "
            f"Pos ({tx_pos[0]:.0f}, {tx_pos[1]:.0f})",
            fontsize=11,
        )
        plt.tight_layout()
        path = os.path.join(save_dir, f"pred_{sample['map_id']}_{sample['tx_idx']}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Map {sample['map_id']}, Tx {sample['tx_idx']}: RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Summary
    rmses = [r["rmse"] for r in results]
    maes = [r["mae"] for r in results]
    print(f"\nInference Summary ({len(results)} samples):")
    print(f"  Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"  Mean MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")

    return results


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config, args.checkpoint, device)
    dataset = RadioMapSeerDataset(
        root_dir=config["data"]["root_dir"],
        gain_method=config["data"]["gain_method"],
        split="test",
        seed=config["training"]["seed"],
    )

    results = run_inference(model, dataset, device, args.num_samples, args.save_dir)
    print(f"Results saved to {args.save_dir}")
