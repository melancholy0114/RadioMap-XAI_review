"""
RadioMapSeer Dataset Loader

Data structure:
  - data/png/buildings_complete/{id}.png  -> building map (256x256, grayscale)
  - data/antenna/{id}.json                -> 80 Tx positions [[x,y], ...]
  - data/gain/{method}/{id}_{tx}.png      -> gain/radio map (256x256, grayscale)

RadioMapSeer stores Tx coordinates with a bottom-left origin, while image
arrays use a top-left origin. Tx coordinates are converted at load time so
all downstream heatmaps and physical priors use image coordinates.

Each sample: (building_map, antenna_heatmap, gain_map)
  - building_map: 1-channel, 256x256
  - antenna_heatmap: 1-channel, 256x256 (Gaussian blob at Tx position)
  - gain_map: 1-channel, 256x256 (target radio map)
"""

import os
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torchvision import transforms


class RadioMapSeerDataset(Dataset):
    """RadioMapSeer dataset for radio map prediction."""

    def __init__(
        self,
        root_dir: str,
        gain_method: str = "DPM",
        img_size: int = 256,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.gain_method = gain_method
        self.img_size = img_size
        self.split = split

        # Paths
        self.buildings_dir = self.root_dir / "png" / "buildings_complete"
        self.antenna_dir = self.root_dir / "antenna"
        self.gain_dir = self.root_dir / "gain" / gain_method
        self._validate_data_root()

        # Build sample list
        all_samples = self._build_sample_list()
        train_samples, val_samples, test_samples = self._split_samples(
            all_samples, train_ratio, val_ratio, seed
        )

        if split == "train":
            self.samples = train_samples
        elif split == "val":
            self.samples = val_samples
        elif split == "test":
            self.samples = test_samples
        else:
            self.samples = all_samples

        self.transform = transform

    def _validate_data_root(self):
        required_paths = [self.buildings_dir, self.antenna_dir, self.gain_dir]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            missing_text = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "RadioMapSeer data was not found. Expected the following paths:\n"
                f"{missing_text}\n\n"
                "Download the dataset separately and place it under the configured data root, "
                "or update data.root_dir in the config file."
            )

    def _build_sample_list(self):
        """Build list of (map_id, tx_idx) tuples."""
        samples = []
        building_files = sorted(
            [f.stem for f in self.buildings_dir.glob("*.png")]
        )

        for map_id in building_files:
            antenna_file = self.antenna_dir / f"{map_id}.json"
            if not antenna_file.exists():
                continue

            with open(antenna_file, "r") as f:
                positions = json.load(f)

            for tx_idx in range(len(positions)):
                gain_file = self.gain_dir / f"{map_id}_{tx_idx}.png"
                if gain_file.exists():
                    samples.append((map_id, tx_idx))

        return samples

    def _split_samples(self, samples, train_ratio, val_ratio, seed):
        """Split samples into train/val/test by map_id to avoid leakage."""
        rng = random.Random(seed)

        # Get unique map IDs
        map_ids = sorted(list(set(s[0] for s in samples)))
        rng.shuffle(map_ids)

        n_total = len(map_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_ids = set(map_ids[:n_train])
        val_ids = set(map_ids[n_train : n_train + n_val])
        test_ids = set(map_ids[n_train + n_val :])

        train_samples = [s for s in samples if s[0] in train_ids]
        val_samples = [s for s in samples if s[0] in val_ids]
        test_samples = [s for s in samples if s[0] in test_ids]

        return train_samples, val_samples, test_samples

    def _create_antenna_heatmap(self, tx_x, tx_y, sigma=5.0):
        """Create a Gaussian heatmap centered at Tx position."""
        x = np.arange(0, self.img_size, 1, dtype=np.float32)
        y = np.arange(0, self.img_size, 1, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        heatmap = np.exp(-((xx - tx_x) ** 2 + (yy - tx_y) ** 2) / (2 * sigma ** 2))
        return heatmap

    @staticmethod
    def _to_image_coordinates(tx_x, tx_y, image_shape):
        """Convert bottom-left-origin Tx coordinates to image coordinates."""
        height, width = image_shape

        try:
            tx_x = float(tx_x)
            tx_y = float(tx_y)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Tx coordinates: ({tx_x}, {tx_y})") from exc

        if not (0 <= tx_x < width and 0 <= tx_y < height):
            raise ValueError(
                f"Tx coordinates ({tx_x}, {tx_y}) are outside image bounds "
                f"{width}x{height}"
            )

        # JSON y increases upward; NumPy/PIL row indices increase downward.
        return tx_x, float(height - 1) - tx_y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        map_id, tx_idx = self.samples[idx]

        # Load building map
        building_path = self.buildings_dir / f"{map_id}.png"
        building = np.array(Image.open(building_path).convert("L"), dtype=np.float32) / 255.0

        # Load Tx position
        antenna_file = self.antenna_dir / f"{map_id}.json"
        with open(antenna_file, "r") as f:
            positions = json.load(f)
        tx_x_raw, tx_y_raw = positions[tx_idx]
        tx_x, tx_y = self._to_image_coordinates(
            tx_x_raw, tx_y_raw, building.shape
        )

        # Create antenna heatmap
        antenna_heatmap = self._create_antenna_heatmap(tx_x, tx_y)

        # Load gain map
        gain_path = self.gain_dir / f"{map_id}_{tx_idx}.png"
        gain = np.array(Image.open(gain_path).convert("L"), dtype=np.float32) / 255.0

        # Stack input: [building_map, antenna_heatmap]
        input_map = np.stack([building, antenna_heatmap], axis=0)  # (2, H, W)
        target = gain[np.newaxis, ...]  # (1, H, W)

        if self.transform:
            # Apply same transform to input and target
            seed_t = torch.randint(2147483647, (1,)).item()
            torch.manual_seed(seed_t)
            input_map = self.transform(torch.from_numpy(input_map))
            torch.manual_seed(seed_t)
            target = self.transform(torch.from_numpy(target))
        else:
            input_map = torch.from_numpy(input_map)
            target = torch.from_numpy(target)

        return {
            "input": input_map,           # (2, H, W)
            "target": target,             # (1, H, W)
            "building": torch.from_numpy(building),  # (H, W)
            "tx_position": torch.tensor([tx_x, tx_y], dtype=torch.float32),
            "map_id": map_id,
            "tx_idx": tx_idx,
        }

    def get_sample_info(self):
        """Return dataset statistics."""
        return {
            "total_samples": len(self.samples),
            "num_maps": len(set(s[0] for s in self.samples)),
            "gain_method": self.gain_method,
            "split": self.split,
        }


def _select_subset(dataset, fraction, seed):
    """Select the same deterministic subset in every distributed process."""
    if not 0 < fraction <= 1:
        raise ValueError(f"subset fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return dataset

    subset_size = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    return Subset(dataset, indices.tolist())


def get_dataloaders(
    config,
    *,
    batch_size=None,
    distributed=False,
    rank=0,
    world_size=1,
    subset_frac=1.0,
):
    """Create train/val/test dataloaders for single-process or DDP training.

    ``batch_size`` is the number of samples loaded by the current process. In
    DDP mode the training entry point derives it from the configured global
    batch size before calling this function.
    """
    data_cfg = config["data"]
    per_process_batch_size = batch_size or config["training"]["batch_size"]

    train_dataset = RadioMapSeerDataset(
        root_dir=data_cfg["root_dir"],
        gain_method=data_cfg["gain_method"],
        img_size=data_cfg["img_size"],
        split="train",
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=config["training"]["seed"],
    )

    val_dataset = RadioMapSeerDataset(
        root_dir=data_cfg["root_dir"],
        gain_method=data_cfg["gain_method"],
        img_size=data_cfg["img_size"],
        split="val",
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=config["training"]["seed"],
    )

    test_dataset = RadioMapSeerDataset(
        root_dir=data_cfg["root_dir"],
        gain_method=data_cfg["gain_method"],
        img_size=data_cfg["img_size"],
        split="test",
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=config["training"]["seed"],
    )

    seed = int(config["training"]["seed"])
    train_dataset = _select_subset(train_dataset, subset_frac, seed)
    val_dataset = _select_subset(val_dataset, subset_frac, seed + 1)

    train_sampler = None
    val_sampler = None
    test_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_process_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=per_process_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        sampler=test_sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )

    return train_loader, val_loader, test_loader
