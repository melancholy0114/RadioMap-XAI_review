"""Reproducibility helpers shared by training and evaluation entry points.

The data split seed is intentionally independent from the training seed. This
lets repeated runs vary initialization, data order, and other stochastic
training effects while keeping exactly the same train/validation/test maps.
"""

from copy import deepcopy
import os
import random

import numpy as np
import torch


def get_training_seed(config):
    """Return the seed controlling stochastic training behavior."""
    return int(config["training"]["seed"])


def get_split_seed(config):
    """Return the map-split seed, with legacy-config compatibility."""
    return int(config["data"].get("split_seed", get_training_seed(config)))


def get_evaluation_seed(config):
    """Return the fixed seed used to select comparable evaluation subsets."""
    return int(config.get("evaluation", {}).get("seed", get_split_seed(config)))


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch without forcing deterministic kernels."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_metadata(config):
    """Return checkpoint metadata needed to reproduce a seeded run."""
    return {
        "training_seed": get_training_seed(config),
        "split_seed": get_split_seed(config),
    }


def validate_seed_metadata(checkpoint, config):
    """Reject a new checkpoint assigned to a different controlled seed.

    Checkpoints created before seed metadata was introduced remain compatible.
    """
    expected = seed_metadata(config)
    for key, expected_value in expected.items():
        recorded_value = checkpoint.get(key)
        if recorded_value is not None and int(recorded_value) != expected_value:
            raise ValueError(
                f"Checkpoint {key}={recorded_value} does not match "
                f"the configured {key}={expected_value}"
            )


def _append_output_tag(path, tag):
    path = os.fspath(path)
    if os.path.basename(os.path.normpath(path)) == tag:
        return path
    return os.path.join(path, tag)


def configure_seeded_run(
    config,
    *,
    training_seed=None,
    split_seed=None,
    isolate_outputs=False,
):
    """Return a copied config with explicit, independently controlled seeds.

    When ``isolate_outputs`` is true, every configured ``*_dir`` is placed
    below ``seed_<training_seed>``. Existing single-run commands therefore
    remain unchanged, while the multi-seed CLI cannot overwrite another run.
    """
    resolved = deepcopy(config)
    original_split_seed = get_split_seed(resolved)

    if training_seed is not None:
        resolved["training"]["seed"] = int(training_seed)
    resolved["data"]["split_seed"] = (
        original_split_seed if split_seed is None else int(split_seed)
    )

    if isolate_outputs:
        output_tag = f"seed_{get_training_seed(resolved)}"
        for key, path in resolved.get("output", {}).items():
            if key.endswith("_dir") and path is not None:
                resolved["output"][key] = _append_output_tag(path, output_tag)

    return resolved
