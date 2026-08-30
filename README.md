# Beyond Accuracy: An Explainable Radio Map Prediction Framework via Physical Alignment and Attribution-Based Diagnostics

This repository contains the official PyTorch implementation of the paper **"Beyond Accuracy: An Explainable Radio Map Prediction Framework via Physical Alignment and Attribution-Based Diagnostics"**. The project combines configurable Restormer and RadioUNet_C predictors with post-hoc explanation methods, physics-inspired priors, and evaluation utilities for studying whether model behavior aligns with wireless propagation structure.

![Framework](assets/archi.png)

## Highlights

- Config-selectable Restormer and CNN-based RadioUNet_C prediction backbones
- L1 and Physics-L1 training for both backbones, including multi-GPU DDP
- Integrated Gradients, Grad-CAM, and occlusion sensitivity for explanation analysis
- Physics-inspired priors for line-of-sight, obstruction, and directional structure
- Metrics for faithfulness, physical alignment, stability, and consistency
- Analysis scripts for ID/OOD behavior, explanation drift, and failure cases

## Repository Layout

```text
radiomap-xai/
├── analysis/          # Evaluation and analysis scripts
├── assets/            # Lightweight figures used in the README
├── configs/           # Experiment configurations
├── datasets/          # RadioMapSeer dataset loader
├── explanation/       # Explanation methods
├── inference/         # Inference utilities
├── losses/            # Training losses
├── metrics/           # Explanation metrics
├── model/             # Separate Restormer/RadioUNet implementations and factory
├── priors/            # Physics-inspired priors
├── scripts/           # Helper scripts
├── training/          # Training and validation pipeline
├── visualization/     # Plotting helpers
└── run_experiment.py  # End-to-end pipeline entry point
```

## Installation

Create a Python environment with Python 3.10+ and install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

This repository does not redistribute the RadioMapSeer dataset. Download the dataset from the official source and place the processed files under `data/` with the following structure:

```text
data/
├── antenna/
│   └── <map_id>.json
├── gain/
│   └── DPM/
│       └── <map_id>_<tx_idx>.png
└── png/
    └── buildings_complete/
        └── <map_id>.png
```

By default, the code expects `data.root_dir` in `configs/config.yaml` to point to `./data`.

## Backbones and configs

The training scripts select a backbone through `model.name`; they do not contain
architecture-specific construction code.

| Experiment | Config | Checkpoint root |
|---|---|---|
| Restormer-L1 | `configs/config.yaml` | `outputs/checkpoints` |
| Restormer-Physics-L1 | `configs/config_ablation*.yaml` | `outputs/improved_checkpoints` |
| RadioUNet-L1 | `configs/config_radiounet.yaml` | `outputs/radiounet_c/l1/checkpoints` |
| RadioUNet-Physics-L1 | `configs/config_radiounet_ablation*.yaml` | `outputs/radiounet_c/physics_l1/checkpoints` |

`RadioUNet_C` here is the first U-Net predictor for the complete-city,
no-measurement setting: its two inputs are the building map and Tx heatmap. Its
topology follows the authors' [public RadioUNet implementation](https://github.com/RonLevie/RadioUNet). The optional retrospective second U-Net/WNet curriculum
is deliberately excluded so this experiment compares one prediction backbone
against another under the same project training protocol.

## Quick Start

Run a short four-GPU DDP smoke test first. The smoke test does not write
checkpoints or TensorBoard logs:

```bash
torchrun --standalone --nproc_per_node=4 training/train.py \
  --config configs/config.yaml \
  --smoke-test-batches 50
```

Train Restormer-L1 with four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 training/train.py \
  --config configs/config.yaml
```

`training.batch_size` is the global batch size. With the default value of 8,
four DDP processes load 2 samples per GPU. Gradient accumulation is 4, so the
effective batch size remains 32. The global batch size must be divisible by
the number of DDP processes.

To select specific GPUs, expose them before launching `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  training/train.py --config configs/config.yaml
```

Single-GPU training remains available, but the configured global batch size
must fit on that GPU:

```bash
python training/train.py --config configs/config.yaml --gpus 0
```

Train Restormer-Physics-L1 with four-GPU DDP in two stages. First warm-start
from the trained Restormer-L1 baseline and retain the 20-epoch ablation:

```bash
torchrun --standalone --nproc_per_node=4 training/train_physics.py \
  --config configs/config_ablation.yaml \
  --resume outputs/checkpoints/best_model.pth
```

Then continue the same Physics-L1 optimizer and scheduler state from epoch 20
through epoch 50:

```bash
torchrun --standalone --nproc_per_node=4 training/train_physics.py \
  --config configs/config_ablation_50ep.yaml \
  --resume outputs/improved_checkpoints/final_model.pth
```

Physics-L1 checkpoints and TensorBoard logs are written to
`outputs/improved_checkpoints` and `outputs/logs_physics`, respectively. Add
`--smoke-test-batches 50` to the first command to verify the setup without
writing either output.

Train RadioUNet-L1 with the same DDP/data/loss protocol:

```bash
torchrun --standalone --nproc_per_node=4 training/train.py \
  --config configs/config_radiounet.yaml
```

Then warm-start the 20-epoch RadioUNet-Physics-L1 stage from the matching
RadioUNet-L1 checkpoint:

```bash
torchrun --standalone --nproc_per_node=4 training/train_physics.py \
  --config configs/config_radiounet_ablation.yaml \
  --resume outputs/radiounet_c/l1/checkpoints/best_model.pth
```

Continue that RadioUNet-Physics-L1 run from epoch 20 through epoch 50:

```bash
torchrun --standalone --nproc_per_node=4 training/train_physics.py \
  --config configs/config_radiounet_ablation_50ep.yaml \
  --resume outputs/radiounet_c/physics_l1/checkpoints/final_model.pth
```

The RadioUNet and Restormer directory trees are disjoint, so these runs cannot
overwrite one another. New checkpoints also record `model_name` and reject a
mismatched config/checkpoint pair. Legacy checkpoints without this field remain
compatible and are interpreted as Restormer checkpoints.

## Controlled multi-seed experiments

For statistical comparisons, `data.split_seed` and `training.seed` have
different roles:

- `data.split_seed` fixes train/validation/test map membership and evaluation
  subsets. Keep it unchanged for every compared run.
- `training.seed` changes model initialization and training data order. Vary
  this value across at least three independently trained runs.

Existing commands remain backward compatible and keep their original output
paths. When `--seed` is supplied explicitly, all outputs are automatically
placed in a `seed_<N>` subdirectory and checkpoints record both seeds.

Run three Restormer-L1 seeds sequentially on the same four GPUs:

```bash
python scripts/run_multi_seed.py \
  --trainer l1 \
  --config configs/config.yaml \
  --seeds 42 123 2026 \
  --split-seed 42 \
  --nproc-per-node 4 \
  --gpus 0,1,2,3
```

The checkpoints are written to, for example,
`outputs/checkpoints/seed_123/best_model.pth`. Train the paired Physics-L1
warm-start runs from the matching L1 seed:

```bash
python scripts/run_multi_seed.py \
  --trainer physics \
  --config configs/config_ablation.yaml \
  --seeds 42 123 2026 \
  --split-seed 42 \
  --resume-template 'outputs/checkpoints/seed_{seed}/best_model.pth' \
  --nproc-per-node 4 \
  --gpus 0,1,2,3
```

Continue each Physics-L1 run from epoch 20 to epoch 50:

```bash
python scripts/run_multi_seed.py \
  --trainer physics \
  --config configs/config_ablation_50ep.yaml \
  --seeds 42 123 2026 \
  --split-seed 42 \
  --resume-template 'outputs/improved_checkpoints/seed_{seed}/final_model.pth' \
  --full-resume \
  --nproc-per-node 4 \
  --gpus 0,1,2,3
```

Use the same commands with `config_radiounet*.yaml` and the corresponding
`outputs/radiounet_c/.../seed_<N>` paths for RadioUNet. The launcher runs seeds
sequentially; add `--dry-run` to inspect every generated `torchrun` command.

After training, evaluate every checkpoint on the identical full test split:

```bash
python analysis/evaluate_multiseed.py \
  --config configs/config.yaml \
  --seeds 42 123 2026 \
  --checkpoint-template 'outputs/checkpoints/seed_{seed}/best_model.pth' \
  --output outputs/multiseed/restormer_l1_test.json

python analysis/evaluate_multiseed.py \
  --config configs/config_ablation_50ep.yaml \
  --seeds 42 123 2026 \
  --checkpoint-template 'outputs/improved_checkpoints/seed_{seed}/best_model.pth' \
  --output outputs/multiseed/restormer_physics_l1_test.json
```

Each result reports per-seed metrics, per-map metrics, mean, sample standard
deviation, and a two-sided 95% Student-t confidence interval. Compare paired
L1/Physics-L1 runs by matching both seed and test map:

```bash
python analysis/compare_multiseed.py \
  --baseline outputs/multiseed/restormer_l1_test.json \
  --candidate outputs/multiseed/restormer_physics_l1_test.json \
  --output outputs/multiseed/restormer_l1_vs_physics.json
```

Do not use `--limit-samples` for manuscript results; it exists only for a quick
evaluation smoke test.

Run inference with a trained checkpoint:

```bash
python inference/infer.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pth --num_samples 10
```

For RadioUNet-L1, use the matching config and checkpoint:

```bash
python inference/infer.py \
  --config configs/config_radiounet.yaml \
  --checkpoint outputs/radiounet_c/l1/checkpoints/best_model.pth \
  --num_samples 10
```

Run the end-to-end pipeline:

```bash
python run_experiment.py --config configs/config.yaml --skip-training
```

Generate dataset visualizations:

```bash
python scripts/visualize_dataset.py
```

## Example Outputs

<table>
  <tr>
    <td><img src="assets/examples/explanation_comparison.png" alt="Explanation comparison" width="420"></td>
    <td><img src="assets/examples/drift_vs_error.png" alt="Drift versus error" width="420"></td>
  </tr>
</table>

## Notes

- Checkpoints, logs, and full experiment outputs are intentionally excluded from the public release.
- The repository keeps output paths relative so results can be reproduced locally under `outputs/`.
- If the dataset path is missing, the loader raises a clear error describing the expected directory layout.

## License

Released under the MIT License.
