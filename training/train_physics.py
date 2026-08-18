"""Physics-weighted L1 training with single-GPU and DDP support.

The network architecture and data split match the L1 baseline. Physics-L1
changes only the training loss by upweighting LoS and near-transmitter pixels.

Typical two-stage workflow:

1. Warm-start the 20-epoch Physics-L1 run from the L1 baseline:

   torchrun --standalone --nproc_per_node=4 training/train_physics.py \
       --config configs/config_ablation.yaml \
       --resume outputs/checkpoints/best_model.pth

2. Continue the same Physics-L1 optimizer trajectory from epoch 20 to 50:

   torchrun --standalone --nproc_per_node=4 training/train_physics.py \
       --config configs/config_ablation_50ep.yaml \
       --resume outputs/improved_checkpoints/final_model.pth
"""

import argparse
from contextlib import nullcontext
import os
import random
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import yaml

from datasets.radiomapseer_dataset import get_dataloaders
from losses.loss import PhysicsWeightedL1Loss
from model.radio_map_model import Restormer
from priors.los_mask import compute_los_mask_fast
from training.validate import validate


_TRAINING_VARIANT = "physics_weighted_l1"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Physics-L1 model")
    parser.add_argument("--config", type=str, default="configs/config_ablation.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to warm-start or resume")
    parser.add_argument(
        "--full-resume",
        "--full_resume",
        dest="full_resume",
        action="store_true",
        help=(
            "Restore optimizer, scheduler, AMP scaler, and epoch. "
            "The config training.full_resume setting has the same effect."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Single-GPU compatibility option, e.g. '0'. For multi-GPU training, "
            "launch with torchrun and select devices through CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument("--subset", type=float, default=1.0, help="Use a fraction of the data")
    parser.add_argument(
        "--smoke-test-batches",
        type=int,
        default=None,
        help="Run only N training batches and exit without validation or checkpoints",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Override the number of batches between progress messages",
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def setup_process(args):
    """Initialize one NCCL process per GPU when launched through torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if args.gpus is not None:
            raise ValueError(
                "Do not pass --gpus to torchrun. Select devices with "
                "CUDA_VISIBLE_DEVICES=0,1,2,3 instead."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("Physics-L1 DDP with NCCL requires CUDA")

        local_rank_text = os.environ.get("LOCAL_RANK")
        if local_rank_text is None and args.local_rank is None:
            raise RuntimeError("LOCAL_RANK was not provided by the distributed launcher")
        local_rank = int(local_rank_text) if local_rank_text is not None else args.local_rank
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA devices are visible"
            )

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
        )
        return True, dist.get_rank(), dist.get_world_size(), local_rank, device

    rank = 0
    local_rank = 0
    if args.gpus is not None:
        gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
        if len(gpu_ids) != 1:
            raise ValueError(
                "Multi-GPU DataParallel is not supported. Use: "
                "torchrun --standalone --nproc_per_node=4 training/train_physics.py "
                "--config configs/config_ablation.yaml "
                "--resume outputs/checkpoints/best_model.pth"
            )
        local_rank = gpu_ids[0]
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device)
    return False, rank, 1, local_rank, device


def cleanup_process():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def main_print(rank, message):
    if is_main_process(rank):
        print(message, flush=True)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def normalize_state_dict(state_dict):
    """Accept weights saved from plain, DataParallel, or DDP models."""
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def process_rss_gib():
    try:
        with open("/proc/self/status", "r") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return 0.0


def reduce_mean(value, distributed, world_size):
    reduced = value.detach().clone()
    if distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= world_size
    return reduced.item()


def resolve_output_dirs(config):
    output_config = config["output"]
    checkpoint_dir = output_config.get("physics_checkpoint_dir")
    if checkpoint_dir is None:
        baseline_dir = output_config["checkpoint_dir"]
        checkpoint_dir = os.path.join(
            os.path.dirname(baseline_dir),
            "improved_checkpoints",
        )

    log_dir = output_config.get(
        "physics_log_dir",
        f"{output_config['log_dir']}_physics",
    )
    return checkpoint_dir, log_dir


def save_physics_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_loss,
    physics_alpha,
    path,
):
    state = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "training_variant": _TRAINING_VARIANT,
        "physics_alpha": physics_alpha,
    }
    torch.save(state, path)


def restore_physics_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    path,
    device,
    rank,
    full_resume,
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(
        normalize_state_dict(checkpoint["model_state_dict"])
    )

    if not full_resume:
        main_print(
            rank,
            f"Warm-started Physics-L1 from {path} (fresh optimizer, epoch 0)",
        )
        return 0, float("inf")

    checkpoint_variant = checkpoint.get("training_variant")
    if checkpoint_variant != _TRAINING_VARIANT:
        raise ValueError(
            "Full resume requires a Physics-L1 checkpoint created by this "
            f"training pipeline; found training_variant={checkpoint_variant!r}. "
            "Use warm-start mode for a Baseline or legacy checkpoint."
        )
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("Full resume requires optimizer_state_dict in the checkpoint")

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "amp_scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    main_print(
        rank,
        f"Fully resumed Physics-L1 from epoch {start_epoch}, "
        f"best val loss: {best_val_loss:.6f}",
    )
    return start_epoch, best_val_loss


def compute_physics_weight_map(building, tx_pos):
    """Build the LoS plus distance-decay weight map used by Physics-L1."""
    height, width = building.shape

    los = compute_los_mask_fast(
        building,
        tx_pos,
        n_directions=360,
        max_radius=200,
    )

    tx_x, tx_y = tx_pos
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    distance = np.sqrt((x_coords - tx_x) ** 2 + (y_coords - tx_y) ** 2) + 1.0
    distance_weight = 1.0 / (1.0 + 0.01 * distance)

    weight_map = 1.0 + 2.0 * los + distance_weight
    return weight_map.astype(np.float32)


def build_physics_weight_batch(batch, device):
    buildings = batch["building"].numpy()
    tx_positions = batch["tx_position"].numpy()
    weight_maps = [
        compute_physics_weight_map(building, tx_position)
        for building, tx_position in zip(buildings, tx_positions)
    ]
    return (
        torch.from_numpy(np.stack(weight_maps))
        .unsqueeze(1)
        .to(device, non_blocking=True)
    )


def train_physics(
    config,
    resume_path=None,
    subset_frac=1.0,
    *,
    full_resume=False,
    distributed=False,
    rank=0,
    world_size=1,
    local_rank=0,
    device=None,
    smoke_test_batches=None,
    log_interval=None,
):
    """Train Physics-L1 with one process per GPU in distributed mode."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    global_batch_size = int(config["training"]["batch_size"])
    if global_batch_size < 1:
        raise ValueError("training.batch_size must be positive")
    if distributed and global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by world size {world_size}"
        )
    per_process_batch_size = global_batch_size // world_size if distributed else global_batch_size

    train_loader, val_loader, _ = get_dataloaders(
        config,
        batch_size=per_process_batch_size,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        subset_frac=subset_frac,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "The training loader is empty. Increase --subset or reduce the global batch size."
        )

    main_print(
        rank,
        f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples",
    )

    model_config = config["model"]
    model = Restormer(
        inp_channels=model_config["inp_channels"],
        out_channels=model_config["out_channels"],
        dim=model_config["dim"],
        num_blocks=model_config["num_blocks"],
        num_refinement_blocks=model_config["num_refinement_blocks"],
        heads=model_config["heads"],
        ffn_expansion_factor=model_config["ffn_expansion_factor"],
        bias=model_config["bias"],
        LayerNorm_type=model_config["LayerNorm_type"],
    ).to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        main_print(rank, f"Using DistributedDataParallel on {world_size} GPUs")
    else:
        main_print(rank, f"Using device: {device}")

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    main_print(rank, f"Model parameters: {parameter_count:,}")

    physics_alpha = float(config["loss"].get("physics_alpha", 0.3))
    criterion = PhysicsWeightedL1Loss(alpha=physics_alpha)
    validation_criterion = nn.L1Loss()
    main_print(rank, f"Loss: PhysicsWeightedL1 (alpha={physics_alpha:g})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["T_max"]),
        eta_min=float(config["training"]["eta_min"]),
    )

    amp_enabled = device.type == "cuda"
    amp_init_scale = float(config["training"].get("amp_init_scale", 1.0))
    if amp_init_scale <= 0:
        raise ValueError("training.amp_init_scale must be positive")
    scaler = GradScaler("cuda", enabled=amp_enabled, init_scale=amp_init_scale)

    start_epoch = 0
    best_val_loss = float("inf")
    if resume_path:
        start_epoch, best_val_loss = restore_physics_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            resume_path,
            device,
            rank,
            full_resume,
        )

    epochs = int(config["training"]["epochs"])
    if start_epoch >= epochs:
        raise ValueError(
            f"Checkpoint resumes at epoch {start_epoch}, but the config ends at epoch {epochs}. "
            "Use warm-start mode for a Baseline checkpoint, or provide an earlier Physics-L1 checkpoint."
        )
    if distributed:
        dist.barrier()

    checkpoint_dir, log_dir = resolve_output_dirs(config)
    writer = None
    if is_main_process(rank) and smoke_test_batches is None:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    if distributed:
        dist.barrier()

    grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("training.grad_accum_steps must be positive")
    effective_batch_size = global_batch_size * grad_accum_steps
    main_print(
        rank,
        f"Global batch: {global_batch_size} = {per_process_batch_size}/GPU x {world_size} GPU(s); "
        f"gradient accumulation: {grad_accum_steps}; effective batch: {effective_batch_size}; "
        f"AMP initial scale: {scaler.get_scale():g}",
    )

    configured_interval = int(config["training"].get("log_interval", 50))
    progress_interval = int(configured_interval if log_interval is None else log_interval)
    if progress_interval < 1:
        raise ValueError("log interval must be positive")
    if smoke_test_batches is not None:
        if smoke_test_batches < 1:
            raise ValueError("--smoke-test-batches must be positive")
        progress_interval = min(progress_interval, max(1, smoke_test_batches // 5))
        main_print(
            rank,
            f"Smoke test: {smoke_test_batches} Physics-L1 batches; no checkpoints will be written",
        )

    grad_clip = float(config["training"]["grad_clip"])
    smoke_initial_rss = process_rss_gib()

    for epoch in range(start_epoch, epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        epoch_totals = torch.zeros(2, dtype=torch.float64, device=device)
        batches_this_epoch = len(train_loader)
        if smoke_test_batches is not None:
            batches_this_epoch = min(batches_this_epoch, smoke_test_batches)

        if distributed:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()
        main_print(rank, f"Epoch [{epoch + 1}/{epochs}] started ({batches_this_epoch} batches)")

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= batches_this_epoch:
                break

            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            weight_tensor = build_physics_weight_batch(batch, device)

            window_start = (batch_idx // grad_accum_steps) * grad_accum_steps
            window_end = min(window_start + grad_accum_steps, batches_this_epoch)
            window_size = window_end - window_start
            should_step = batch_idx + 1 == window_end
            sync_context = model.no_sync() if distributed and not should_step else nullcontext()

            with sync_context:
                with autocast("cuda", enabled=amp_enabled):
                    outputs = model(inputs)
                    raw_loss = criterion(
                        outputs,
                        targets,
                        weight_map=weight_tensor,
                    )
                    loss = raw_loss / window_size
                scaler.scale(loss).backward()

            if should_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)

            local_batch_size = inputs.shape[0]
            epoch_totals[0] += raw_loss.detach().to(torch.float64) * local_batch_size
            epoch_totals[1] += local_batch_size

            completed_batches = batch_idx + 1
            if completed_batches % progress_interval == 0 or completed_batches == batches_this_epoch:
                mean_loss = reduce_mean(raw_loss, distributed, world_size)
                elapsed = time.time() - epoch_start
                seconds_per_batch = elapsed / completed_batches
                eta_seconds = seconds_per_batch * (batches_this_epoch - completed_batches)
                processed_samples = min(
                    completed_batches * global_batch_size,
                    len(train_loader.dataset),
                )
                main_print(
                    rank,
                    f"Epoch [{epoch + 1}/{epochs}] Batch [{completed_batches}/{batches_this_epoch}] "
                    f"Physics Loss: {mean_loss:.6f} | {seconds_per_batch:.3f}s/batch | "
                    f"ETA: {eta_seconds / 60:.1f} min | "
                    f"AMP scale: {scaler.get_scale():.0f} | Optimizer steps: {optimizer_steps} | "
                    f"Samples: {processed_samples}/{len(train_loader.dataset)}",
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.time() - epoch_start

        if distributed:
            dist.all_reduce(epoch_totals, op=dist.ReduceOp.SUM)
        average_train_loss = epoch_totals[0].item() / max(epoch_totals[1].item(), 1.0)

        if smoke_test_batches is not None:
            current_rss = process_rss_gib()
            diagnostics = torch.tensor(
                [
                    current_rss,
                    current_rss - smoke_initial_rss,
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    if device.type == "cuda"
                    else 0.0,
                ],
                dtype=torch.float64,
                device=device,
            )
            if distributed:
                dist.all_reduce(diagnostics, op=dist.ReduceOp.MAX)
            main_print(
                rank,
                f"Physics-L1 smoke test complete: loss={average_train_loss:.6f}, "
                f"time={train_seconds:.1f}s, {train_seconds / batches_this_epoch:.3f}s/batch, "
                f"optimizer_steps={optimizer_steps}, "
                f"max RSS/rank={diagnostics[0].item():.2f} GiB, "
                f"max RSS growth/rank={diagnostics[1].item():.2f} GiB, "
                f"max GPU allocated/rank={diagnostics[2].item():.2f} GiB",
            )
            return {
                "loss": average_train_loss,
                "seconds_per_batch": train_seconds / batches_this_epoch,
                "optimizer_steps": optimizer_steps,
                "max_rss_gib": diagnostics[0].item(),
                "max_rss_growth_gib": diagnostics[1].item(),
                "max_gpu_allocated_gib": diagnostics[2].item(),
            }

        validation_start = time.time()
        val_loss, val_rmse, val_mae = validate(
            model,
            val_loader,
            validation_criterion,
            device,
        )
        validation_seconds = time.time() - validation_start
        learning_rate = optimizer.param_groups[0]["lr"]

        if writer is not None:
            writer.add_scalar("Loss/train", average_train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metrics/val_rmse", val_rmse, epoch)
            writer.add_scalar("Metrics/val_mae", val_mae, epoch)
            writer.add_scalar("LR", learning_rate, epoch)
            writer.add_scalar("Time/train_seconds", train_seconds, epoch)
            writer.add_scalar("Time/validation_seconds", validation_seconds, epoch)

        main_print(
            rank,
            f"Epoch [{epoch + 1}/{epochs}] Train: {train_seconds / 60:.1f} min | "
            f"Validation: {validation_seconds / 60:.1f} min | "
            f"Physics Train Loss: {average_train_loss:.6f} | Val L1: {val_loss:.6f} | "
            f"Val RMSE: {val_rmse:.6f} ({val_rmse * 139:.2f} dB) | "
            f"Val MAE: {val_mae:.6f} | LR: {learning_rate:.2e}",
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        if optimizer_steps > 0:
            scheduler.step()
        else:
            main_print(rank, "Warning: all optimizer steps were skipped; LR scheduler was not advanced")

        if is_main_process(rank):
            if improved:
                save_physics_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_val_loss,
                    physics_alpha,
                    os.path.join(checkpoint_dir, "best_model.pth"),
                )
                print(f"  -> New best Physics-L1 model saved (val_loss: {best_val_loss:.6f})", flush=True)
            if (epoch + 1) % 5 == 0:
                save_physics_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_val_loss,
                    physics_alpha,
                    os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
                )
        if distributed:
            dist.barrier()

    if is_main_process(rank):
        save_physics_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epochs - 1,
            best_val_loss,
            physics_alpha,
            os.path.join(checkpoint_dir, "final_model.pth"),
        )
        if writer is not None:
            writer.close()
        print(f"\nPhysics-L1 training complete. Best val loss: {best_val_loss:.6f}", flush=True)
    if distributed:
        dist.barrier()


def main():
    args = parse_args()
    distributed = False
    try:
        distributed, rank, world_size, local_rank, device = setup_process(args)
        config = load_config(args.config)
        resume_path = args.resume or config["training"].get("resume")
        full_resume = bool(
            args.full_resume
            or config["training"].get("full_resume", False)
        )
        train_physics(
            config,
            resume_path=resume_path,
            subset_frac=args.subset,
            full_resume=full_resume,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            smoke_test_batches=args.smoke_test_batches,
            log_interval=args.log_interval,
        )
    finally:
        if distributed:
            cleanup_process()


if __name__ == "__main__":
    main()
