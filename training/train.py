"""Training pipeline with single-GPU and DistributedDataParallel support."""

import argparse
from contextlib import nullcontext
import os
import sys
import time

# Ensure imports work when launched from the repository root or this file's directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import yaml

from datasets.radiomapseer_dataset import get_dataloaders
from losses import build_loss
from model import (
    build_model,
    checkpoint_metadata,
    get_model_name,
    normalize_state_dict,
    validate_checkpoint_model,
)
from training.validate import validate
from utils import (
    configure_seeded_run,
    get_split_seed,
    get_training_seed,
    seed_everything,
    seed_metadata,
    validate_seed_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a radio map prediction model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override training.seed. Explicit seeded runs are written below "
            "seed_<N> subdirectories so repeated runs cannot overwrite each other."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override the fixed train/validation/test map split seed",
    )
    parser.add_argument("--subset", type=float, default=1.0, help="Use fraction of training data")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Single-GPU compatibility option, e.g. '0'. For multi-GPU training, "
            "launch with torchrun and select devices through CUDA_VISIBLE_DEVICES."
        ),
    )
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
    """Initialize torch.distributed when launched through torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if args.gpus is not None:
            raise ValueError(
                "Do not pass --gpus to torchrun. Select devices with "
                "CUDA_VISIBLE_DEVICES=0,1,2,3 instead."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with the NCCL backend requires CUDA")

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
                "Multi-GPU nn.DataParallel has been removed because it was unstable for this model. "
                "Use: torchrun --standalone --nproc_per_node=4 training/train.py "
                "--config configs/config.yaml"
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


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path, config):
    state = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        **seed_metadata(config),
        **checkpoint_metadata(model),
    }
    torch.save(state, path)


def restore_checkpoint(model, optimizer, scheduler, path, device, rank, config):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    validate_checkpoint_model(checkpoint, model)
    validate_seed_metadata(checkpoint, config)
    state_dict = normalize_state_dict(checkpoint["model_state_dict"])
    unwrap_model(model).load_state_dict(state_dict)

    start_epoch = 0
    best_val_loss = float("inf")
    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            main_print(
                rank,
                f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}",
            )
        except Exception as exc:
            main_print(rank, f"Could not restore optimizer state ({exc}); using a fresh optimizer")
    else:
        main_print(rank, f"Warm-started model weights from {path}")

    return start_epoch, best_val_loss


def process_rss_gib():
    """Return this process's current resident memory without extra dependencies."""
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


def train(
    config,
    resume_path=None,
    subset_frac=1.0,
    *,
    distributed=False,
    rank=0,
    world_size=1,
    local_rank=0,
    device=None,
    smoke_test_batches=None,
    log_interval=None,
):
    """Train with one process per GPU when distributed is true."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_seed = get_training_seed(config)
    split_seed = get_split_seed(config)
    seed_everything(training_seed)
    main_print(
        rank,
        f"Seeds: training={training_seed}, data_split={split_seed}",
    )

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

    model = build_model(config["model"]).to(device)
    backbone_name = get_model_name(model)
    main_print(rank, f"Backbone: {backbone_name}")

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

    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    main_print(rank, f"Model parameters: {parameter_count:,}")

    criterion = build_loss(config)
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

    start_epoch = 0
    best_val_loss = float("inf")
    if resume_path:
        start_epoch, best_val_loss = restore_checkpoint(
            model, optimizer, scheduler, resume_path, device, rank, config
        )
    if distributed:
        dist.barrier()

    writer = None
    checkpoint_dir = config["output"]["checkpoint_dir"]
    if is_main_process(rank) and smoke_test_batches is None:
        os.makedirs(config["output"]["log_dir"], exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=config["output"]["log_dir"])
    if distributed:
        dist.barrier()

    amp_enabled = device.type == "cuda"
    amp_init_scale = float(config["training"].get("amp_init_scale", 1.0))
    if amp_init_scale <= 0:
        raise ValueError("training.amp_init_scale must be positive")
    scaler = GradScaler("cuda", enabled=amp_enabled, init_scale=amp_init_scale)
    grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("training.grad_accum_steps must be positive")
    effective_batch_size = global_batch_size * grad_accum_steps
    main_print(
        rank,
        f"Global batch: {global_batch_size} = {per_process_batch_size}/GPU x {world_size} GPU(s); "
        f"gradient accumulation: {grad_accum_steps}; effective batch: {effective_batch_size}; "
        f"AMP initial scale: {amp_init_scale:g}",
    )

    configured_interval = int(config["training"].get("log_interval", 50))
    progress_interval = int(configured_interval if log_interval is None else log_interval)
    if progress_interval < 1:
        raise ValueError("log interval must be positive")
    if smoke_test_batches is not None:
        if smoke_test_batches < 1:
            raise ValueError("--smoke-test-batches must be positive")
        progress_interval = min(progress_interval, max(1, smoke_test_batches // 5))
        main_print(rank, f"Smoke test: {smoke_test_batches} training batches; no checkpoints will be written")

    epochs = int(config["training"]["epochs"])
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

            window_start = (batch_idx // grad_accum_steps) * grad_accum_steps
            window_end = min(window_start + grad_accum_steps, batches_this_epoch)
            window_size = window_end - window_start
            should_step = batch_idx + 1 == window_end
            sync_context = model.no_sync() if distributed and not should_step else nullcontext()

            with sync_context:
                with autocast("cuda", enabled=amp_enabled):
                    outputs = model(inputs)
                    raw_loss = criterion(outputs, targets)
                    loss = raw_loss / window_size
                scaler.scale(loss).backward()

            if should_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # A lower scale means GradScaler skipped the optimizer step
                # because at least one gradient was non-finite.
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
                    f"Loss: {mean_loss:.6f} | {seconds_per_batch:.3f}s/batch | "
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
                f"Smoke test complete: loss={average_train_loss:.6f}, "
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
        val_loss, val_rmse, val_mae = validate(model, val_loader, criterion, device)
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
            f"Train Loss: {average_train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val RMSE: {val_rmse:.6f} | Val MAE: {val_mae:.6f} | LR: {learning_rate:.2e}",
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        # Advance before saving so a resumed run starts with the next epoch's LR.
        # If every AMP step overflowed, no parameter update occurred and the
        # schedule must remain at the current epoch.
        if optimizer_steps > 0:
            scheduler.step()
        else:
            main_print(rank, "Warning: all optimizer steps were skipped; LR scheduler was not advanced")

        if is_main_process(rank):
            if improved:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    os.path.join(checkpoint_dir, "best_model.pth"),
                    config,
                )
                print(f"  -> New best model saved (val_loss: {best_val_loss:.6f})", flush=True)
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
                    config,
                )
        if distributed:
            dist.barrier()

    if is_main_process(rank):
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epochs - 1,
            best_val_loss,
            os.path.join(checkpoint_dir, "final_model.pth"),
            config,
        )
        if writer is not None:
            writer.close()
        print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}", flush=True)
    if distributed:
        dist.barrier()


def main():
    args = parse_args()
    distributed = False
    try:
        distributed, rank, world_size, local_rank, device = setup_process(args)
        config = configure_seeded_run(
            load_config(args.config),
            training_seed=args.seed,
            split_seed=args.split_seed,
            isolate_outputs=args.seed is not None,
        )
        if args.resume:
            config["training"]["resume"] = args.resume
        train(
            config,
            resume_path=config["training"].get("resume"),
            subset_frac=args.subset,
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
