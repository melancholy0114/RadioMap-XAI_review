"""Validation for single-process and DistributedDataParallel training."""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import autocast


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Run validation and return globally reduced loss, RMSE, and MAE."""
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    amp_enabled = device.type == "cuda"

    for batch in val_loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with autocast("cuda", enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        totals[0] += loss.detach().to(torch.float64)
        totals[1] += torch.sqrt(
            nn.functional.mse_loss(outputs, targets)
        ).to(torch.float64)
        totals[2] += nn.functional.l1_loss(outputs, targets).to(torch.float64)
        totals[3] += 1

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    num_batches = max(totals[3].item(), 1.0)
    avg_loss = totals[0].item() / num_batches
    avg_rmse = totals[1].item() / num_batches
    avg_mae = totals[2].item() / num_batches

    return avg_loss, avg_rmse, avg_mae
