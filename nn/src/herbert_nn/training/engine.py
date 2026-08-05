# SPDX-License-Identifier: MIT
"""Reusable train/eval epoch loop: mixed precision, gradient accumulation, clipping.

Used by both the full ``herbert_nn.train`` Hydra-driven CLI and the fast
``herbert_nn.smoketest`` path, so both exercise the exact same forward/
backward code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from herbert_nn.models.losses import CompositeLoss

logger = logging.getLogger(__name__)

_TARGET_KEYS = ("mouse_target", "discrete_target", "place_block_type", "place_mask")


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move every tensor in a batch dict to ``device`` (non-blocking where possible)."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@dataclass
class EpochStats:
    """Running (and finally averaged) per-head loss totals for one epoch."""

    total: float = 0.0
    mouse: float = 0.0
    discrete: float = 0.0
    block_placement: float = 0.0
    num_batches: int = 0
    num_samples: int = 0

    def update(self, breakdown: dict[str, torch.Tensor], batch_size: int) -> None:
        """Accumulate one batch's loss breakdown into the running totals.

        Args:
            breakdown: A :class:`herbert_nn.models.losses.LossBreakdown` for one batch.
            batch_size: Number of samples in that batch, used to weight the running average.
        """
        self.total += float(breakdown["total"].detach().item()) * batch_size
        self.mouse += float(breakdown["mouse"].detach().item()) * batch_size
        self.discrete += float(breakdown["discrete"].detach().item()) * batch_size
        self.block_placement += (
            float(breakdown["block_placement"].detach().item()) * batch_size
        )
        self.num_batches += 1
        self.num_samples += batch_size

    def averaged(self) -> dict[str, float]:
        """Return the running totals divided by the number of samples seen so far."""
        n = max(1, self.num_samples)
        return {
            "total": self.total / n,
            "mouse": self.mouse / n,
            "discrete": self.discrete / n,
            "block_placement": self.block_placement / n,
        }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CompositeLoss,
    device: torch.device,
    optimizer: Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
    grad_accum_steps: int = 1,
    grad_clip_norm: float | None = None,
    is_train: bool = True,
) -> dict[str, float]:
    """Run one full pass over ``loader``, training or evaluating.

    Args:
        model: The policy model.
        loader: A ``DataLoader`` yielding batches from :mod:`herbert_nn.data.dataset`.
        loss_fn: The composite loss.
        device: Device to run on.
        optimizer: Required if ``is_train`` is ``True``.
        scheduler: Optional LR scheduler, stepped once per optimizer step
            (i.e. once every ``grad_accum_steps`` batches).
        scaler: Optional ``GradScaler`` for AMP; required (and only used) if
            ``amp`` is ``True`` and ``device.type == "cuda"``.
        amp: Whether to run the forward pass under ``torch.autocast``.
            Automatically disabled on non-CUDA devices regardless of this
            flag (autocast on CPU brings no benefit here and GradScaler is
            CUDA-only), with a one-time warning.
        grad_accum_steps: Number of batches to accumulate gradients over
            before each optimizer step.
        grad_clip_norm: If set, clip gradient global norm to this value
            before each optimizer step.
        is_train: If ``True``, run in training mode with backward passes; if
            ``False``, run in ``eval()`` mode under ``torch.no_grad()``.

    Returns:
        Dict of epoch-averaged losses: ``{"total", "mouse", "discrete", "block_placement"}``.
    """
    if is_train and optimizer is None:
        raise ValueError("optimizer is required when is_train=True")
    # Narrow Optional[Optimizer] -> Optimizer once, for use in the `is_train`
    # branches below (mypy can't otherwise carry the None-check across the
    # loop/with-block boundaries).
    train_optimizer = cast(Optimizer, optimizer)

    effective_amp = amp and device.type == "cuda"
    if amp and not effective_amp:
        logger.warning(
            "AMP requested but device is %s (not cuda); running in full precision.",
            device.type,
        )

    model.train(mode=is_train)
    stats = EpochStats()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        if is_train:
            train_optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            batch_size = batch["mouse_target"].shape[0]

            with torch.autocast(device_type=device.type, enabled=effective_amp):
                predictions = model(batch)
                breakdown = loss_fn(predictions, {k: batch[k] for k in _TARGET_KEYS})
                loss = breakdown["total"]

            if is_train:
                loss_to_backward = loss / grad_accum_steps
                if scaler is not None and effective_amp:
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

                is_last_micro_batch = (step + 1) % grad_accum_steps == 0 or (
                    step + 1 == len(loader)
                )
                if is_last_micro_batch:
                    if scaler is not None and effective_amp:
                        if grad_clip_norm is not None:
                            scaler.unscale_(train_optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip_norm
                            )
                        scaler.step(train_optimizer)
                        scaler.update()
                    else:
                        if grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip_norm
                            )
                        train_optimizer.step()
                    train_optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            stats.update(breakdown, batch_size)

    return stats.averaged()
