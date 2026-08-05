# SPDX-License-Identifier: MIT
"""Cosine learning-rate schedule with linear warmup."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_lr_scheduler(
    optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0
) -> LambdaLR:
    """Build a ``LambdaLR`` implementing linear warmup then cosine decay.

    The multiplier applied to each param group's base ``lr`` is:

    * ``step / warmup_steps`` during the first ``warmup_steps`` steps (linear
      ramp from 0 to 1).
    * A cosine decay from 1.0 down to ``min_lr_ratio`` over the remaining
      ``total_steps - warmup_steps`` steps.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of optimizer steps to linearly warm up over.
            Clamped to at least 1 internally to avoid a division by zero.
        total_steps: Total number of optimizer steps the schedule spans
            (typically ``epochs * steps_per_epoch // grad_accum_steps``).
        min_lr_ratio: Floor for the cosine decay, as a fraction of the base
            LR (e.g. ``0.1`` never decays below 10% of the base LR).

    Returns:
        A :class:`torch.optim.lr_scheduler.LambdaLR` to call ``.step()`` on
        once per optimizer step (not once per epoch).
    """
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
