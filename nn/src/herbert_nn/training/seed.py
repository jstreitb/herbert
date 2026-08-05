# SPDX-License-Identifier: MIT
"""Deterministic seeding across Python, NumPy, and PyTorch (CPU + CUDA)."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Seed every RNG the training pipeline touches, for reproducible runs.

    Args:
        seed: The seed value to apply everywhere.
        deterministic_cudnn: If ``True``, set ``torch.backends.cudnn.deterministic
            = True`` and ``torch.backends.cudnn.benchmark = False``. Slightly
            slower but makes GPU convolution/GRU kernels reproducible; disable
            for speed once a config is stable and reproducibility no longer matters.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(
        "Seeded random/numpy/torch with seed=%d (deterministic_cudnn=%s)",
        seed,
        deterministic_cudnn,
    )
