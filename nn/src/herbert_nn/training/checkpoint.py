# SPDX-License-Identifier: MIT
"""Checkpoint save/load helpers.

A checkpoint embeds not just model weights but everything needed to
reconstruct the exact model architecture and locate the exact preprocessing
cache it was trained against, so ``herbert_nn.evaluate`` and
``herbert_nn.inspect`` can load a checkpoint standalone (no need to re-supply
Hydra overrides matching the original training run).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from herbert_nn.models.base import DataMeta, build_model

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_cfg: dict[str, Any],
    data_meta: DataMeta,
    cache_path: str,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    optimizer: Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint to ``path``.

    Args:
        path: Destination file path (e.g. ``.../best.pt``).
        model: The model whose ``state_dict`` to save.
        model_cfg: The resolved ``model`` Hydra config group, as a plain dict
            (must be sufficient, together with ``data_meta``, to rebuild the
            model via :func:`herbert_nn.models.base.build_model`).
        data_meta: Dataset-derived sizes the model was built with.
        cache_path: Filesystem path to the preprocessing cache directory used
            for this run, so downstream tools can reload the matching
            normalizer/vocabularies.
        epoch: Epoch number this checkpoint was produced at.
        global_step: Optimizer step count at save time.
        best_val_loss: Best validation composite loss observed so far.
        optimizer: If provided, its ``state_dict`` is included so training
            can be resumed exactly.
        extra: Any additional JSON/tensor-safe metadata to embed (e.g. the
            full resolved training config, for provenance).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_cfg": dict(model_cfg),
        "data_meta": asdict(data_meta),
        "cache_path": cache_path,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    logger.info(
        "Saved checkpoint to %s (epoch=%d, best_val_loss=%.6f)",
        path,
        epoch,
        best_val_loss,
    )


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a raw checkpoint dict from disk (does not construct a model).

    Args:
        path: Checkpoint file path.
        map_location: Passed through to ``torch.load``.

    Returns:
        The raw checkpoint dict as written by :func:`save_checkpoint`.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def build_model_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    """Reconstruct a model (with weights loaded) from a checkpoint dict.

    Args:
        checkpoint: The dict returned by :func:`load_checkpoint`.

    Returns:
        The reconstructed model in ``eval()`` mode.
    """
    data_meta = DataMeta(**checkpoint["data_meta"])
    model = build_model(checkpoint["model_cfg"], data_meta)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
