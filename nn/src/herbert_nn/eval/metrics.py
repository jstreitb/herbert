# SPDX-License-Identifier: MIT
"""Per-head test-set metrics: MAE (mouse/movement), accuracy/F1/AUC (discrete), top-1/top-3 (block placement)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from herbert_nn.constants import DISCRETE_ACTION_NAMES, MOVEMENT_AXIS_NAMES
from herbert_nn.training.engine import move_batch_to_device

logger = logging.getLogger(__name__)


@dataclass
class CollectedPredictions:
    """Raw model outputs and targets gathered over an entire dataloader pass."""

    mouse_pred: np.ndarray
    mouse_target: np.ndarray
    discrete_prob: np.ndarray
    discrete_target: np.ndarray
    block_logits: np.ndarray
    block_target: np.ndarray
    place_mask: np.ndarray
    movement_pred: np.ndarray
    movement_target: np.ndarray


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> CollectedPredictions:
    """Run the model over every batch in ``loader`` and collect predictions/targets to CPU numpy.

    Args:
        model: A trained policy model (``MLPPolicy`` or ``GRUPolicy``).
        loader: DataLoader over the split to evaluate.
        device: Device to run inference on.

    Returns:
        A :class:`CollectedPredictions` with one row per dataset sample.
    """
    model.eval()
    model.to(device)
    mouse_preds, mouse_targets = [], []
    discrete_probs, discrete_targets = [], []
    block_logits_list, block_targets = [], []
    place_masks = []
    movement_preds, movement_targets = [], []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        output = model(batch)
        mouse_preds.append(output["mouse"].cpu().numpy())
        mouse_targets.append(batch["mouse_target"].cpu().numpy())
        discrete_probs.append(torch.sigmoid(output["discrete"]).cpu().numpy())
        discrete_targets.append(batch["discrete_target"].cpu().numpy())
        block_logits_list.append(output["block_placement"].cpu().numpy())
        block_targets.append(batch["place_block_type"].cpu().numpy())
        place_masks.append(batch["place_mask"].cpu().numpy())
        movement_preds.append(output["movement"].cpu().numpy())
        movement_targets.append(batch["movement_target"].cpu().numpy())

    return CollectedPredictions(
        mouse_pred=np.concatenate(mouse_preds, axis=0),
        mouse_target=np.concatenate(mouse_targets, axis=0),
        discrete_prob=np.concatenate(discrete_probs, axis=0),
        discrete_target=np.concatenate(discrete_targets, axis=0),
        block_logits=np.concatenate(block_logits_list, axis=0),
        block_target=np.concatenate(block_targets, axis=0),
        place_mask=np.concatenate(place_masks, axis=0),
        movement_pred=np.concatenate(movement_preds, axis=0),
        movement_target=np.concatenate(movement_targets, axis=0),
    )


def compute_metrics(collected: CollectedPredictions) -> dict[str, Any]:
    """Compute the full per-head metric report from collected predictions.

    Args:
        collected: Output of :func:`collect_predictions`.

    Returns:
        Nested dict: ``{"mouse": {...}, "discrete": {action: {...}}, "block_placement": {...},
        "movement": {...}}``.
    """
    mouse_mae_per_dim = mean_absolute_error(
        collected.mouse_target, collected.mouse_pred, multioutput="raw_values"
    )
    mouse_metrics = {
        "d_yaw_mae": float(mouse_mae_per_dim[0]),
        "d_pitch_mae": float(mouse_mae_per_dim[1]),
        "overall_mae": float(mouse_mae_per_dim.mean()),
    }

    discrete_metrics: dict[str, dict[str, float | None]] = {}
    for i, name in enumerate(DISCRETE_ACTION_NAMES):
        target = collected.discrete_target[:, i]
        prob = collected.discrete_prob[:, i]
        pred_label = (prob >= 0.5).astype(np.int64)
        accuracy = float(accuracy_score(target, pred_label))
        f1 = float(f1_score(target, pred_label, zero_division=0))
        auc: float | None
        try:
            if len(np.unique(target)) < 2:
                raise ValueError("only one class present")
            auc = float(roc_auc_score(target, prob))
        except ValueError as exc:
            logger.warning(
                "Could not compute ROC-AUC for discrete action %r: %s", name, exc
            )
            auc = None
        discrete_metrics[name] = {"accuracy": accuracy, "f1": f1, "roc_auc": auc}

    place_indices = np.nonzero(collected.place_mask > 0.5)[0]
    if place_indices.size == 0:
        logger.warning(
            "No active place ticks in this split; block_placement metrics are null."
        )
        block_metrics: dict[str, float | None] = {
            "top1_accuracy": None,
            "top3_accuracy": None,
            "num_place_ticks": 0,
        }
    else:
        logits = collected.block_logits[place_indices]
        targets = collected.block_target[place_indices]
        top1_pred = logits.argmax(axis=1)
        top1_acc = float((top1_pred == targets).mean())
        k = min(3, logits.shape[1])
        top_k_pred = np.argsort(-logits, axis=1)[:, :k]
        top3_acc = float((top_k_pred == targets[:, None]).any(axis=1).mean())
        block_metrics = {
            "top1_accuracy": top1_acc,
            "top3_accuracy": top3_acc,
            "num_place_ticks": int(place_indices.size),
        }

    movement_mae_per_axis = mean_absolute_error(
        collected.movement_target, collected.movement_pred, multioutput="raw_values"
    )
    # Bucket raw regression outputs to {-1, 0, 1} the same way
    # `rl/src/herbert_rl/env/action_wrapper.py`'s `_bucket_ternary` does, so this accuracy is
    # interpretable as "would this be the right in-game movement action," not just raw MAE.
    pred_bucketed = _bucket_ternary(collected.movement_pred)
    target_bucketed = _bucket_ternary(collected.movement_target)
    movement_metrics: dict[str, float] = {
        "overall_mae": float(movement_mae_per_axis.mean())
    }
    for i, axis in enumerate(MOVEMENT_AXIS_NAMES):
        movement_metrics[f"{axis}_mae"] = float(movement_mae_per_axis[i])
        movement_metrics[f"{axis}_bucketed_accuracy"] = float(
            accuracy_score(target_bucketed[:, i], pred_bucketed[:, i])
        )

    return {
        "mouse": mouse_metrics,
        "discrete": discrete_metrics,
        "block_placement": block_metrics,
        "movement": movement_metrics,
    }


def _bucket_ternary(values: np.ndarray) -> np.ndarray:
    """Threshold raw regression outputs to {-1, 0, 1}, matching `env/action_wrapper.py`."""
    bucketed = np.zeros_like(values, dtype=np.int64)
    bucketed[values > 0.5] = 1
    bucketed[values < -0.5] = -1
    return bucketed
