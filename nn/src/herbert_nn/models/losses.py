# SPDX-License-Identifier: MIT
"""Per-head losses and the configurable-weight composite loss.

* :class:`MouseHeadLoss` -- Huber/SmoothL1 regression loss on ``(d_yaw, d_pitch)``.
* :class:`DiscreteHeadLoss` -- BCE-with-logits multi-label loss on
  ``(jump, sneak, attack, place)``.
* :class:`BlockPlacementHeadLoss` -- cross-entropy over the block-type
  vocabulary, masked to only contribute on ticks where
  ``input.place_occurred`` was true (via ``place_mask``); ticks with no
  place event contribute 0 to this head's loss.
* :class:`MovementHeadLoss` -- Huber/SmoothL1 regression loss on
  ``(forward, strafe)``, unmasked (recorded on every tick).
* :class:`CompositeLoss` -- the weighted sum of the four, with weights
  supplied via the Hydra training config.
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn


class LossBreakdown(TypedDict):
    """Individual and combined loss values (all detachable scalars for logging)."""

    total: torch.Tensor
    mouse: torch.Tensor
    discrete: torch.Tensor
    block_placement: torch.Tensor
    movement: torch.Tensor


class MouseHeadLoss(nn.Module):
    """Huber loss for the mouse-movement regression head."""

    def __init__(self, huber_delta: float = 1.0) -> None:
        """Initialize the loss.

        Args:
            huber_delta: Threshold at which the Huber loss transitions from quadratic to
                linear (``nn.HuberLoss``'s ``delta`` parameter).
        """
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=huber_delta, reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the mean Huber loss over the batch.

        Args:
            pred: Predicted ``(d_yaw, d_pitch)``, shape ``[batch, 2]``.
            target: Ground-truth ``(d_yaw, d_pitch)``, shape ``[batch, 2]``.

        Returns:
            A scalar loss tensor.
        """
        return self.loss_fn(pred, target)


class MovementHeadLoss(nn.Module):
    """Huber loss for the movement-axis regression head.

    Reuses the same ``huber_delta`` hyperparameter as :class:`MouseHeadLoss` rather than
    introducing a second one -- both heads regress against small bounded targets, so a
    shared delta is a reasonable default.
    """

    def __init__(self, huber_delta: float = 1.0) -> None:
        """Initialize the loss.

        Args:
            huber_delta: Threshold at which the Huber loss transitions from quadratic to
                linear (``nn.HuberLoss``'s ``delta`` parameter).
        """
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=huber_delta, reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the mean Huber loss over the batch.

        Args:
            pred: Predicted ``(forward, strafe)``, shape ``[batch, 2]``.
            target: Ground-truth ``(forward, strafe)``, shape ``[batch, 2]``.

        Returns:
            A scalar loss tensor.
        """
        return self.loss_fn(pred, target)


class DiscreteHeadLoss(nn.Module):
    """BCE-with-logits loss for the binary multi-label discrete-action head."""

    def __init__(self) -> None:
        """Initialize the loss with mean reduction over the batch."""
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the mean BCE-with-logits loss over the batch.

        Args:
            pred_logits: Raw (pre-sigmoid) predictions, shape ``[batch, 4]``.
            target: Ground-truth binary labels, shape ``[batch, 4]``.

        Returns:
            A scalar loss tensor.
        """
        return self.loss_fn(pred_logits, target)


class BlockPlacementHeadLoss(nn.Module):
    """Cross-entropy loss over the block-type vocabulary, masked to place-active ticks."""

    def __init__(self) -> None:
        """Initialize the loss.

        Uses per-sample (unreduced) cross-entropy, reduced manually in :meth:`forward`
        after masking to active-place ticks.
        """
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self, pred_logits: torch.Tensor, target: torch.Tensor, place_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the mean cross-entropy over ticks where ``place_mask == 1``.

        Args:
            pred_logits: ``[batch, vocab_size]`` raw logits.
            target: ``[batch]`` int64 class indices (meaningless where
                ``place_mask`` is 0, since they're never selected).
            place_mask: ``[batch]`` float tensor, 1.0 on ticks with an active
                block-placement event, 0.0 otherwise.

        Returns:
            A scalar tensor: the mean per-tick loss over active-place ticks
            only, or exactly ``0.0`` (still connected to the graph) if the
            batch contains no active-place ticks.
        """
        per_sample = self.loss_fn(pred_logits, target)  # [batch]
        masked = per_sample * place_mask
        denom = place_mask.sum()
        if denom.item() == 0:
            # No place events in this batch: avoid a NaN from 0/0. Multiplying
            # by the (all-zero) mask keeps this connected to the graph as a
            # true zero rather than detaching it.
            return masked.sum()
        return masked.sum() / denom


class CompositeLoss(nn.Module):
    """Weighted sum of the three per-head losses.

    Weights are supplied from the Hydra training config
    (``training.loss_weights.{mouse,discrete,block_placement}``) so users can
    rebalance the objective (e.g. up-weight block placement, which is rare)
    without touching code.
    """

    def __init__(
        self,
        mouse_weight: float = 1.0,
        discrete_weight: float = 1.0,
        block_placement_weight: float = 1.0,
        movement_weight: float = 1.0,
        huber_delta: float = 1.0,
    ) -> None:
        """Initialize the composite loss.

        Args:
            mouse_weight: Weight applied to the mouse-movement head's loss in the total sum.
            discrete_weight: Weight applied to the discrete-action head's loss.
            block_placement_weight: Weight applied to the block-placement head's loss.
            movement_weight: Weight applied to the movement head's loss.
            huber_delta: Passed through to :class:`MouseHeadLoss`'s and
                :class:`MovementHeadLoss`'s ``huber_delta``.
        """
        super().__init__()
        self.mouse_weight = mouse_weight
        self.discrete_weight = discrete_weight
        self.block_placement_weight = block_placement_weight
        self.movement_weight = movement_weight
        self.mouse_loss = MouseHeadLoss(huber_delta=huber_delta)
        self.discrete_loss = DiscreteHeadLoss()
        self.block_placement_loss = BlockPlacementHeadLoss()
        self.movement_loss = MovementHeadLoss(huber_delta=huber_delta)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> LossBreakdown:
        """Compute all per-head losses and their weighted sum.

        Args:
            predictions: Model output dict with keys ``mouse``, ``discrete``,
                ``block_placement``, ``movement`` (see
                :class:`herbert_nn.models.base.PolicyOutput`).
            targets: Batch dict with keys ``mouse_target``, ``discrete_target``,
                ``place_block_type``, ``place_mask``, ``movement_target`` (see
                :mod:`herbert_nn.data.dataset`).

        Returns:
            A :class:`LossBreakdown` with the total (weighted) loss and each
            individual head's (unweighted) loss value, for logging.
        """
        mouse = self.mouse_loss(predictions["mouse"], targets["mouse_target"])
        discrete = self.discrete_loss(
            predictions["discrete"], targets["discrete_target"]
        )
        block_placement = self.block_placement_loss(
            predictions["block_placement"],
            targets["place_block_type"],
            targets["place_mask"],
        )
        movement = self.movement_loss(
            predictions["movement"], targets["movement_target"]
        )
        total = (
            self.mouse_weight * mouse
            + self.discrete_weight * discrete
            + self.block_placement_weight * block_placement
            + self.movement_weight * movement
        )
        return LossBreakdown(
            total=total,
            mouse=mouse,
            discrete=discrete,
            block_placement=block_placement,
            movement=movement,
        )
