# SPDX-License-Identifier: MIT
"""Tests for per-head losses and the composite weighted-sum loss."""

from __future__ import annotations

import torch
from torch import nn

from herbert_nn.models.losses import (
    BlockPlacementHeadLoss,
    CompositeLoss,
    DiscreteHeadLoss,
    MouseHeadLoss,
)


def test_mouse_head_loss_matches_manual_huber() -> None:
    pred = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
    loss_module = MouseHeadLoss(huber_delta=1.0)
    actual = loss_module(pred, target)
    expected = nn.HuberLoss(delta=1.0, reduction="mean")(pred, target)
    assert torch.allclose(actual, expected)


def test_discrete_head_loss_matches_manual_bce() -> None:
    pred_logits = torch.tensor([[2.0, -2.0, 0.0, 1.0], [-1.0, 1.0, 0.5, -0.5]])
    target = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    loss_module = DiscreteHeadLoss()
    actual = loss_module(pred_logits, target)
    expected = nn.BCEWithLogitsLoss(reduction="mean")(pred_logits, target)
    assert torch.allclose(actual, expected)


def test_block_placement_loss_ignores_ticks_with_no_place_event() -> None:
    vocab_size = 5
    pred_logits = torch.randn(4, vocab_size)
    target = torch.tensor([0, 1, 2, 3])
    place_mask = torch.tensor([0.0, 0.0, 0.0, 0.0])
    loss_module = BlockPlacementHeadLoss()
    loss = loss_module(pred_logits, target, place_mask)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_block_placement_loss_only_averages_over_active_place_ticks() -> None:
    pred_logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],  # confidently correct at index 0
            [0.0, 0.0, 0.0, 0.0],  # irrelevant (masked out)
            [0.0, 0.0, 0.0, 5.0],  # confidently correct at index 3
        ]
    )
    target = torch.tensor([0, 2, 3])
    place_mask = torch.tensor([1.0, 0.0, 1.0])
    loss_module = BlockPlacementHeadLoss()
    actual = loss_module(pred_logits, target, place_mask)

    manual = nn.CrossEntropyLoss(reduction="none")(pred_logits, target)
    expected = (manual * place_mask).sum() / place_mask.sum()
    assert torch.allclose(actual, expected)
    # Sanity: this should be a small loss since both active predictions are confident+correct.
    assert actual.item() < 0.5


def test_composite_loss_equals_weighted_sum_of_individual_heads() -> None:
    batch_size, vocab_size = 6, 5
    predictions = {
        "mouse": torch.randn(batch_size, 2),
        "discrete": torch.randn(batch_size, 4),
        "block_placement": torch.randn(batch_size, vocab_size),
    }
    targets = {
        "mouse_target": torch.randn(batch_size, 2),
        "discrete_target": (torch.rand(batch_size, 4) > 0.5).float(),
        "place_block_type": torch.randint(0, vocab_size, (batch_size,)),
        "place_mask": (torch.rand(batch_size) > 0.5).float(),
    }
    weights = {"mouse": 2.0, "discrete": 0.5, "block_placement": 3.0}
    composite = CompositeLoss(
        mouse_weight=weights["mouse"],
        discrete_weight=weights["discrete"],
        block_placement_weight=weights["block_placement"],
    )
    breakdown = composite(predictions, targets)

    expected_total = (
        weights["mouse"] * breakdown["mouse"]
        + weights["discrete"] * breakdown["discrete"]
        + weights["block_placement"] * breakdown["block_placement"]
    )
    assert torch.allclose(breakdown["total"], expected_total)


def test_composite_loss_is_differentiable_end_to_end() -> None:
    batch_size, vocab_size = 4, 3
    predictions = {
        "mouse": torch.randn(batch_size, 2, requires_grad=True),
        "discrete": torch.randn(batch_size, 4, requires_grad=True),
        "block_placement": torch.randn(batch_size, vocab_size, requires_grad=True),
    }
    targets = {
        "mouse_target": torch.randn(batch_size, 2),
        "discrete_target": (torch.rand(batch_size, 4) > 0.5).float(),
        "place_block_type": torch.randint(0, vocab_size, (batch_size,)),
        "place_mask": torch.ones(batch_size),
    }
    composite = CompositeLoss()
    breakdown = composite(predictions, targets)
    breakdown["total"].backward()
    assert predictions["mouse"].grad is not None
    assert predictions["discrete"].grad is not None
    assert predictions["block_placement"].grad is not None
