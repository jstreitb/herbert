# SPDX-License-Identifier: MIT
"""Model forward-pass shape tests (CPU only, dummy batches)."""

from __future__ import annotations

import torch

from herbert_nn.constants import CONTINUOUS_FEATURE_DIM, NUM_DISCRETE_ACTIONS
from herbert_nn.models.base import DataMeta, build_model

_DATA_META = DataMeta(
    block_grid_shape=(2, 2, 2),
    item_type_vocab_size=10,
    kit_type_vocab_size=6,
    place_block_type_vocab_size=8,
)

_MLP_CFG = {
    "family": "mlp",
    "hidden_dims": [16, 8],
    "dropout": 0.0,
    "block_cell_embed_dim": 3,
    "item_type_embed_dim": 4,
    "kit_type_embed_dim": 2,
    "held_item_embed_dim": 2,
    "hotbar_slot_embed_dim": 2,
}
_GRU_CFG = {
    "family": "gru",
    "hidden_size": 16,
    "num_layers": 2,
    "dropout": 0.1,
    "trunk_hidden_dims": [8],
    "block_cell_embed_dim": 3,
    "item_type_embed_dim": 4,
    "kit_type_embed_dim": 2,
    "held_item_embed_dim": 2,
    "hotbar_slot_embed_dim": 2,
}


def _make_single_tick_batch(batch_size: int) -> dict[str, torch.Tensor]:
    num_cells = _DATA_META.num_block_cells
    return {
        "continuous": torch.randn(batch_size, CONTINUOUS_FEATURE_DIM),
        "block_grid_cells": torch.randint(0, 5, (batch_size, num_cells)),
        "hotbar_slot_index": torch.randint(0, 9, (batch_size,)),
        "hotbar_item_type": torch.randint(
            0, _DATA_META.item_type_vocab_size, (batch_size,)
        ),
        "opponent_held_item_category": torch.randint(0, 5, (batch_size,)),
        "match_kit_type": torch.randint(
            0, _DATA_META.kit_type_vocab_size, (batch_size,)
        ),
    }


def _make_window_batch(batch_size: int, window_length: int) -> dict[str, torch.Tensor]:
    num_cells = _DATA_META.num_block_cells
    return {
        "continuous": torch.randn(batch_size, window_length, CONTINUOUS_FEATURE_DIM),
        "block_grid_cells": torch.randint(0, 5, (batch_size, window_length, num_cells)),
        "hotbar_slot_index": torch.randint(0, 9, (batch_size, window_length)),
        "hotbar_item_type": torch.randint(
            0, _DATA_META.item_type_vocab_size, (batch_size, window_length)
        ),
        "opponent_held_item_category": torch.randint(0, 5, (batch_size, window_length)),
        "match_kit_type": torch.randint(
            0, _DATA_META.kit_type_vocab_size, (batch_size, window_length)
        ),
    }


def test_mlp_policy_forward_shapes() -> None:
    model = build_model(_MLP_CFG, _DATA_META)
    batch = _make_single_tick_batch(batch_size=5)
    output = model(batch)
    assert output["mouse"].shape == (5, 2)
    assert output["discrete"].shape == (5, NUM_DISCRETE_ACTIONS)
    assert output["block_placement"].shape == (
        5,
        _DATA_META.place_block_type_vocab_size,
    )


def test_gru_policy_forward_shapes() -> None:
    model = build_model(_GRU_CFG, _DATA_META)
    batch = _make_window_batch(batch_size=5, window_length=7)
    output = model(batch)
    assert output["mouse"].shape == (5, 2)
    assert output["discrete"].shape == (5, NUM_DISCRETE_ACTIONS)
    assert output["block_placement"].shape == (
        5,
        _DATA_META.place_block_type_vocab_size,
    )


def test_mlp_policy_backward_pass_produces_gradients() -> None:
    model = build_model(_MLP_CFG, _DATA_META)
    batch = _make_single_tick_batch(batch_size=4)
    output = model(batch)
    loss = (
        output["mouse"].sum()
        + output["discrete"].sum()
        + output["block_placement"].sum()
    )
    loss.backward()
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    assert len(grad_norms) > 0
    assert any(g > 0 for g in grad_norms)


def test_gru_policy_backward_pass_produces_gradients() -> None:
    model = build_model(_GRU_CFG, _DATA_META)
    batch = _make_window_batch(batch_size=4, window_length=6)
    output = model(batch)
    loss = (
        output["mouse"].sum()
        + output["discrete"].sum()
        + output["block_placement"].sum()
    )
    loss.backward()
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    assert len(grad_norms) > 0
    assert any(g > 0 for g in grad_norms)
