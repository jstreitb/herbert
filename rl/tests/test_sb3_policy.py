# SPDX-License-Identifier: MIT
"""Tests for `herbert_rl.policy.sb3_policy` -- splicing pretrained `/nn` heads into PPO's action_net."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from factories import make_cache_stats
from herbert_rl.env.action_wrapper import FLAT_ACTION_DIM
from herbert_rl.env.spaces import build_observation_space
from herbert_rl.policy.backbone import RLDataMeta, build_backbone
from herbert_rl.policy.checkpoint_adapter import (
    DISCRETE_ACTION_NET_ROWS,
    MOUSE_ACTION_NET_ROWS,
    MOVEMENT_ACTION_NET_ROWS,
    load_nn_checkpoint,
)
from herbert_rl.policy.sb3_policy import HerbertRLPolicy

_MLP_MODEL_CFG = {
    "family": "mlp",
    "hidden_dims": [16, 8],
    "dropout": 0.0,
    "block_cell_embed_dim": 4,
    "item_type_embed_dim": 4,
    "kit_type_embed_dim": 4,
    "held_item_embed_dim": 4,
    "hotbar_slot_embed_dim": 4,
}


def _write_synthetic_checkpoint(path, data_meta: RLDataMeta) -> dict[str, torch.Tensor]:
    """Write a `/nn`-shaped checkpoint with all four pretrained heads populated."""
    backbone = build_backbone(_MLP_MODEL_CFG, data_meta)
    state_dict = dict(backbone.state_dict())
    trunk_dim = backbone.output_dim
    for name, out_dim in (
        ("mouse_head", 2),
        ("discrete_head", 4),
        ("block_placement_head", 6),
        ("movement_head", 2),
    ):
        state_dict[f"{name}.linear.weight"] = torch.randn(out_dim, trunk_dim)
        state_dict[f"{name}.linear.bias"] = torch.randn(out_dim)

    payload = {
        "model_state_dict": state_dict,
        "model_cfg": _MLP_MODEL_CFG,
        "data_meta": {
            "block_grid_shape": list(data_meta.block_grid_shape),
            "item_type_vocab_size": data_meta.item_type_vocab_size,
            "kit_type_vocab_size": data_meta.kit_type_vocab_size,
            "place_block_type_vocab_size": 6,
        },
        "cache_path": "<synthetic-test-cache>",
        "epoch": 1,
        "global_step": 1,
        "best_val_loss": 0.0,
    }
    torch.save(payload, path)
    return state_dict


def _build_flat_action_space() -> spaces.Box:
    # Mirrors `env/action_wrapper.py::FlattenHerbertActionWrapper`'s Box(8,) exactly.
    low = np.array([-1, -1, -1, -1, -1, -1, -180.0, -180.0], dtype=np.float32)
    high = np.array([1, 1, 1, 1, 1, 1, 180.0, 180.0], dtype=np.float32)
    return spaces.Box(low=low, high=high, shape=(FLAT_ACTION_DIM,), dtype=np.float32)


def test_pretrained_heads_spliced_into_action_net_including_movement(tmp_path):
    data_meta = RLDataMeta(
        block_grid_shape=(7, 3, 7), item_type_vocab_size=10, kit_type_vocab_size=5
    )
    checkpoint_path = tmp_path / "synthetic.pt"
    state_dict = _write_synthetic_checkpoint(checkpoint_path, data_meta)
    loaded = load_nn_checkpoint(checkpoint_path)

    cache_stats = make_cache_stats(
        block_grid_shape=(7, 3, 7), item_type_vocab_size=10, kit_type_vocab_size=5
    )
    observation_space = build_observation_space(cache_stats, window_length=1)
    action_space = _build_flat_action_space()

    policy = HerbertRLPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _progress_remaining: 3e-4,
        loaded_backbone=loaded,
    )

    assert torch.allclose(
        policy.action_net.weight[MOVEMENT_ACTION_NET_ROWS],
        state_dict["movement_head.linear.weight"],
    )
    assert torch.allclose(
        policy.action_net.bias[MOVEMENT_ACTION_NET_ROWS],
        state_dict["movement_head.linear.bias"],
    )
    assert torch.allclose(
        policy.action_net.weight[DISCRETE_ACTION_NET_ROWS],
        state_dict["discrete_head.linear.weight"],
    )
    assert torch.allclose(
        policy.action_net.weight[MOUSE_ACTION_NET_ROWS],
        state_dict["mouse_head.linear.weight"],
    )
