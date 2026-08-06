# SPDX-License-Identifier: MIT
"""Tests for `herbert_rl.policy.checkpoint_adapter` -- loading a `/nn`-shaped checkpoint.

Without importing `herbert_nn`: a synthetic checkpoint dict is built here in exactly the
shape `herbert_nn.training.checkpoint.save_checkpoint` would produce, per that module's
documented format (`model_state_dict`/`model_cfg`/`data_meta`).
"""

from __future__ import annotations

import torch

from herbert_rl.policy.backbone import RLDataMeta, build_backbone
from herbert_rl.policy.checkpoint_adapter import load_nn_checkpoint

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


def _write_synthetic_nn_checkpoint(path, model_cfg, data_meta: RLDataMeta):
    """Build a checkpoint matching /nn's on-disk format.

    Includes pretrained mouse_head/discrete_head/block_placement_head weights (which
    build_backbone() intentionally does not produce, since the RL backbone omits those
    heads -- see backbone.py).
    """
    backbone = build_backbone(model_cfg, data_meta)
    state_dict = dict(backbone.state_dict())
    trunk_dim = backbone.output_dim
    state_dict["mouse_head.linear.weight"] = torch.randn(2, trunk_dim)
    state_dict["mouse_head.linear.bias"] = torch.randn(2)
    state_dict["discrete_head.linear.weight"] = torch.randn(4, trunk_dim)
    state_dict["discrete_head.linear.bias"] = torch.randn(4)
    state_dict["block_placement_head.linear.weight"] = torch.randn(6, trunk_dim)
    state_dict["block_placement_head.linear.bias"] = torch.randn(6)
    state_dict["movement_head.linear.weight"] = torch.randn(2, trunk_dim)
    state_dict["movement_head.linear.bias"] = torch.randn(2)

    payload = {
        "model_state_dict": state_dict,
        "model_cfg": model_cfg,
        "data_meta": {
            "block_grid_shape": list(data_meta.block_grid_shape),
            "item_type_vocab_size": data_meta.item_type_vocab_size,
            "kit_type_vocab_size": data_meta.kit_type_vocab_size,
            "place_block_type_vocab_size": 6,  # present in real /nn checkpoints, ignored by RLDataMeta
        },
        "cache_path": "<synthetic-test-cache>",
        "epoch": 1,
        "global_step": 1,
        "best_val_loss": 0.0,
    }
    torch.save(payload, path)
    return backbone, state_dict


def test_load_nn_checkpoint_reconstructs_backbone_with_matching_weights(tmp_path):
    data_meta = RLDataMeta(
        block_grid_shape=(7, 3, 7), item_type_vocab_size=10, kit_type_vocab_size=5
    )
    checkpoint_path = tmp_path / "synthetic.pt"
    _reference_backbone, state_dict = _write_synthetic_nn_checkpoint(
        checkpoint_path, _MLP_MODEL_CFG, data_meta
    )

    loaded = load_nn_checkpoint(checkpoint_path)

    assert loaded.model_family == "mlp"
    assert loaded.checkpoint_path == str(checkpoint_path)
    # Encoder weights should have loaded exactly (not left at a fresh random init).
    loaded_encoder_weight = loaded.backbone.state_dict()[
        "encoder.block_cell_embedding.weight"
    ]
    assert torch.equal(
        loaded_encoder_weight, state_dict["encoder.block_cell_embedding.weight"]
    )


def test_load_nn_checkpoint_extracts_pretrained_head_weights(tmp_path):
    data_meta = RLDataMeta(
        block_grid_shape=(7, 3, 7), item_type_vocab_size=10, kit_type_vocab_size=5
    )
    checkpoint_path = tmp_path / "synthetic.pt"
    _reference_backbone, state_dict = _write_synthetic_nn_checkpoint(
        checkpoint_path, _MLP_MODEL_CFG, data_meta
    )

    loaded = load_nn_checkpoint(checkpoint_path)

    assert torch.equal(
        loaded.pretrained_heads.mouse_weight, state_dict["mouse_head.linear.weight"]
    )
    assert torch.equal(
        loaded.pretrained_heads.mouse_bias, state_dict["mouse_head.linear.bias"]
    )
    assert torch.equal(
        loaded.pretrained_heads.discrete_weight,
        state_dict["discrete_head.linear.weight"],
    )
    assert torch.equal(
        loaded.pretrained_heads.discrete_bias, state_dict["discrete_head.linear.bias"]
    )
    assert torch.equal(
        loaded.pretrained_heads.movement_weight,
        state_dict["movement_head.linear.weight"],
    )
    assert torch.equal(
        loaded.pretrained_heads.movement_bias, state_dict["movement_head.linear.bias"]
    )


def test_build_fresh_backbone_has_no_pretrained_heads():
    from herbert_rl.policy.checkpoint_adapter import build_fresh_backbone

    data_meta = RLDataMeta(
        block_grid_shape=(7, 3, 7), item_type_vocab_size=10, kit_type_vocab_size=5
    )
    loaded = build_fresh_backbone(_MLP_MODEL_CFG, data_meta)

    assert loaded.checkpoint_path is None
    assert loaded.pretrained_heads.mouse_weight is None
    assert loaded.pretrained_heads.discrete_weight is None
    assert loaded.pretrained_heads.movement_weight is None
