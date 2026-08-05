# SPDX-License-Identifier: MIT
"""Common types and the model factory shared across MLP/GRU policies.

Both :class:`herbert_nn.models.mlp.MLPPolicy` and
:class:`herbert_nn.models.gru.GRUPolicy` implement the same interface:
``forward(batch: dict[str, Tensor]) -> PolicyOutput``, where ``batch`` holds
the tensors produced by :mod:`herbert_nn.data.dataset` (with or without a
window dimension, depending on the model) and ``PolicyOutput`` is a dict
with keys ``"mouse"``, ``"discrete"``, ``"block_placement"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

import torch
from torch import nn


class PolicyOutput(TypedDict):
    """Raw (pre-activation) outputs of the three shared heads.

    Keys:
        mouse: ``[batch, 2]`` -- raw regression output for (d_yaw, d_pitch).
        discrete: ``[batch, 4]`` -- raw logits for (jump, sneak, attack, place).
        block_placement: ``[batch, block_type_vocab_size]`` -- raw logits.
    """

    mouse: torch.Tensor
    discrete: torch.Tensor
    block_placement: torch.Tensor


@dataclass(frozen=True)
class DataMeta:
    """Dataset-derived sizes a model needs to build its input/output layers.

    Sourced from a :class:`herbert_nn.data.cache.CacheManifest` so the model
    architecture always matches whatever cache it will be trained/evaluated
    against (vocab sizes and block-grid shape are dataset properties, not
    hyperparameters).
    """

    block_grid_shape: tuple[int, int, int]
    item_type_vocab_size: int
    kit_type_vocab_size: int
    place_block_type_vocab_size: int

    @property
    def num_block_cells(self) -> int:
        """Total flattened block-grid cell count (``width * height * depth``)."""
        w, h, d = self.block_grid_shape
        return w * h * d


def build_model(model_cfg: Mapping[str, Any], data_meta: DataMeta) -> nn.Module:
    """Instantiate the policy model described by a Hydra ``model`` config group.

    Args:
        model_cfg: The resolved ``conf/model/*.yaml`` config (as a plain
            mapping / ``DictConfig``). Must contain a ``family`` key equal to
            ``"mlp"`` or ``"gru"``, plus that family's hyperparameters.
        data_meta: Dataset-derived sizes (see :class:`DataMeta`).

    Returns:
        An ``nn.Module`` implementing ``forward(batch) -> PolicyOutput``.

    Raises:
        ValueError: If ``model_cfg["family"]`` is not ``"mlp"`` or ``"gru"``.
    """
    # Imported lazily to avoid a circular import (mlp.py / gru.py import
    # DataMeta from this module).
    from herbert_nn.models.gru import GRUPolicy
    from herbert_nn.models.mlp import MLPPolicy

    family = model_cfg["family"]
    if family == "mlp":
        return MLPPolicy(
            data_meta=data_meta,
            hidden_dims=list(model_cfg["hidden_dims"]),
            dropout=float(model_cfg["dropout"]),
            block_cell_embed_dim=int(model_cfg["block_cell_embed_dim"]),
            item_type_embed_dim=int(model_cfg["item_type_embed_dim"]),
            kit_type_embed_dim=int(model_cfg["kit_type_embed_dim"]),
            held_item_embed_dim=int(model_cfg["held_item_embed_dim"]),
            hotbar_slot_embed_dim=int(model_cfg["hotbar_slot_embed_dim"]),
        )
    if family == "gru":
        return GRUPolicy(
            data_meta=data_meta,
            hidden_size=int(model_cfg["hidden_size"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            trunk_hidden_dims=list(model_cfg["trunk_hidden_dims"]),
            block_cell_embed_dim=int(model_cfg["block_cell_embed_dim"]),
            item_type_embed_dim=int(model_cfg["item_type_embed_dim"]),
            kit_type_embed_dim=int(model_cfg["kit_type_embed_dim"]),
            held_item_embed_dim=int(model_cfg["held_item_embed_dim"]),
            hotbar_slot_embed_dim=int(model_cfg["hotbar_slot_embed_dim"]),
        )
    raise ValueError(f"Unknown model family {family!r}, expected 'mlp' or 'gru'.")
