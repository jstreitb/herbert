"""Shared per-tick feature encoder used by both :class:`MLPPolicy` and :class:`GRUPolicy`.

Turns one tick's raw feature tensors (the continuous vector plus several
categorical index tensors -- see :mod:`herbert_nn.data.dataset`) into a
single dense embedding vector. Operates generically over an arbitrary number
of leading dimensions, so the exact same module encodes single ticks
(``[batch, ...]``, used by the MLP) and ticks within a window
(``[batch, window, ...]``, used by the GRU) without any special-casing.
"""

from __future__ import annotations

import torch
from torch import nn

from herbert_nn.constants import (
    CONTINUOUS_FEATURE_DIM,
    NUM_HELD_ITEM_CATEGORIES,
    NUM_HOTBAR_SLOTS,
)


class FeatureEncoder(nn.Module):
    """Embeds categorical tick fields and concatenates them with continuous features."""

    def __init__(
        self,
        num_block_cells: int,
        block_cell_embed_dim: int,
        item_type_vocab_size: int,
        item_type_embed_dim: int,
        kit_type_vocab_size: int,
        kit_type_embed_dim: int,
        held_item_embed_dim: int = 8,
        hotbar_slot_embed_dim: int = 8,
        block_cell_num_types: int = 5,
    ) -> None:
        """Configure embedding tables for every categorical tick field.

        Args:
            num_block_cells: ``width * height * depth`` of the block grid
                (i.e. the number of cells per tick), from the cache manifest's
                ``block_grid_shape``.
            block_cell_embed_dim: Embedding dim per block-grid cell.
            item_type_vocab_size: Size of the fitted hotbar item-type vocabulary.
            item_type_embed_dim: Embedding dim for hotbar item type.
            kit_type_vocab_size: Size of the fitted match kit-type vocabulary.
            kit_type_embed_dim: Embedding dim for match kit type.
            held_item_embed_dim: Embedding dim for the opponent's held-item category.
            hotbar_slot_embed_dim: Embedding dim for the hotbar slot index.
            block_cell_num_types: Number of distinct block-cell enum values
                (fixed by the schema; see ``NUM_BLOCK_CELL_TYPES``).
        """
        super().__init__()
        self.num_block_cells = num_block_cells
        self.block_cell_embedding = nn.Embedding(block_cell_num_types, block_cell_embed_dim)
        self.item_type_embedding = nn.Embedding(item_type_vocab_size, item_type_embed_dim)
        self.kit_type_embedding = nn.Embedding(kit_type_vocab_size, kit_type_embed_dim)
        self.opponent_held_item_embedding = nn.Embedding(
            NUM_HELD_ITEM_CATEGORIES, held_item_embed_dim
        )
        self.hotbar_slot_embedding = nn.Embedding(NUM_HOTBAR_SLOTS, hotbar_slot_embed_dim)

        self.output_dim = (
            CONTINUOUS_FEATURE_DIM
            + num_block_cells * block_cell_embed_dim
            + item_type_embed_dim
            + kit_type_embed_dim
            + held_item_embed_dim
            + hotbar_slot_embed_dim
        )

    def forward(
        self,
        continuous: torch.Tensor,
        block_grid_cells: torch.Tensor,
        hotbar_slot_index: torch.Tensor,
        hotbar_item_type: torch.Tensor,
        opponent_held_item_category: torch.Tensor,
        match_kit_type: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of ticks (optionally with a window dimension).

        Args:
            continuous: Float tensor, shape ``[..., CONTINUOUS_FEATURE_DIM]``.
            block_grid_cells: Int64 tensor, shape ``[..., num_block_cells]``.
            hotbar_slot_index: Int64 tensor, shape ``[...]``.
            hotbar_item_type: Int64 tensor, shape ``[...]``.
            opponent_held_item_category: Int64 tensor, shape ``[...]``.
            match_kit_type: Int64 tensor, shape ``[...]``.

        Returns:
            Float tensor, shape ``[..., self.output_dim]``.
        """
        leading_shape = continuous.shape[:-1]
        block_embed = self.block_cell_embedding(block_grid_cells)
        block_flat = block_embed.reshape(*leading_shape, -1)
        item_embed = self.item_type_embedding(hotbar_item_type)
        kit_embed = self.kit_type_embedding(match_kit_type)
        held_embed = self.opponent_held_item_embedding(opponent_held_item_category)
        slot_embed = self.hotbar_slot_embedding(hotbar_slot_index)
        return torch.cat(
            [continuous, block_flat, item_embed, kit_embed, held_embed, slot_embed], dim=-1
        )
