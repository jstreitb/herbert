# SPDX-License-Identifier: MIT
"""MLPPolicy: the single-tick baseline policy.

Feeds one tick's encoded feature vector through a plain feed-forward trunk
and the three shared output heads. Fastest to train; no notion of
history/timing -- use :class:`herbert_nn.models.gru.GRUPolicy` to capture
sequential structure.
"""

from __future__ import annotations

import torch
from torch import nn

from herbert_nn.models.base import DataMeta, PolicyOutput
from herbert_nn.models.encoder import FeatureEncoder
from herbert_nn.models.heads import BlockPlacementHead, DiscreteHead, MouseHead


class MLPPolicy(nn.Module):
    """Single-tick feature vector -> action heads."""

    def __init__(
        self,
        data_meta: DataMeta,
        hidden_dims: list[int],
        dropout: float,
        block_cell_embed_dim: int,
        item_type_embed_dim: int,
        kit_type_embed_dim: int,
        held_item_embed_dim: int,
        hotbar_slot_embed_dim: int,
    ) -> None:
        """Build the encoder, MLP trunk, and heads.

        Args:
            data_meta: Dataset-derived sizes (block-grid shape, vocab sizes).
            hidden_dims: Sizes of successive ``Linear -> ReLU -> Dropout``
                trunk layers, e.g. ``[256, 128]``.
            dropout: Dropout probability applied after every trunk hidden layer.
            block_cell_embed_dim: Embedding dim per block-grid cell.
            item_type_embed_dim: Embedding dim for hotbar item type.
            kit_type_embed_dim: Embedding dim for match kit type.
            held_item_embed_dim: Embedding dim for opponent held-item category.
            hotbar_slot_embed_dim: Embedding dim for the hotbar slot index.
        """
        super().__init__()
        self.data_meta = data_meta
        self.encoder = FeatureEncoder(
            num_block_cells=data_meta.num_block_cells,
            block_cell_embed_dim=block_cell_embed_dim,
            item_type_vocab_size=data_meta.item_type_vocab_size,
            item_type_embed_dim=item_type_embed_dim,
            kit_type_vocab_size=data_meta.kit_type_vocab_size,
            kit_type_embed_dim=kit_type_embed_dim,
            held_item_embed_dim=held_item_embed_dim,
            hotbar_slot_embed_dim=hotbar_slot_embed_dim,
        )

        layers: list[nn.Module] = []
        in_dim = self.encoder.output_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        trunk_dim = in_dim

        self.mouse_head = MouseHead(trunk_dim)
        self.discrete_head = DiscreteHead(trunk_dim)
        self.block_placement_head = BlockPlacementHead(
            trunk_dim, data_meta.place_block_type_vocab_size
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> PolicyOutput:
        """Run a forward pass.

        Args:
            batch: Dict with (at least) keys ``continuous``, ``block_grid_cells``,
                ``hotbar_slot_index``, ``hotbar_item_type``,
                ``opponent_held_item_category``, ``match_kit_type`` -- each
                shaped ``[batch_size, ...]`` (single-tick, no window dim).

        Returns:
            A :class:`~herbert_nn.models.base.PolicyOutput` dict.
        """
        encoded = self.encoder(
            continuous=batch["continuous"],
            block_grid_cells=batch["block_grid_cells"],
            hotbar_slot_index=batch["hotbar_slot_index"],
            hotbar_item_type=batch["hotbar_item_type"],
            opponent_held_item_category=batch["opponent_held_item_category"],
            match_kit_type=batch["match_kit_type"],
        )
        trunk_out = self.trunk(encoded)
        return PolicyOutput(
            mouse=self.mouse_head(trunk_out),
            discrete=self.discrete_head(trunk_out),
            block_placement=self.block_placement_head(trunk_out),
        )
