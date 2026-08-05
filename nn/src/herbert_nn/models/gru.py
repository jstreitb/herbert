# SPDX-License-Identifier: MIT
"""GRUPolicy: the sequence-aware policy over a sliding window of ticks.

Encodes every tick in the window with the same :class:`FeatureEncoder` used
by :class:`herbert_nn.models.mlp.MLPPolicy`, runs the resulting sequence
through a ``nn.GRU``, and feeds the final timestep's hidden state through an
optional MLP trunk and the three shared output heads. Intended to capture
timing/sequential structure (e.g. wind-up before a jump-place, strafe
patterns) that a single-tick model cannot see.
"""

from __future__ import annotations

import torch
from torch import nn

from herbert_nn.models.base import DataMeta, PolicyOutput
from herbert_nn.models.encoder import FeatureEncoder
from herbert_nn.models.heads import BlockPlacementHead, DiscreteHead, MouseHead


class GRUPolicy(nn.Module):
    """Sliding window of ticks -> action heads, via a GRU over encoded per-tick features."""

    def __init__(
        self,
        data_meta: DataMeta,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        trunk_hidden_dims: list[int],
        block_cell_embed_dim: int,
        item_type_embed_dim: int,
        kit_type_embed_dim: int,
        held_item_embed_dim: int,
        hotbar_slot_embed_dim: int,
    ) -> None:
        """Build the encoder, GRU, optional MLP trunk, and heads.

        Args:
            data_meta: Dataset-derived sizes (block-grid shape, vocab sizes).
            hidden_size: GRU hidden state size.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability. Applied between GRU layers (only
                when ``num_layers > 1``, per ``nn.GRU`` semantics) and after
                every trunk hidden layer.
            trunk_hidden_dims: Sizes of successive ``Linear -> ReLU -> Dropout``
                layers applied to the GRU's final hidden state before the
                heads; may be empty to feed the GRU output straight to the heads.
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

        self.gru = nn.GRU(
            input_size=self.encoder.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        layers: list[nn.Module] = []
        in_dim = hidden_size
        for hidden_dim in trunk_hidden_dims:
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
                shaped ``[batch_size, window_length, ...]``.

        Returns:
            A :class:`~herbert_nn.models.base.PolicyOutput` dict, computed
            from the GRU's final-timestep hidden state.
        """
        encoded = self.encoder(
            continuous=batch["continuous"],
            block_grid_cells=batch["block_grid_cells"],
            hotbar_slot_index=batch["hotbar_slot_index"],
            hotbar_item_type=batch["hotbar_item_type"],
            opponent_held_item_category=batch["opponent_held_item_category"],
            match_kit_type=batch["match_kit_type"],
        )  # [batch, window, encoder_dim]
        gru_out, _ = self.gru(encoded)  # [batch, window, hidden_size]
        last_hidden = gru_out[:, -1, :]  # [batch, hidden_size]
        trunk_out = self.trunk(last_hidden)
        return PolicyOutput(
            mouse=self.mouse_head(trunk_out),
            discrete=self.discrete_head(trunk_out),
            block_placement=self.block_placement_head(trunk_out),
        )
