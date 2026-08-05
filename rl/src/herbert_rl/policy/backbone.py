"""RL policy backbone: a hand-synced architectural copy of `/nn`'s `FeatureEncoder` +
`MLPPolicy`/`GRUPolicy` trunks (see `nn/src/herbert_nn/models/{encoder,mlp,gru}.py`).

**Why duplicated instead of imported:** `/rl` may not import `herbert_nn` at runtime (see
`constants.py`), but a `/nn` checkpoint's `model_state_dict` is a plain dict of tensors keyed by
submodule attribute path (e.g. ``"encoder.block_cell_embedding.weight"``, ``"gru.weight_ih_l0"``,
``"trunk.0.weight"``). By rebuilding an architecturally identical module tree here -- same
submodule names, same layer order, same shapes -- `policy/checkpoint_adapter.py` can load a `/nn`
checkpoint's weights directly via `nn.Module.load_state_dict(..., strict=False)` (the ``False``
tolerates the checkpoint's `mouse_head`/`discrete_head`/`block_placement_head` keys, which this
module intentionally omits -- see `checkpoint_adapter.py` for where those get spliced into the
PPO action head instead of being loaded as ordinary submodules).

If `/nn`'s encoder/trunk architecture changes, this file must be updated by hand to match, or
checkpoint loading will silently drop weights (`load_state_dict(strict=False)` only warns, it
doesn't raise) -- `checkpoint_adapter.py` logs every missing/unexpected key at load time
specifically so a drift like this is caught early rather than silently training from
random-init weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

#: MUST match `herbert_nn.constants.CONTINUOUS_FEATURE_DIM` / `NUM_HELD_ITEM_CATEGORIES` /
#: `NUM_HOTBAR_SLOTS` -- see `herbert_rl.constants`.
from herbert_rl.constants import CONTINUOUS_FEATURE_DIM, NUM_HELD_ITEM_CATEGORIES, NUM_HOTBAR_SLOTS


@dataclass(frozen=True)
class RLDataMeta:
    """The subset of `/nn`'s `DataMeta` the RL backbone needs (no `place_block_type_vocab_size`
    -- block-placement *type* selection is out of scope for the RL action space; see
    `env/spaces.py`)."""

    block_grid_shape: tuple[int, int, int]
    item_type_vocab_size: int
    kit_type_vocab_size: int

    @property
    def num_block_cells(self) -> int:
        w, h, d = self.block_grid_shape
        return w * h * d

    @classmethod
    def from_checkpoint_data_meta(cls, data_meta: dict) -> RLDataMeta:
        """Build from a `/nn` checkpoint's embedded ``data_meta`` dict (extra keys ignored)."""
        return cls(
            block_grid_shape=tuple(data_meta["block_grid_shape"]),  # type: ignore[arg-type]
            item_type_vocab_size=int(data_meta["item_type_vocab_size"]),
            kit_type_vocab_size=int(data_meta["kit_type_vocab_size"]),
        )


class RLFeatureEncoder(nn.Module):
    """Architectural copy of `herbert_nn.models.encoder.FeatureEncoder`. Submodule names and
    forward-pass shape handling are identical so a checkpoint's `encoder.*` weights load as-is.
    """

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


class RLMLPBackbone(nn.Module):
    """Architectural copy of `herbert_nn.models.mlp.MLPPolicy`, minus the three output heads.

    Consumes the *last* tick of a windowed batch (``[batch, window, ...]``, window collapsed to
    the final timestep) -- ``window_length`` is expected to be 1 for an MLP-family checkpoint,
    but this indexing keeps the same code path correct even if it isn't.
    """

    def __init__(
        self,
        data_meta: RLDataMeta,
        hidden_dims: list[int],
        dropout: float,
        block_cell_embed_dim: int,
        item_type_embed_dim: int,
        kit_type_embed_dim: int,
        held_item_embed_dim: int,
        hotbar_slot_embed_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = RLFeatureEncoder(
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
        self.output_dim = in_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        last = {key: tensor[:, -1] for key, tensor in batch.items()}
        encoded = self.encoder(
            continuous=last["continuous"],
            block_grid_cells=last["block_grid_cells"],
            hotbar_slot_index=last["hotbar_slot_index"],
            hotbar_item_type=last["hotbar_item_type"],
            opponent_held_item_category=last["opponent_held_item_category"],
            match_kit_type=last["match_kit_type"],
        )
        return self.trunk(encoded)


class RLGRUBackbone(nn.Module):
    """Architectural copy of `herbert_nn.models.gru.GRUPolicy`, minus the three output heads."""

    def __init__(
        self,
        data_meta: RLDataMeta,
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
        super().__init__()
        self.encoder = RLFeatureEncoder(
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
        self.output_dim = in_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.encoder(
            continuous=batch["continuous"],
            block_grid_cells=batch["block_grid_cells"],
            hotbar_slot_index=batch["hotbar_slot_index"],
            hotbar_item_type=batch["hotbar_item_type"],
            opponent_held_item_category=batch["opponent_held_item_category"],
            match_kit_type=batch["match_kit_type"],
        )
        gru_out, _ = self.gru(encoded)
        last_hidden = gru_out[:, -1, :]
        return self.trunk(last_hidden)


def build_backbone(model_cfg: dict, data_meta: RLDataMeta) -> nn.Module:
    """Instantiate `RLMLPBackbone` or `RLGRUBackbone` from a `/nn`-shaped ``model_cfg`` dict
    (the same ``family``/hyperparameter keys as `nn/conf/model/{mlp,gru}.yaml`, and the same
    dict embedded verbatim in a `/nn` checkpoint's ``model_cfg`` field).
    """
    family = model_cfg["family"]
    if family == "mlp":
        return RLMLPBackbone(
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
        return RLGRUBackbone(
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
