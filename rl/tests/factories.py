# SPDX-License-Identifier: MIT
"""Test factories: build minimal-but-valid `herbert_rl` objects without a real server/bridge."""

from __future__ import annotations

from herbert_rl.env.reward import RewardWeights
from herbert_rl.nn_cache import NNCacheStats
from herbert_rl.normalization import Standardizer
from herbert_rl.schema import (
    BlockGrid,
    HeldItem,
    InputState,
    MatchState,
    Opponent,
    PlayerState,
    TickRecordRL,
)
from herbert_rl.vocab import CategoricalVocab

DEFAULT_BLOCK_GRID_SHAPE = (7, 3, 7)
DEFAULT_NUM_CELLS = (
    DEFAULT_BLOCK_GRID_SHAPE[0]
    * DEFAULT_BLOCK_GRID_SHAPE[1]
    * DEFAULT_BLOCK_GRID_SHAPE[2]
)


def make_block_grid(
    cells: list[str] | None = None,
    shape: tuple[int, int, int] = DEFAULT_BLOCK_GRID_SHAPE,
) -> BlockGrid:
    width, height, depth = shape
    num_cells = width * height * depth
    if cells is None:
        cells = ["AIR"] * num_cells
    return BlockGrid(
        width=width,
        height=height,
        depth=depth,
        origin="player_feet_centered",
        cells=cells,
    )


def make_input_state(**overrides) -> InputState:
    defaults = {
        "forward": 0,
        "strafe": 0,
        "jump": False,
        "sneak": False,
        "delta_yaw": 0.0,
        "delta_pitch": 0.0,
        "attack_occurred": False,
        "attack_target_type": None,
        "place_occurred": False,
        "place_block_type": None,
        "place_x": None,
        "place_y": None,
        "place_z": None,
    }
    defaults.update(overrides)
    return InputState(**defaults)


def make_player_state(**overrides) -> PlayerState:
    defaults = {
        "x": 0.0,
        "y": 64.0,
        "z": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "on_ground": True,
        "sneaking": False,
        "health": 20.0,
        "food": 20,
    }
    defaults.update(overrides)
    return PlayerState(**defaults)


def make_tick_record(
    tick: int = 0,
    player: PlayerState | None = None,
    block_grid: BlockGrid | None = None,
    held_item: HeldItem | None = None,
    opponent: Opponent | None = None,
    match: MatchState | None = None,
    input: InputState | None = None,
    disconnected: bool = False,
    chat: list[str] | None = None,
) -> TickRecordRL:
    return TickRecordRL(
        tick=tick,
        timestamp="2026-08-05T00:00:00.000Z",
        player=player or make_player_state(),
        block_grid=block_grid or make_block_grid(),
        held_item=held_item
        or HeldItem(hotbar_slot=0, item_id="minecraft:wool", count=64),
        opponent=opponent,
        match=match,
        input=input or make_input_state(),
        disconnected=disconnected,
        chat=chat or [],
    )


def make_opponent(**overrides) -> Opponent:
    defaults = {
        "rel_x": 1.0,
        "rel_y": 0.0,
        "rel_z": 1.0,
        "rel_vx": 0.0,
        "rel_vy": 0.0,
        "rel_vz": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "health": 20.0,
        "held_item_category": "SWORD",
    }
    defaults.update(overrides)
    return Opponent(**defaults)


def _fitted_vocab(name: str, size: int) -> CategoricalVocab:
    """Build a `CategoricalVocab` with exactly ``size`` total entries.

    Includes the 2 reserved special tokens, fit on ``size - 2`` synthetic distinct token
    values.
    """
    vocab = CategoricalVocab(name)
    vocab.fit([f"{name}_{i}" for i in range(max(size - 2, 0))])
    assert vocab.size == size
    return vocab


def make_cache_stats(
    block_grid_shape: tuple[int, int, int] = DEFAULT_BLOCK_GRID_SHAPE,
    item_type_vocab_size: int = 8,
    kit_type_vocab_size: int = 4,
) -> NNCacheStats:
    from herbert_rl.constants import CONTINUOUS_FEATURE_DIM

    standardizer = Standardizer.identity(CONTINUOUS_FEATURE_DIM)
    return NNCacheStats(
        block_grid_shape=block_grid_shape,
        standardizer=standardizer,
        item_type_vocab=_fitted_vocab("hotbar_item_type", item_type_vocab_size),
        kit_type_vocab=_fitted_vocab("match_kit_type", kit_type_vocab_size),
    )


def make_reward_weights(**overrides) -> RewardWeights:
    return RewardWeights(**overrides)
