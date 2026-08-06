# SPDX-License-Identifier: MIT
"""Turn validated :class:`TickRecordV1` sequences into raw per-tick feature arrays.

This module performs *feature engineering* (position deltas, sin/cos angle
encoding, presence-flag construction) but deliberately stops short of
z-score normalization or open-vocabulary index encoding: those depend on
statistics/vocabularies fitted on the training split only, and are applied
later by :mod:`herbert_nn.data.preprocess`. Everything in this module is a
pure, split-agnostic function of a single session's records.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from herbert_nn.constants import (
    BLOCK_CELL_TYPE_TO_INDEX,
    CONTINUOUS_FEATURE_DIM,
    CONTINUOUS_FEATURE_NAMES,
    HELD_ITEM_CATEGORY_ABSENT_INDEX,
    HELD_ITEM_CATEGORY_TO_INDEX,
)
from herbert_nn.schemas.v1_0_0 import TickRecordV1

logger = logging.getLogger(__name__)

#: Index of each continuous feature name, for readable assignment below.
_IDX = {name: i for i, name in enumerate(CONTINUOUS_FEATURE_NAMES)}


class BlockGridShapeError(ValueError):
    """Raised when a session's block grids do not match the expected/canonical shape."""


@dataclass
class SessionArrays:
    """Raw (pre-normalization, pre-vocab-encoding) per-tick arrays for one session.

    All arrays share a leading dimension ``T`` = number of ticks in the
    session (after any filtering).
    """

    session_id: str
    tick: np.ndarray  # (T,) int64
    continuous: (
        np.ndarray
    )  # (T, CONTINUOUS_FEATURE_DIM) float32, RAW (not yet standardized)
    block_grid_cells: np.ndarray  # (T, num_cells) int64, embedding indices (fixed enum)
    hotbar_slot_index: np.ndarray  # (T,) int64
    hotbar_item_type_raw: list[str | None]
    opponent_held_item_category: np.ndarray  # (T,) int64, fixed enum -> already encoded
    match_kit_type_raw: list[str | None]
    mouse_target: np.ndarray  # (T, 2) float32: delta_yaw, delta_pitch
    discrete_target: (
        np.ndarray
    )  # (T, 4) float32: jump, sneak, attack_occurred, place_occurred
    place_block_type_raw: list[str | None]
    place_mask: np.ndarray  # (T,) float32
    movement_target: (
        np.ndarray
    )  # (T, 2) float32: forward, strafe (raw -1/0/1, not remapped)


def _sin_cos(degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    return math.sin(radians), math.cos(radians)


def encode_session_raw(
    records: list[TickRecordV1],
    session_id: str,
    block_grid_shape: tuple[int, int, int],
) -> SessionArrays:
    """Convert one session's validated tick records into raw numpy feature arrays.

    Args:
        records: Tick records for a single session, in tick order.
        session_id: The owning session's id (for array bookkeeping and error messages).
        block_grid_shape: The ``(width, height, depth)`` every record's
            ``block_grid`` must match; a mismatch raises
            :class:`BlockGridShapeError` with the offending tick number.

    Returns:
        A :class:`SessionArrays` with one row per input record.

    Raises:
        BlockGridShapeError: If any record's block grid shape/length does not
            match ``block_grid_shape``.
    """
    num_ticks = len(records)
    continuous = np.zeros((num_ticks, CONTINUOUS_FEATURE_DIM), dtype=np.float32)
    expected_cells = block_grid_shape[0] * block_grid_shape[1] * block_grid_shape[2]
    block_grid_cells = np.zeros((num_ticks, expected_cells), dtype=np.int64)
    hotbar_slot_index = np.zeros((num_ticks,), dtype=np.int64)
    hotbar_item_type_raw: list[str | None] = [None] * num_ticks
    opponent_held_item_category = np.full(
        (num_ticks,), HELD_ITEM_CATEGORY_ABSENT_INDEX, dtype=np.int64
    )
    match_kit_type_raw: list[str | None] = [None] * num_ticks
    mouse_target = np.zeros((num_ticks, 2), dtype=np.float32)
    discrete_target = np.zeros((num_ticks, 4), dtype=np.float32)
    place_block_type_raw: list[str | None] = [None] * num_ticks
    place_mask = np.zeros((num_ticks,), dtype=np.float32)
    movement_target = np.zeros((num_ticks, 2), dtype=np.float32)
    tick_arr = np.zeros((num_ticks,), dtype=np.int64)

    prev_x: float | None = None
    prev_y: float | None = None
    prev_z: float | None = None

    for i, record in enumerate(records):
        shape = (
            record.block_grid.width,
            record.block_grid.height,
            record.block_grid.depth,
        )
        if shape != block_grid_shape:
            raise BlockGridShapeError(
                f"session {session_id!r} tick {record.tick}: block_grid shape {shape} "
                f"!= expected canonical shape {block_grid_shape}."
            )
        if len(record.block_grid.cells) != expected_cells:
            raise BlockGridShapeError(
                f"session {session_id!r} tick {record.tick}: block_grid has "
                f"{len(record.block_grid.cells)} cells, expected {expected_cells} "
                f"(= {block_grid_shape[0]}*{block_grid_shape[1]}*{block_grid_shape[2]})."
            )

        tick_arr[i] = record.tick
        p = record.player

        dx = 0.0 if prev_x is None else p.x - prev_x
        dy = 0.0 if prev_y is None else p.y - prev_y
        dz = 0.0 if prev_z is None else p.z - prev_z
        prev_x, prev_y, prev_z = p.x, p.y, p.z

        yaw_sin, yaw_cos = _sin_cos(p.yaw)
        pitch_sin, pitch_cos = _sin_cos(p.pitch)

        row = continuous[i]
        row[_IDX["player_dx"]] = dx
        row[_IDX["player_dy"]] = dy
        row[_IDX["player_dz"]] = dz
        row[_IDX["player_vx"]] = p.vx
        row[_IDX["player_vy"]] = p.vy
        row[_IDX["player_vz"]] = p.vz
        row[_IDX["player_yaw_sin"]] = yaw_sin
        row[_IDX["player_yaw_cos"]] = yaw_cos
        row[_IDX["player_pitch_sin"]] = pitch_sin
        row[_IDX["player_pitch_cos"]] = pitch_cos
        row[_IDX["player_health"]] = p.health
        row[_IDX["player_food"]] = float(p.food)
        row[_IDX["player_on_ground"]] = float(p.on_ground)
        row[_IDX["player_is_sneaking"]] = float(p.sneaking)

        hotbar_slot_index[i] = record.held_item.hotbar_slot
        hotbar_item_type_raw[i] = record.held_item.item_id
        row[_IDX["hotbar_count_value"]] = float(record.held_item.count)
        row[_IDX["hotbar_count_present"]] = float(record.held_item.count > 0)

        block_grid_cells[i] = [
            BLOCK_CELL_TYPE_TO_INDEX[cell.value] for cell in record.block_grid.cells
        ]

        if record.opponent is not None:
            o = record.opponent
            row[_IDX["opponent_present"]] = 1.0
            row[_IDX["opponent_rel_x"]] = o.rel_x
            row[_IDX["opponent_rel_y"]] = o.rel_y
            row[_IDX["opponent_rel_z"]] = o.rel_z
            row[_IDX["opponent_rel_vx"]] = o.rel_vx
            row[_IDX["opponent_rel_vy"]] = o.rel_vy
            row[_IDX["opponent_rel_vz"]] = o.rel_vz
            o_yaw_sin, o_yaw_cos = _sin_cos(o.yaw)
            o_pitch_sin, o_pitch_cos = _sin_cos(o.pitch)
            row[_IDX["opponent_yaw_sin"]] = o_yaw_sin
            row[_IDX["opponent_yaw_cos"]] = o_yaw_cos
            row[_IDX["opponent_pitch_sin"]] = o_pitch_sin
            row[_IDX["opponent_pitch_cos"]] = o_pitch_cos
            row[_IDX["opponent_health"]] = o.health
            opponent_held_item_category[i] = HELD_ITEM_CATEGORY_TO_INDEX[
                o.held_item_category.value
            ]

        if record.match is not None:
            mc = record.match
            row[_IDX["match_context_present"]] = 1.0
            if mc.own_score is not None:
                row[_IDX["match_own_score_value"]] = float(mc.own_score)
                row[_IDX["match_own_score_present"]] = 1.0
            if mc.opponent_score is not None:
                row[_IDX["match_opponent_score_value"]] = float(mc.opponent_score)
                row[_IDX["match_opponent_score_present"]] = 1.0
            if mc.elapsed_seconds is not None:
                row[_IDX["match_elapsed_seconds_value"]] = float(mc.elapsed_seconds)
                row[_IDX["match_elapsed_seconds_present"]] = 1.0
            match_kit_type_raw[i] = mc.kit

        inp = record.input
        mouse_target[i, 0] = inp.delta_yaw
        mouse_target[i, 1] = inp.delta_pitch
        movement_target[i, 0] = float(inp.forward)
        movement_target[i, 1] = float(inp.strafe)
        discrete_target[i, 0] = float(inp.jump)
        discrete_target[i, 1] = float(inp.sneak)
        discrete_target[i, 2] = float(inp.attack_occurred)
        discrete_target[i, 3] = float(inp.place_occurred)
        if inp.place_occurred:
            place_block_type_raw[i] = inp.place_block_type
            place_mask[i] = 1.0

    return SessionArrays(
        session_id=session_id,
        tick=tick_arr,
        continuous=continuous,
        block_grid_cells=block_grid_cells,
        hotbar_slot_index=hotbar_slot_index,
        hotbar_item_type_raw=hotbar_item_type_raw,
        opponent_held_item_category=opponent_held_item_category,
        match_kit_type_raw=match_kit_type_raw,
        mouse_target=mouse_target,
        discrete_target=discrete_target,
        place_block_type_raw=place_block_type_raw,
        place_mask=place_mask,
        movement_target=movement_target,
    )
