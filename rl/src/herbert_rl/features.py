# SPDX-License-Identifier: MIT
"""Streaming, single-tick feature encoding: `TickRecordRL` -> the `/nn`-compatible feature arrays.

This is the RL-side counterpart of `herbert_nn.data.features.encode_session_raw`, adapted for
*online* use: the RL environment sees one tick at a time as it arrives from a bridge process,
rather than a whole session's records at once. The feature *definitions* (which fields go where,
sin/cos angle encoding, position deltas, presence flags) are identical to `/nn`'s -- see
`constants.py` for the sync-point disclaimer -- only the calling convention differs (one record
in, one feature row out, with position-delta state carried across calls instead of computed via
`numpy` array slicing over a whole session).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from herbert_rl.constants import (
    BLOCK_CELL_TYPE_TO_INDEX,
    CONTINUOUS_FEATURE_DIM,
    CONTINUOUS_FEATURE_NAMES,
    HELD_ITEM_CATEGORY_ABSENT_INDEX,
    HELD_ITEM_CATEGORY_TO_INDEX,
)
from herbert_rl.nn_cache import NNCacheStats
from herbert_rl.schema import TickRecordRL

_IDX = {name: i for i, name in enumerate(CONTINUOUS_FEATURE_NAMES)}


class BlockGridShapeError(ValueError):
    """Raised when a bridge-emitted tick's block grid does not match the configured cache shape."""


@dataclass
class TickFeatures:
    """One tick's raw-encoded, `/nn`-compatible feature tensors (ready for `policy/backbone.py`)."""

    continuous: np.ndarray  # (CONTINUOUS_FEATURE_DIM,) float32, standardized
    block_grid_cells: np.ndarray  # (num_cells,) int64, embedding indices
    hotbar_slot_index: np.int64
    hotbar_item_type: np.int64  # vocab-encoded
    opponent_held_item_category: np.int64
    match_kit_type: np.int64  # vocab-encoded


class TickFeatureEncoder:
    """Stateful per-episode encoder: turns raw `TickRecordRL`s into standardized feature arrays.

    Carries the previous tick's position across calls (to compute `player_dx/dy/dz`, matching
    `/nn`'s "delta from previous tick, zero at session start" convention) -- call :meth:`reset`
    at the start of every episode so the first tick's deltas are correctly zeroed rather than
    leaking the previous episode's final position.
    """

    def __init__(self, cache_stats: NNCacheStats) -> None:
        """Initialize the encoder.

        Args:
            cache_stats: Fitted standardizer/vocabularies/block-grid shape from the `/nn`
                preprocessing cache the checkpoint being fine-tuned was trained against.
        """
        self.cache_stats = cache_stats
        self._prev_x: float | None = None
        self._prev_y: float | None = None
        self._prev_z: float | None = None

    def reset(self) -> None:
        """Clear position-delta state; call once at the start of each new episode."""
        self._prev_x = None
        self._prev_y = None
        self._prev_z = None

    def encode(self, record: TickRecordRL) -> TickFeatures:
        """Encode one tick's raw record into standardized, `/nn`-compatible feature arrays.

        Raises:
            BlockGridShapeError: If ``record.block_grid``'s dimensions don't match the shape the
                configured `/nn` cache was built with.
        """
        shape = (
            record.block_grid.width,
            record.block_grid.height,
            record.block_grid.depth,
        )
        if shape != self.cache_stats.block_grid_shape:
            raise BlockGridShapeError(
                f"tick {record.tick}: block_grid shape {shape} != /nn cache's canonical shape "
                f"{self.cache_stats.block_grid_shape}. Reconfigure the bridge's blockGridWidth/"
                "Height/Depth to match the /nn cache the checkpoint was trained against."
            )
        expected_cells = self.cache_stats.num_block_cells
        if len(record.block_grid.cells) != expected_cells:
            raise BlockGridShapeError(
                f"tick {record.tick}: block_grid has {len(record.block_grid.cells)} cells, "
                f"expected {expected_cells}."
            )

        continuous = np.zeros((CONTINUOUS_FEATURE_DIM,), dtype=np.float32)
        p = record.player

        dx = 0.0 if self._prev_x is None else p.x - self._prev_x
        dy = 0.0 if self._prev_y is None else p.y - self._prev_y
        dz = 0.0 if self._prev_z is None else p.z - self._prev_z
        self._prev_x, self._prev_y, self._prev_z = p.x, p.y, p.z

        yaw_sin, yaw_cos = _sin_cos(p.yaw)
        pitch_sin, pitch_cos = _sin_cos(p.pitch)

        continuous[_IDX["player_dx"]] = dx
        continuous[_IDX["player_dy"]] = dy
        continuous[_IDX["player_dz"]] = dz
        continuous[_IDX["player_vx"]] = p.vx
        continuous[_IDX["player_vy"]] = p.vy
        continuous[_IDX["player_vz"]] = p.vz
        continuous[_IDX["player_yaw_sin"]] = yaw_sin
        continuous[_IDX["player_yaw_cos"]] = yaw_cos
        continuous[_IDX["player_pitch_sin"]] = pitch_sin
        continuous[_IDX["player_pitch_cos"]] = pitch_cos
        continuous[_IDX["player_health"]] = p.health
        continuous[_IDX["player_food"]] = float(p.food)
        continuous[_IDX["player_on_ground"]] = float(p.on_ground)
        continuous[_IDX["player_is_sneaking"]] = float(p.sneaking)

        continuous[_IDX["hotbar_count_value"]] = float(record.held_item.count)
        continuous[_IDX["hotbar_count_present"]] = float(record.held_item.count > 0)

        block_grid_cells = np.asarray(
            [BLOCK_CELL_TYPE_TO_INDEX[cell.value] for cell in record.block_grid.cells],
            dtype=np.int64,
        )

        opponent_held_item_category = np.int64(HELD_ITEM_CATEGORY_ABSENT_INDEX)
        if record.opponent is not None:
            o = record.opponent
            continuous[_IDX["opponent_present"]] = 1.0
            continuous[_IDX["opponent_rel_x"]] = o.rel_x
            continuous[_IDX["opponent_rel_y"]] = o.rel_y
            continuous[_IDX["opponent_rel_z"]] = o.rel_z
            continuous[_IDX["opponent_rel_vx"]] = o.rel_vx
            continuous[_IDX["opponent_rel_vy"]] = o.rel_vy
            continuous[_IDX["opponent_rel_vz"]] = o.rel_vz
            o_yaw_sin, o_yaw_cos = _sin_cos(o.yaw)
            o_pitch_sin, o_pitch_cos = _sin_cos(o.pitch)
            continuous[_IDX["opponent_yaw_sin"]] = o_yaw_sin
            continuous[_IDX["opponent_yaw_cos"]] = o_yaw_cos
            continuous[_IDX["opponent_pitch_sin"]] = o_pitch_sin
            continuous[_IDX["opponent_pitch_cos"]] = o_pitch_cos
            continuous[_IDX["opponent_health"]] = o.health
            opponent_held_item_category = np.int64(
                HELD_ITEM_CATEGORY_TO_INDEX[o.held_item_category.value]
            )

        if record.match is not None:
            mc = record.match
            continuous[_IDX["match_context_present"]] = 1.0
            if mc.own_score is not None:
                continuous[_IDX["match_own_score_value"]] = float(mc.own_score)
                continuous[_IDX["match_own_score_present"]] = 1.0
            if mc.opponent_score is not None:
                continuous[_IDX["match_opponent_score_value"]] = float(
                    mc.opponent_score
                )
                continuous[_IDX["match_opponent_score_present"]] = 1.0
            if mc.elapsed_seconds is not None:
                continuous[_IDX["match_elapsed_seconds_value"]] = float(
                    mc.elapsed_seconds
                )
                continuous[_IDX["match_elapsed_seconds_present"]] = 1.0

        continuous = self.cache_stats.standardizer.transform(continuous[np.newaxis, :])[
            0
        ]

        return TickFeatures(
            continuous=continuous,
            block_grid_cells=block_grid_cells,
            hotbar_slot_index=np.int64(record.held_item.hotbar_slot),
            hotbar_item_type=np.int64(
                self.cache_stats.item_type_vocab.encode(record.held_item.item_id)
            ),
            opponent_held_item_category=opponent_held_item_category,
            match_kit_type=np.int64(
                self.cache_stats.kit_type_vocab.encode(
                    record.match.kit if record.match else None
                )
            ),
        )


def _sin_cos(degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    return math.sin(radians), math.cos(radians)
