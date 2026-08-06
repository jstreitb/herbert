# SPDX-License-Identifier: MIT
"""Canonical feature layout and enum-to-index mappings shared across the pipeline.

Every module that turns a validated :class:`herbert_nn.schemas.v1_0_0.TickRecordV1`
into tensors (preprocessing, dataset construction, models, evaluation,
inspection) imports the constants defined here so that the feature *order*
and *dimensionality* can never silently drift between preprocessing and the
model's input layer.
"""

from __future__ import annotations

from herbert_nn.schemas.v1_0_0 import BlockCellType, HeldItemCategory

#: Fixed, ordered list of continuous (float) feature names produced per tick.
#: Any change to this list changes ``CONTINUOUS_FEATURE_DIM`` and therefore the
#: model input layer shape -- bump the preprocessing config (which flows into
#: the cache content hash) if you change this so stale caches are invalidated.
CONTINUOUS_FEATURE_NAMES: list[str] = [
    # Player kinematics: position expressed as a delta from the previous
    # tick (0.0 at the first tick of a session) rather than absolute
    # coordinates, since absolute map coordinates do not generalize across
    # different Bridge maps/sessions.
    "player_dx",
    "player_dy",
    "player_dz",
    "player_vx",
    "player_vy",
    "player_vz",
    # Angles encoded as sin/cos to avoid the +-180 degree wraparound
    # discontinuity that plain z-score normalization of raw degrees would
    # introduce.
    "player_yaw_sin",
    "player_yaw_cos",
    "player_pitch_sin",
    "player_pitch_cos",
    "player_health",
    "player_food",
    "player_on_ground",
    "player_is_sneaking",
    # Hotbar
    "hotbar_count_value",
    "hotbar_count_present",
    # Opponent (best-effort detected; zero-filled + flagged when absent)
    "opponent_present",
    "opponent_rel_x",
    "opponent_rel_y",
    "opponent_rel_z",
    "opponent_rel_vx",
    "opponent_rel_vy",
    "opponent_rel_vz",
    "opponent_yaw_sin",
    "opponent_yaw_cos",
    "opponent_pitch_sin",
    "opponent_pitch_cos",
    "opponent_health",
    # Match context (best-effort detected; each field independently nullable)
    "match_context_present",
    "match_own_score_value",
    "match_own_score_present",
    "match_opponent_score_value",
    "match_opponent_score_present",
    "match_elapsed_seconds_value",
    "match_elapsed_seconds_present",
]

CONTINUOUS_FEATURE_DIM: int = len(CONTINUOUS_FEATURE_NAMES)

#: Ordered mapping of block-grid cell enum values to embedding indices.
BLOCK_CELL_TYPE_TO_INDEX: dict[str, int] = {
    BlockCellType.AIR.value: 0,
    BlockCellType.SOLID_BRIDGEABLE.value: 1,
    BlockCellType.LIQUID.value: 2,
    BlockCellType.VOID.value: 3,
    BlockCellType.OTHER_SOLID.value: 4,
}
NUM_BLOCK_CELL_TYPES: int = len(BLOCK_CELL_TYPE_TO_INDEX)

#: Opponent held-item category, index 0 reserved for "no opponent detected".
HELD_ITEM_CATEGORY_ABSENT_INDEX: int = 0
HELD_ITEM_CATEGORY_TO_INDEX: dict[str, int] = {
    HeldItemCategory.SWORD.value: 1,
    HeldItemCategory.BOW.value: 2,
    HeldItemCategory.BLOCKS.value: 3,
    HeldItemCategory.OTHER.value: 4,
}
NUM_HELD_ITEM_CATEGORIES: int = len(HELD_ITEM_CATEGORY_TO_INDEX) + 1  # + absent

NUM_HOTBAR_SLOTS: int = 9

#: Special tokens shared by every open-vocabulary categorical field
#: (hotbar item type, match kit type, placed block type). Index 0/1 are
#: reserved across all such vocabularies so embedding tables are consistent.
NULL_TOKEN: str = "<NULL>"
UNK_TOKEN: str = "<UNK>"
VOCAB_NULL_INDEX: int = 0
VOCAB_UNK_INDEX: int = 1

#: Names (and order) of the four binary action fields predicted by DiscreteHead.
DISCRETE_ACTION_NAMES: list[str] = ["jump", "sneak", "attack", "place"]
NUM_DISCRETE_ACTIONS: int = len(DISCRETE_ACTION_NAMES)

#: Dimensionality of the MouseHead regression target (d_yaw, d_pitch).
MOUSE_TARGET_DIM: int = 2

#: Names (and order) of the two ternary movement axes predicted by MovementHead.
MOVEMENT_AXIS_NAMES: list[str] = ["forward", "strafe"]
#: Dimensionality of the MovementHead regression target (forward, strafe), each in {-1, 0, 1}.
MOVEMENT_TARGET_DIM: int = 2
