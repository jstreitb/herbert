"""Canonical feature layout and enum-to-index mappings, kept in lockstep with `/nn`.

**Sync point, not a runtime import.** `/rl` is not allowed to import `herbert_nn` at runtime
(the RL bridge/trainer must run standalone against a live Minecraft server, independent of the
`/nn` package's installation), so this module is a hand-maintained *copy* of the feature layout
defined in [`nn/src/herbert_nn/constants.py`](../../../nn/src/herbert_nn/constants.py). If that
file changes (a new continuous feature, a new block-cell enum value, a reordered feature list),
this file must be updated to match by hand, or a checkpoint produced by `/nn` will silently
misalign with the RL environment's observation tensors.

The two lists this file is most sensitive to are ``CONTINUOUS_FEATURE_NAMES`` (order matters --
it is the model's input layer ordering) and ``BLOCK_CELL_TYPE_TO_INDEX`` (must match the
embedding table indices baked into any pretrained `/nn` checkpoint).
"""

from __future__ import annotations

#: Fixed, ordered list of continuous (float) feature names produced per tick.
#: MUST match `herbert_nn.constants.CONTINUOUS_FEATURE_NAMES` exactly (name AND order).
CONTINUOUS_FEATURE_NAMES: list[str] = [
    "player_dx",
    "player_dy",
    "player_dz",
    "player_vx",
    "player_vy",
    "player_vz",
    "player_yaw_sin",
    "player_yaw_cos",
    "player_pitch_sin",
    "player_pitch_cos",
    "player_health",
    "player_food",
    "player_on_ground",
    "player_is_sneaking",
    "hotbar_count_value",
    "hotbar_count_present",
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
#: MUST match `herbert_nn.constants.BLOCK_CELL_TYPE_TO_INDEX`.
BLOCK_CELL_TYPE_TO_INDEX: dict[str, int] = {
    "AIR": 0,
    "SOLID_BRIDGEABLE": 1,
    "LIQUID": 2,
    "VOID": 3,
    "OTHER_SOLID": 4,
}
NUM_BLOCK_CELL_TYPES: int = len(BLOCK_CELL_TYPE_TO_INDEX)

#: Opponent held-item category, index 0 reserved for "no opponent detected".
#: MUST match `herbert_nn.constants.HELD_ITEM_CATEGORY_TO_INDEX`.
HELD_ITEM_CATEGORY_ABSENT_INDEX: int = 0
HELD_ITEM_CATEGORY_TO_INDEX: dict[str, int] = {
    "SWORD": 1,
    "BOW": 2,
    "BLOCKS": 3,
    "OTHER": 4,
}
NUM_HELD_ITEM_CATEGORIES: int = len(HELD_ITEM_CATEGORY_TO_INDEX) + 1  # + absent

NUM_HOTBAR_SLOTS: int = 9

#: Special tokens shared by every open-vocabulary categorical field (hotbar item type, match
#: kit type). MUST match `herbert_nn.constants`.
NULL_TOKEN: str = "<NULL>"
UNK_TOKEN: str = "<UNK>"
VOCAB_NULL_INDEX: int = 0
VOCAB_UNK_INDEX: int = 1

#: Names (and order) of the four binary action fields predicted by `/nn`'s DiscreteHead.
#: RL keeps this exact order because the checkpoint-adapter splices DiscreteHead's pretrained
#: weight rows directly into the PPO action head at these positions.
DISCRETE_ACTION_NAMES: list[str] = ["jump", "sneak", "attack", "place"]
NUM_DISCRETE_ACTIONS: int = len(DISCRETE_ACTION_NAMES)

#: Dimensionality of `/nn`'s MouseHead regression target (d_yaw, d_pitch).
MOUSE_TARGET_DIM: int = 2

#: Movement axes the RL action space adds on top of `/nn`'s action heads (see
#: "Known limitations" in `nn/README.md`: `/nn` does not model forward/strafe at all). These are
#: new, randomly-initialized outputs -- there is no pretrained weight to reuse for them.
MOVEMENT_AXIS_NAMES: list[str] = ["move_forward", "strafe"]
#: Each movement axis is ternary: -1 (backward/left), 0 (none), 1 (forward/right).
MOVEMENT_AXIS_VALUES: list[int] = [-1, 0, 1]
