"""Per-tick observation schema emitted by `/rl/bridge`, byte-for-byte compatible with `/mod`.

**Sync point, not a runtime import.** This is a hand-maintained copy of the per-tick portion of
[`nn/src/herbert_nn/schemas/v1_0_0.py`](../../../nn/src/herbert_nn/schemas/v1_0_0.py) (schema
version ``"1.0.0"``), which is itself a re-model of the JSONL contract documented in
`mod/README.md` ("JSONL schema (contract for `/nn`)"). `/rl` cannot import `herbert_nn` at
runtime (see `constants.py` docstring), so the field names, types, and nullability here are
copied by hand and must be kept in sync with both of those sources.

The one deliberate difference from `/mod`'s output: a `/rl/bridge` process emits only
:class:`TickRecordRL` lines (no :class:`SessionHeaderV1`-equivalent header line), since a bridge
process is not itself the owner of Herbert session-log provenance metadata -- the Python training
layer knows the session/run identity already. If RL-collected ticks are ever written to disk as a
standalone ``.jsonl`` file for reuse by `/nn`'s preprocessing pipeline, a conforming header line
must be prepended first (see `herbert_rl.logging_utils.write_session_header` for a helper that
does this).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

#: The exact `/mod` schema_version this module's field set corresponds to.
SCHEMA_VERSION = "1.0.0"


class _StrictModel(BaseModel):
    """Forbids unknown fields so a bridge/schema drift fails loudly, not silently."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BlockCellType(str, Enum):
    AIR = "AIR"
    SOLID_BRIDGEABLE = "SOLID_BRIDGEABLE"
    LIQUID = "LIQUID"
    VOID = "VOID"
    OTHER_SOLID = "OTHER_SOLID"


class HeldItemCategory(str, Enum):
    SWORD = "SWORD"
    BOW = "BOW"
    BLOCKS = "BLOCKS"
    OTHER = "OTHER"


class PlayerState(_StrictModel):
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    yaw: float
    pitch: float
    on_ground: bool
    sneaking: bool
    health: float
    food: int


class BlockGrid(_StrictModel):
    width: int
    height: int
    depth: int
    origin: str
    cells: list[BlockCellType]


class HeldItem(_StrictModel):
    hotbar_slot: int = Field(..., ge=0, le=8)
    item_id: str | None = None
    count: int


class Opponent(_StrictModel):
    rel_x: float
    rel_y: float
    rel_z: float
    rel_vx: float
    rel_vy: float
    rel_vz: float
    yaw: float
    pitch: float
    health: float
    held_item_category: HeldItemCategory


class MatchState(_StrictModel):
    own_score: int | None = None
    opponent_score: int | None = None
    elapsed_seconds: int | None = None
    kit: str | None = None


class InputState(_StrictModel):
    """The action that was in effect for this tick.

    For `/mod`, this is the human's recorded input (the imitation-learning target). For
    `/rl/bridge`, this is an *echo* of the action command the Python env sent for this tick
    (see `env/ipc.py`), so an RL trajectory logged to disk has the exact same shape as a human
    session and can be run through `/nn`'s preprocessing unmodified.
    """

    forward: int = Field(..., ge=-1, le=1)
    strafe: int = Field(..., ge=-1, le=1)
    jump: bool
    sneak: bool
    delta_yaw: float
    delta_pitch: float
    attack_occurred: bool
    attack_target_type: str | None = None
    place_occurred: bool
    place_block_type: str | None = None
    place_x: int | None = None
    place_y: int | None = None
    place_z: int | None = None


class TickRecordRL(_StrictModel):
    """A single per-tick observation record emitted by a `/rl/bridge` process on stdout.

    Field-for-field identical to `/mod`'s `TickRecordV1` (minus the session-level header line;
    see module docstring). ``disconnected`` is an `/rl`-specific extension: when ``True``, every
    other field carries stale/placeholder values and must not be treated as a real observation
    (see `env/bridge_process.py`).
    """

    tick: int
    timestamp: str
    player: PlayerState
    block_grid: BlockGrid
    held_item: HeldItem
    opponent: Opponent | None = None
    match: MatchState | None = None
    input: InputState
    disconnected: bool = False
    chat: list[str] = Field(
        default_factory=list,
        description="Chat lines observed by the bridge since the previous tick, in order "
        "(used by match-end detection; see env/reward.py and env/herbert_bridge_env.py).",
    )
