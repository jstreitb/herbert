"""BridgeLogger JSONL schema, version ``"1.0.0"``.

This module is the single source of truth for the on-disk shape of schema
version ``1.0.0`` session logs produced by the ``/mod`` component. It must
never be mutated once released (old recorded sessions must remain parseable
forever) -- schema evolution happens by adding a *new* versioned module (see
:mod:`herbert_nn.schemas`) and registering it in
:mod:`herbert_nn.schemas.registry`.

A session ``.jsonl`` file has the shape::

    <SessionHeaderV1 as JSON>\\n
    <TickRecordV1 as JSON>\\n
    <TickRecordV1 as JSON>\\n
    ...

All models use Pydantic v2 with ``extra="forbid"`` so that unexpected fields
(e.g. from a future, incompatible mod build that failed to bump
``schema_version``) fail loudly at parse time rather than being silently
dropped.

Field names throughout this module are kept identical to the JSON keys
documented in ``/mod/README.md`` (section "JSONL schema (contract for
``/nn``)"), rather than being renamed for Python-side readability, so that
this file is directly diffable against the mod's authoritative contract.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

#: The exact ``schema_version`` string this module implements.
SCHEMA_VERSION = "1.0.0"


class _StrictModel(BaseModel):
    """Base model shared by all v1.0.0 schema objects.

    Forbids unknown fields so malformed or unexpectedly-shaped records raise
    a clear :class:`pydantic.ValidationError` instead of silently ignoring
    data the mod actually emitted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class BlockCellType(str, Enum):
    """Enum of block-grid cell classifications emitted by the mod."""

    AIR = "AIR"
    SOLID_BRIDGEABLE = "SOLID_BRIDGEABLE"
    LIQUID = "LIQUID"
    VOID = "VOID"
    OTHER_SOLID = "OTHER_SOLID"


class HeldItemCategory(str, Enum):
    """Enum of coarse held-item categories used for the opponent's hand."""

    SWORD = "SWORD"
    BOW = "BOW"
    BLOCKS = "BLOCKS"
    OTHER = "OTHER"


class SessionHeaderV1(_StrictModel):
    """First line of a ``.jsonl`` session log: session-level metadata."""

    schema_version: str = Field(
        ..., description="Semver string identifying the record schema used below."
    )
    herbert_mod_version: str = Field(..., description="Version string of the /mod build.")
    session_id: str = Field(..., description="UUID4 string identifying this recording session.")
    recording_start_timestamp: str = Field(
        ..., description="ISO-8601 timestamp of when recording started."
    )
    player_username_hash: str = Field(
        ..., description="SHA-256 hex digest of the recording player's username."
    )


class PlayerState(_StrictModel):
    """Player kinematic and vital state at a single tick."""

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
    """Flattened local block grid around the player at a single tick.

    ``cells`` is a row-major flattened array of length
    ``width * height * depth`` following ``index = (yIndex * depth + zIndex)
    * width + xIndex``; validated by
    :meth:`herbert_nn.schemas.v1_0_0.BlockGrid.cell_count` helper-free here
    (validation of the length-vs-dimensions invariant is intentionally left
    to the preprocessing layer, since a mismatch there is a data-integrity
    issue best surfaced with rich context such as the session id and tick
    number, which this model does not have access to).
    """

    width: int
    height: int
    depth: int
    origin: str
    cells: list[BlockCellType]


class HeldItem(_StrictModel):
    """Player's currently-selected hotbar slot at a single tick."""

    hotbar_slot: int = Field(..., ge=0, le=8)
    item_id: str | None = None
    count: int


class Opponent(_StrictModel):
    """Best-effort detected opponent state, relative to the player.

    ``None`` at the :class:`TickRecordV1` level whenever the mod could not
    detect an opponent for that tick.
    """

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
    """Best-effort detected match/scoreboard context.

    ``None`` at the :class:`TickRecordV1` level whenever nothing at all was
    parseable that tick; individual fields are independently nullable since
    each is detected by its own best-effort heuristic.
    """

    own_score: int | None = None
    opponent_score: int | None = None
    elapsed_seconds: int | None = None
    kit: str | None = None


class InputState(_StrictModel):
    """Recorded player input/action for a single tick (the imitation target)."""

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


class TickRecordV1(_StrictModel):
    """A single per-tick record: full game state + recorded player input."""

    tick: int
    timestamp: str
    player: PlayerState
    block_grid: BlockGrid
    held_item: HeldItem
    opponent: Opponent | None = None
    match: MatchState | None = None
    input: InputState
