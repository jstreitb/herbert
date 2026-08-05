# SPDX-License-Identifier: MIT
"""Versioned Pydantic schema definitions for BridgeLogger JSONL session logs.

The ``/mod`` component (a Forge 1.8.9 Minecraft mod) logs each recorded
Bridge duel session as a ``.jsonl`` file: the first line is a *session
header* object, and every subsequent line is a *tick record* describing the
game state and the player's input for a single game tick.

The on-disk format is versioned via the ``schema_version`` field in the
session header (a semver string, e.g. ``"1.0.0"``). This subpackage exposes:

* One module per schema version (e.g. :mod:`herbert_nn.schemas.v1_0_0`)
  containing the Pydantic models for that exact version of the contract.
* A :mod:`herbert_nn.schemas.registry` module that maps a ``schema_version``
  string to the (header model, record model) pair for that version, and
  provides version-dispatching helpers to parse a session file without the
  caller needing to know in advance which schema version it uses.

Adding a new schema version (e.g. because the mod bumps to ``"1.1.0"`` or
``"2.0.0"``) means: (1) add a new ``v1_1_0.py`` module with its own Pydantic
models (copy-and-modify the previous version's module -- do NOT mutate old
version modules, they must remain byte-for-byte able to parse old files),
and (2) register it in :data:`herbert_nn.schemas.registry.SCHEMA_REGISTRY`.
No other code needs to change; all downstream consumers dispatch through the
registry.
"""

from __future__ import annotations

from herbert_nn.schemas.registry import (
    SCHEMA_REGISTRY,
    SchemaVersionError,
    SessionParseError,
    get_models_for_version,
    load_session,
    parse_header_line,
    parse_record_line,
)
from herbert_nn.schemas.v1_0_0 import (
    BlockCellType,
    BlockGrid,
    HeldItem,
    HeldItemCategory,
    InputState,
    MatchState,
    Opponent,
    PlayerState,
    SessionHeaderV1,
    TickRecordV1,
)

__all__ = [
    "SCHEMA_REGISTRY",
    "SchemaVersionError",
    "SessionParseError",
    "get_models_for_version",
    "load_session",
    "parse_header_line",
    "parse_record_line",
    "BlockCellType",
    "BlockGrid",
    "HeldItem",
    "HeldItemCategory",
    "InputState",
    "MatchState",
    "Opponent",
    "PlayerState",
    "SessionHeaderV1",
    "TickRecordV1",
]
