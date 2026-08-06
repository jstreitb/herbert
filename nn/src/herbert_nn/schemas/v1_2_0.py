# SPDX-License-Identifier: MIT
"""BridgeLogger JSONL schema, header version ``"1.2.0"``.

Schema version ``1.2.0`` only changed the *header* line relative to ``1.0.0``: it added two
optional fields, ``player_username_display`` (introduced in the ``1.1.0`` bump, which was
never independently shipped as a live ``schema_version`` -- ``/mod``'s history goes straight
from ``1.0.0`` to ``1.2.0``, so no ``v1_1_0`` module exists here) and
``chunk_index``/``chunk_total`` (for chunked uploads). See ``mod/README.md``'s "JSONL schema"
and "Chunked uploads" sections for the authoritative field-by-field documentation.

The per-tick record schema is byte-for-byte unchanged since ``1.0.0`` -- verified against real
``1.2.0`` session data, every field identical. :mod:`herbert_nn.schemas.registry` therefore
registers :class:`herbert_nn.schemas.v1_0_0.TickRecordV1` directly as this version's record
model rather than duplicating an identical class here; only the header gets a new,
self-contained model, per this package's usual one-module-per-version convention.

This module must never be mutated once released, same as every other versioned schema module
-- schema evolution happens by adding a new version module.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: The exact ``schema_version`` string this module implements.
SCHEMA_VERSION = "1.2.0"


class SessionHeaderV1_2_0(BaseModel):
    """First line of a ``.jsonl`` session log for schema version ``1.2.0``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(
        ..., description="Semver string identifying the record schema used below."
    )
    herbert_mod_version: str = Field(
        ..., description="Version string of the /mod build."
    )
    session_id: str = Field(
        ..., description="UUID4 string identifying this recording session."
    )
    recording_start_timestamp: str = Field(
        ..., description="ISO-8601 timestamp of when recording started."
    )
    player_username_hash: str = Field(
        ..., description="SHA-256 hex digest of the recording player's username."
    )
    player_username_display: str | None = Field(
        default=None,
        description=(
            "The player's raw username, present only if they opted in to displaying it "
            "publicly. Omitted entirely (never present as null) otherwise. Added in 1.1.0."
        ),
    )
    chunk_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "0-based index of this chunk among chunk_total chunks. Present only on a chunk "
            "file produced by SessionChunker; absent on a non-chunked session. Added in 1.2.0."
        ),
    )
    chunk_total: int | None = Field(
        default=None,
        ge=2,
        description=(
            "Total number of chunks the session was split into. Present under the same rule "
            "as chunk_index; always >= 2 when present. Added in 1.2.0."
        ),
    )
