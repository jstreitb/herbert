# SPDX-License-Identifier: MIT
"""Tests for schema parsing, version dispatch, and error handling."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from factories import make_header, make_record, write_session_file
from herbert_nn.schemas.registry import (
    SchemaVersionError,
    SessionParseError,
    get_models_for_version,
    load_session,
    parse_header_line,
    parse_record_line,
)
from herbert_nn.schemas.v1_0_0 import SessionHeaderV1, TickRecordV1
from herbert_nn.schemas.v1_2_0 import SessionHeaderV1_2_0


def test_get_models_for_known_version() -> None:
    models = get_models_for_version("1.0.0")
    assert models.header_model is SessionHeaderV1
    assert models.record_model is TickRecordV1


def test_get_models_for_1_2_0_reuses_1_0_0_record_model() -> None:
    models = get_models_for_version("1.2.0")
    assert models.header_model is SessionHeaderV1_2_0
    assert models.record_model is TickRecordV1


def test_1_2_0_header_parses_without_optional_fields() -> None:
    header = parse_header_line(json.dumps(make_header(schema_version="1.2.0")))
    assert isinstance(header, SessionHeaderV1_2_0)
    assert header.player_username_display is None
    assert header.chunk_index is None
    assert header.chunk_total is None


def test_1_2_0_header_parses_with_chunk_fields() -> None:
    header = parse_header_line(
        json.dumps(make_header(schema_version="1.2.0", chunk_index=0, chunk_total=2))
    )
    assert header.chunk_index == 0
    assert header.chunk_total == 2


def test_1_2_0_header_rejects_chunk_total_below_2() -> None:
    with pytest.raises(ValidationError):
        SessionHeaderV1_2_0(
            **make_header(schema_version="1.2.0", chunk_index=0, chunk_total=1)
        )


def test_1_0_0_tick_record_parses_under_1_2_0_dispatch() -> None:
    # The 1.2.0 record model is literally TickRecordV1 -- ticks recorded under either
    # version must parse identically.
    record = parse_record_line(json.dumps(make_record(0)), "1.2.0")
    assert isinstance(record, TickRecordV1)


def test_get_models_for_unknown_version_raises_clear_error() -> None:
    with pytest.raises(SchemaVersionError, match="Unsupported schema_version"):
        get_models_for_version("9.9.9")


@pytest.mark.parametrize(
    "record_overrides",
    [
        {"tick": 0},
        {"tick": 12345},
    ],
)
def test_valid_record_parses(record_overrides: dict) -> None:
    header = parse_header_line(json.dumps(make_header()))
    record_dict = make_record(**record_overrides)
    record = parse_record_line(json.dumps(record_dict), header.schema_version)
    assert isinstance(record, TickRecordV1)
    assert record.tick == record_overrides["tick"]


@pytest.mark.parametrize(
    "bad_record",
    [
        # Missing a required field entirely.
        {k: v for k, v in make_record(0).items() if k != "player"},
        # Invalid enum value for block_grid cell type.
        {
            **make_record(0),
            "block_grid": {
                **make_record(0)["block_grid"],
                "cells": ["NOT_A_REAL_CELL_TYPE"] * 8,
            },
        },
        # Wrong type: forward must be an int in [-1, 1].
        {**make_record(0), "input": {**make_record(0)["input"], "forward": 5}},
    ],
)
def test_malformed_record_raises_validation_error(bad_record: dict) -> None:
    with pytest.raises(ValidationError):
        parse_record_line(json.dumps(bad_record), "1.0.0")


def test_load_session_end_to_end(tmp_path) -> None:
    path = write_session_file(tmp_path / "s.jsonl", "sess-1", num_ticks=5)
    header, records = load_session(path)
    assert header.session_id == "sess-1"
    assert len(records) == 5
    assert [r.tick for r in records] == [0, 1, 2, 3, 4]


def test_load_session_malformed_line_raises_with_line_context(tmp_path) -> None:
    header_line = json.dumps(make_header(session_id="sess-2"))
    good_line = json.dumps(make_record(0))
    bad_line = json.dumps({**make_record(1), "player": None})
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join([header_line, good_line, bad_line]) + "\n")

    with pytest.raises(SessionParseError, match="line 3") as exc_info:
        load_session(path)
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_empty_file_raises_clear_error(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(SessionParseError, match="empty"):
        load_session(path)
