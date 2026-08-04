"""Synthetic BridgeLogger session builders shared by tests and conftest fixtures.

Kept separate from ``conftest.py`` so test modules can import these factory
functions directly (``from factories import make_record``) without relying
on cross-package imports of ``conftest`` itself.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

#: A tiny 2x2x2 = 8-cell block grid, used everywhere in tests to keep
#: tensors small and fast.
TEST_BLOCK_GRID_SHAPE = (2, 2, 2)
TEST_NUM_CELLS = TEST_BLOCK_GRID_SHAPE[0] * TEST_BLOCK_GRID_SHAPE[1] * TEST_BLOCK_GRID_SHAPE[2]


def make_header(session_id: str | None = None, **overrides: Any) -> dict[str, Any]:
    """Build a valid v1.0.0 session header dict, with any field overridable."""
    header = {
        "schema_version": "1.0.0",
        "herbert_mod_version": "1.0.0",
        "session_id": session_id or str(uuid.uuid4()),
        "recording_start_timestamp": "2026-08-04T12:00:00Z",
        "player_username_hash": "a" * 64,
    }
    header.update(overrides)
    return header


def make_record(
    tick: int, place_active: bool = False, with_opponent: bool = True, **overrides: Any
) -> dict[str, Any]:
    """Build a valid v1.0.0 tick-record dict, with any top-level field overridable."""
    record: dict[str, Any] = {
        "tick": tick,
        "timestamp": f"2026-08-04T12:00:{tick % 60:02d}Z",
        "player": {
            "x": float(tick) * 0.1,
            "y": 64.0,
            "z": 0.0,
            "vx": 0.1,
            "vy": 0.0,
            "vz": 0.0,
            "yaw": float(tick % 360),
            "pitch": 0.0,
            "on_ground": True,
            "sneaking": False,
            "health": 20.0,
            "food": 20,
        },
        "block_grid": {
            "width": TEST_BLOCK_GRID_SHAPE[0],
            "height": TEST_BLOCK_GRID_SHAPE[1],
            "depth": TEST_BLOCK_GRID_SHAPE[2],
            "origin": "player_feet_centered",
            "cells": ["AIR"] * (TEST_NUM_CELLS - 1) + ["SOLID_BRIDGEABLE"],
        },
        "held_item": {"hotbar_slot": tick % 9, "item_id": "minecraft:wool", "count": 64},
        "opponent": (
            {
                "rel_x": 1.0,
                "rel_y": 0.0,
                "rel_z": 1.0,
                "rel_vx": 0.0,
                "rel_vy": 0.0,
                "rel_vz": 0.0,
                "yaw": 90.0,
                "pitch": 0.0,
                "health": 20.0,
                "held_item_category": "SWORD",
            }
            if with_opponent
            else None
        ),
        "match": {
            "own_score": 0,
            "opponent_score": 0,
            "elapsed_seconds": tick // 20,
            "kit": "default",
        },
        "input": {
            "forward": 1,
            "strafe": 0,
            "jump": tick % 5 == 0,
            "sneak": False,
            "delta_yaw": 0.5,
            "delta_pitch": -0.1,
            "attack_occurred": tick % 7 == 0,
            "attack_target_type": "PLAYER" if tick % 7 == 0 else None,
            "place_occurred": place_active,
            "place_block_type": "minecraft:wool" if place_active else None,
            "place_x": 1 if place_active else None,
            "place_y": 64 if place_active else None,
            "place_z": 1 if place_active else None,
        },
    }
    record.update(overrides)
    return record


def write_session_file(
    path: Path, session_id: str, num_ticks: int, place_every: int | None = None
) -> Path:
    """Write a full session .jsonl file (header + num_ticks records) to ``path``."""
    lines = [json.dumps(make_header(session_id=session_id))]
    for t in range(num_ticks):
        place_active = place_every is not None and t % place_every == 0 and t > 0
        lines.append(json.dumps(make_record(t, place_active=place_active)))
    path.write_text("\n".join(lines) + "\n")
    return path
