# SPDX-License-Identifier: MIT
"""Shared pytest fixtures, built on top of the synthetic factories in ``factories.py``.

None of these tests depend on real recorded session data -- every fixture
builds small, in-memory-shaped session files from scratch so the test suite
is fully self-contained and fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from factories import TEST_BLOCK_GRID_SHAPE, write_session_file


@pytest.fixture
def block_grid_shape() -> tuple[int, int, int]:
    return TEST_BLOCK_GRID_SHAPE


@pytest.fixture
def raw_session_dir(tmp_path: Path) -> Path:
    """Build a directory with 6 small synthetic sessions (for split / cache tests)."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for i in range(6):
        session_id = f"session-{i:02d}"
        write_session_file(
            raw_dir / f"{session_id}.jsonl", session_id, num_ticks=60, place_every=10
        )
    return raw_dir
