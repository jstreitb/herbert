# SPDX-License-Identifier: MIT
"""End-to-end preprocessing cache tests: raw JSONL -> normalized, cached tensors.

Exercises herbert_nn.data.cache against the synthetic ``raw_session_dir``
fixture (6 sessions x 60 ticks each; see tests/conftest.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from factories import TEST_BLOCK_GRID_SHAPE, make_record
from herbert_nn.data.cache import build_or_load_cache
from herbert_nn.data.config import PreprocessConfig
from herbert_nn.data.dataset import build_dataset
from herbert_nn.data.features import encode_session_raw
from herbert_nn.schemas.registry import SchemaVersionError
from herbert_nn.schemas.v1_0_0 import TickRecordV1


def _base_config(raw_dir: Path, cache_dir: Path, **overrides) -> PreprocessConfig:
    kwargs = {
        "raw_dir": str(raw_dir),
        "cache_dir": str(cache_dir),
        "window_length": 8,
        "window_stride": 1,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "split_seed": 0,
        "item_type_vocab_size": 16,
        "kit_type_vocab_size": 8,
        "place_block_type_vocab_size": 8,
    }
    kwargs.update(overrides)
    return PreprocessConfig(**kwargs)


def test_build_cache_produces_all_splits_with_no_session_overlap(
    raw_session_dir: Path, tmp_path: Path
) -> None:
    config = _base_config(raw_session_dir, tmp_path / "cache")
    bundle = build_or_load_cache(config)

    train_ids = set(bundle.manifest.split_session_ids["train"])
    val_ids = set(bundle.manifest.split_session_ids["val"])
    test_ids = set(bundle.manifest.split_session_ids["test"])
    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()
    assert train_ids | val_ids | test_ids == {f"session-{i:02d}" for i in range(6)}

    for split in ("train", "val", "test"):
        tensors = bundle.load_split(split)
        assert tensors["continuous"].shape[0] == bundle.manifest.counts[split]
        assert tensors["continuous"].shape[1] > 0


def test_cache_is_reused_on_identical_rerun(
    raw_session_dir: Path, tmp_path: Path
) -> None:
    config = _base_config(raw_session_dir, tmp_path / "cache")
    bundle1 = build_or_load_cache(config)
    bundle2 = build_or_load_cache(config)
    assert bundle1.cache_path == bundle2.cache_path
    assert bundle1.manifest.config_hash == bundle2.manifest.config_hash


def test_cache_invalidated_by_config_change(
    raw_session_dir: Path, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    config_a = _base_config(raw_session_dir, cache_dir, window_length=8)
    config_b = _base_config(raw_session_dir, cache_dir, window_length=16)
    bundle_a = build_or_load_cache(config_a)
    bundle_b = build_or_load_cache(config_b)
    assert bundle_a.cache_path != bundle_b.cache_path


def test_build_cache_empty_raw_dir_raises_file_not_found(tmp_path: Path) -> None:
    empty_raw_dir = tmp_path / "empty_raw"
    empty_raw_dir.mkdir()
    config = _base_config(empty_raw_dir, tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="No \\*.jsonl session files"):
        build_or_load_cache(config)


def test_build_cache_empty_first_raw_file_raises_clear_error(
    raw_session_dir: Path, tmp_path: Path
) -> None:
    # An empty raw file sorts before every real "session-*.jsonl" file, so it's the one
    # the cheap schema-version pre-pass reads first.
    (raw_session_dir / "0_empty.jsonl").write_text("")
    config = _base_config(raw_session_dir, tmp_path / "cache")
    with pytest.raises(SchemaVersionError, match="file is empty"):
        build_or_load_cache(config)


def test_build_cache_malformed_first_raw_file_raises_clear_error(
    raw_session_dir: Path, tmp_path: Path
) -> None:
    (raw_session_dir / "0_malformed.jsonl").write_text("this is not json at all\n")
    config = _base_config(raw_session_dir, tmp_path / "cache")
    with pytest.raises(SchemaVersionError, match="not valid JSON"):
        build_or_load_cache(config)


def test_normalization_stats_come_from_train_split_only(
    raw_session_dir: Path, tmp_path: Path
) -> None:
    import numpy as np

    config = _base_config(raw_session_dir, tmp_path / "cache")
    bundle = build_or_load_cache(config)
    train_tensors = bundle.load_split("train")
    # The standardized training continuous features should have ~zero mean.
    train_continuous = train_tensors["continuous"].numpy()
    assert np.allclose(train_continuous.mean(axis=0), 0.0, atol=0.2)


def test_window_and_tick_datasets_buildable_from_cache(
    raw_session_dir: Path, tmp_path: Path
) -> None:
    config = _base_config(raw_session_dir, tmp_path / "cache", window_length=8)
    bundle = build_or_load_cache(config)

    train_tensors = bundle.load_split("train")
    boundaries = bundle.manifest.split_boundaries["train"]

    tick_dataset = build_dataset(train_tensors, boundaries, "mlp")
    assert len(tick_dataset) == train_tensors["continuous"].shape[0]
    sample = tick_dataset[0]
    assert sample["continuous"].shape[0] == train_tensors["continuous"].shape[1]

    window_dataset = build_dataset(
        train_tensors, boundaries, "gru", window_length=8, window_stride=1
    )
    assert len(window_dataset) > 0
    window_sample = window_dataset[0]
    assert window_sample["continuous"].shape[0] == 8


def test_cache_invalidated_by_feature_schema_version_bump(
    raw_session_dir: Path, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    config_a = _base_config(raw_session_dir, cache_dir, feature_schema_version=2)
    config_b = _base_config(raw_session_dir, cache_dir, feature_schema_version=3)
    bundle_a = build_or_load_cache(config_a)
    bundle_b = build_or_load_cache(config_b)
    assert bundle_a.cache_path != bundle_b.cache_path


def test_movement_target_preserves_raw_ternary_boundary_values() -> None:
    record_dict = make_record(0, place_active=False)
    record_dict["input"]["forward"] = -1
    record_dict["input"]["strafe"] = 1
    record = TickRecordV1(**record_dict)

    arrays = encode_session_raw([record], "sid", TEST_BLOCK_GRID_SHAPE)

    # Raw floats, not remapped class indices -- see MovementHead's design rationale.
    assert arrays.movement_target[0].tolist() == [-1.0, 1.0]
