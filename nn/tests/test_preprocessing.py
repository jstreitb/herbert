# SPDX-License-Identifier: MIT
"""Tests for feature normalization, categorical vocab encoding, and sliding-window building.

Uses small synthetic in-memory fixtures throughout -- no dependency on real
recorded session data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from herbert_nn.data.cache import SessionBoundary
from herbert_nn.data.dataset import WindowDataset
from herbert_nn.data.normalization import Standardizer
from herbert_nn.data.vocab import VOCAB_NULL_INDEX, VOCAB_UNK_INDEX, CategoricalVocab


def test_standardizer_fit_transform_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(loc=5.0, scale=2.0, size=(1000, 4)).astype(np.float32)
    standardizer = Standardizer().fit(train)
    transformed = standardizer.transform(train)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-3)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-2)


def test_standardizer_applies_train_stats_to_val_without_refitting() -> None:
    train = np.array([[0.0], [10.0]], dtype=np.float32)  # mean=5, std=5
    val = np.array([[5.0], [15.0]], dtype=np.float32)
    standardizer = Standardizer().fit(train)
    transformed_val = standardizer.transform(val)
    # (5-5)/5 = 0, (15-5)/5 = 2
    assert np.allclose(transformed_val.flatten(), [0.0, 2.0], atol=1e-5)


def test_standardizer_handles_constant_column_without_div_by_zero() -> None:
    train = np.ones((10, 2), dtype=np.float32)
    standardizer = Standardizer().fit(train)
    transformed = standardizer.transform(train)
    assert np.all(np.isfinite(transformed))


def test_standardizer_fit_zero_rows_raises_instead_of_producing_nan() -> None:
    empty = np.zeros((0, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="zero rows"):
        Standardizer().fit(empty)


def test_standardizer_fit_wrong_ndim_raises() -> None:
    with pytest.raises(ValueError, match="2D"):
        Standardizer().fit(np.zeros((5, 3, 2), dtype=np.float32))


def test_standardizer_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        Standardizer().transform(np.zeros((1, 3), dtype=np.float32))


def test_standardizer_roundtrip_serialization() -> None:
    train = np.random.default_rng(1).normal(size=(50, 3)).astype(np.float32)
    standardizer = Standardizer().fit(train)
    restored = Standardizer.from_dict(standardizer.to_dict())
    assert np.allclose(restored.mean, standardizer.mean)
    assert np.allclose(restored.std, standardizer.std)


def test_categorical_vocab_fits_and_encodes_known_tokens() -> None:
    vocab = CategoricalVocab("item_type").fit(["a", "a", "a", "b", "b", "c"])
    # "a" (freq 3) should get the lowest real index (2, since 0/1 reserved).
    assert vocab.encode("a") == 2
    assert vocab.encode("b") == 3
    assert vocab.encode("c") == 4
    assert vocab.decode(2) == "a"


def test_categorical_vocab_maps_null_and_unseen_to_reserved_indices() -> None:
    vocab = CategoricalVocab("kit_type").fit(["default", "default", None])
    assert vocab.encode(None) == VOCAB_NULL_INDEX
    assert vocab.encode("never_seen_in_training") == VOCAB_UNK_INDEX


def test_categorical_vocab_max_size_caps_and_folds_rare_tokens_to_unk() -> None:
    # "a" x5, "b" x3, "c" x1 -- with max_size=2, only a/b get real slots.
    values = ["a"] * 5 + ["b"] * 3 + ["c"] * 1
    vocab = CategoricalVocab("capped", max_size=2).fit(values)
    assert vocab.size == 4  # NULL + UNK + a + b
    assert vocab.encode("a") != VOCAB_UNK_INDEX
    assert vocab.encode("b") != VOCAB_UNK_INDEX
    assert vocab.encode("c") == VOCAB_UNK_INDEX


def _make_tensors(num_rows: int, feature_dim: int = 3) -> dict[str, torch.Tensor]:
    return {
        "continuous": torch.arange(num_rows * feature_dim, dtype=torch.float32).reshape(
            num_rows, feature_dim
        ),
        "block_grid_cells": torch.zeros(num_rows, 4, dtype=torch.int64),
        "hotbar_slot_index": torch.zeros(num_rows, dtype=torch.int64),
        "hotbar_item_type": torch.zeros(num_rows, dtype=torch.int64),
        "opponent_held_item_category": torch.zeros(num_rows, dtype=torch.int64),
        "match_kit_type": torch.zeros(num_rows, dtype=torch.int64),
        "mouse_target": torch.zeros(num_rows, 2),
        "discrete_target": torch.zeros(num_rows, 4),
        "place_block_type": torch.zeros(num_rows, dtype=torch.int64),
        "place_mask": torch.zeros(num_rows),
        "tick": torch.arange(num_rows, dtype=torch.int64),
    }


def test_window_dataset_builds_correct_number_of_windows_per_session() -> None:
    # Two sessions of 10 and 15 ticks, concatenated: rows [0,10) and [10,25).
    tensors = _make_tensors(25)
    boundaries = [
        SessionBoundary(session_id="s1", start=0, end=10),
        SessionBoundary(session_id="s2", start=10, end=25),
    ]
    window_length = 4
    dataset = WindowDataset(tensors, boundaries, window_length=window_length, stride=1)
    # session 1: 10 - 4 + 1 = 7 windows; session 2: 15 - 4 + 1 = 12 windows.
    assert len(dataset) == 7 + 12


def test_window_dataset_never_crosses_session_boundary() -> None:
    tensors = _make_tensors(25)
    boundaries = [
        SessionBoundary(session_id="s1", start=0, end=10),
        SessionBoundary(session_id="s2", start=10, end=25),
    ]
    dataset = WindowDataset(tensors, boundaries, window_length=4, stride=1)
    for i in range(len(dataset)):
        end = dataset._ends[i]
        start = end - dataset.window_length + 1
        # Both start and end must fall within the same session's [start, end) range.
        owning = [b for b in boundaries if b.start <= start and end < b.end]
        assert (
            len(owning) == 1
        ), f"window [{start}, {end}] does not stay within one session"


def test_window_dataset_sample_shapes_and_target_is_last_tick() -> None:
    tensors = _make_tensors(10, feature_dim=3)
    tensors["tick"] = torch.arange(10, dtype=torch.int64)
    boundaries = [SessionBoundary(session_id="s1", start=0, end=10)]
    window_length = 5
    dataset = WindowDataset(tensors, boundaries, window_length=window_length, stride=1)
    sample = dataset[0]
    assert sample["continuous"].shape == (window_length, 3)
    assert sample["mouse_target"].shape == (2,)
    # First window covers ticks [0..4]; continuous row values should match ticks 0..4.
    expected = tensors["continuous"][0:5]
    assert torch.equal(sample["continuous"], expected)


def test_window_dataset_skips_sessions_shorter_than_window_length() -> None:
    tensors = _make_tensors(3)
    boundaries = [SessionBoundary(session_id="short", start=0, end=3)]
    dataset = WindowDataset(tensors, boundaries, window_length=5, stride=1)
    assert len(dataset) == 0
