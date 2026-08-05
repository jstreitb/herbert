# SPDX-License-Identifier: MIT
"""Tests for session-level train/val/test splitting."""

from __future__ import annotations

import pytest

from herbert_nn.data.split import split_sessions


def test_no_session_overlap_between_splits() -> None:
    session_ids = [f"s{i}" for i in range(30)]
    split = split_sessions(
        session_ids, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=123
    )

    train_set, val_set, test_set = set(split.train), set(split.val), set(split.test)
    assert train_set & val_set == set()
    assert train_set & test_set == set()
    assert val_set & test_set == set()
    assert train_set | val_set | test_set == set(session_ids)


def test_split_ratios_approximately_correct() -> None:
    session_ids = [f"s{i}" for i in range(100)]
    split = split_sessions(
        session_ids, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1
    )
    assert len(split.train) == 80
    assert len(split.val) == 10
    assert len(split.test) == 10


def test_split_is_deterministic_given_seed() -> None:
    session_ids = [f"s{i}" for i in range(40)]
    split_a = split_sessions(session_ids, seed=7)
    split_b = split_sessions(session_ids, seed=7)
    assert split_a.train == split_b.train
    assert split_a.val == split_b.val
    assert split_a.test == split_b.test


def test_different_seeds_generally_differ() -> None:
    session_ids = [f"s{i}" for i in range(40)]
    split_a = split_sessions(session_ids, seed=1)
    split_b = split_sessions(session_ids, seed=2)
    assert split_a.train != split_b.train


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_sessions(["a", "b"], train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


def test_negative_ratio_that_still_sums_to_one_is_rejected() -> None:
    # A negative ratio (offset by an over-1.0 ratio elsewhere) would otherwise pass the
    # sum-to-1.0 check while producing a nonsensical negative split size.
    with pytest.raises(ValueError, match=r"train_ratio must be within \[0.0, 1.0\]"):
        split_sessions(["a", "b"], train_ratio=1.5, val_ratio=-0.5, test_ratio=0.0)


def test_ratio_above_one_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"train_ratio must be within \[0.0, 1.0\]"):
        split_sessions(["a", "b"], train_ratio=1.1, val_ratio=-0.1, test_ratio=0.0)


def test_empty_session_list_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        split_sessions([])


def test_too_few_sessions_for_nonzero_ratio_raises() -> None:
    # A single session can't populate all three splits when every ratio is > 0.
    with pytest.raises(ValueError, match="val split has ratio"):
        split_sessions(["only-one"], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)


def test_zero_ratio_split_produces_empty_list_not_error() -> None:
    session_ids = [f"s{i}" for i in range(10)]
    split = split_sessions(
        session_ids, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0, seed=1
    )
    assert split.test == []
    assert len(split.train) + len(split.val) == 10


def test_duplicate_session_ids_are_deduplicated() -> None:
    session_ids = ["a", "b", "c", "a", "b"]
    split = split_sessions(
        session_ids, train_ratio=0.34, val_ratio=0.33, test_ratio=0.33, seed=1
    )
    assert len(split.train) + len(split.val) + len(split.test) == 3


def test_split_of_lookup() -> None:
    session_ids = [f"s{i}" for i in range(10)]
    split = split_sessions(
        session_ids, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1
    )
    for sid in split.train:
        assert split.split_of(sid) == "train"
    for sid in split.val:
        assert split.split_of(sid) == "val"
    for sid in split.test:
        assert split.split_of(sid) == "test"
    with pytest.raises(KeyError):
        split.split_of("not-a-real-session")
