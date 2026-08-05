# SPDX-License-Identifier: MIT
"""Tests for `herbert_rl.vocab.CategoricalVocab` and `herbert_rl.normalization.Standardizer`.

Both classes are hand-synced copies of their `/nn` counterparts (see each module's
docstring for why); these tests cover the same behavior directly against the `/rl` copies
since `/rl` may not import `herbert_nn` at runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

from herbert_rl.normalization import Standardizer
from herbert_rl.vocab import VOCAB_NULL_INDEX, VOCAB_UNK_INDEX, CategoricalVocab


def test_vocab_fit_and_encode_known_tokens() -> None:
    vocab = CategoricalVocab("item_type").fit(["a", "b", "a", "a", "b", "c"])
    assert vocab.encode("a") != vocab.encode("b") != vocab.encode("c")
    assert vocab.encode("a") not in (VOCAB_NULL_INDEX, VOCAB_UNK_INDEX)


def test_vocab_encode_maps_null_and_unseen_to_reserved_indices() -> None:
    vocab = CategoricalVocab("item_type").fit(["a", "b"])
    assert vocab.encode(None) == VOCAB_NULL_INDEX
    assert vocab.encode("never-seen") == VOCAB_UNK_INDEX


def test_vocab_max_size_folds_rare_tokens_to_unk() -> None:
    vocab = CategoricalVocab("item_type", max_size=1).fit(
        ["common"] * 10 + ["rare"] * 2
    )
    assert vocab.encode("common") == vocab.encode("common")
    assert vocab.encode("rare") == VOCAB_UNK_INDEX
    assert vocab.size == 3  # NULL + UNK + 1 real token


def test_vocab_encode_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        CategoricalVocab("item_type").encode("a")


def test_vocab_decode_roundtrip() -> None:
    vocab = CategoricalVocab("item_type").fit(["a", "b"])
    index = vocab.encode("a")
    assert vocab.decode(index) == "a"


def test_vocab_decode_out_of_range_raises() -> None:
    vocab = CategoricalVocab("item_type").fit(["a"])
    with pytest.raises(KeyError):
        vocab.decode(999)


def test_vocab_to_dict_from_dict_roundtrip() -> None:
    vocab = CategoricalVocab("item_type", max_size=5).fit(["a", "b", "c"])
    restored = CategoricalVocab.from_dict(vocab.to_dict())
    assert restored.size == vocab.size
    assert restored.encode("a") == vocab.encode("a")


def test_vocab_empty_has_only_special_tokens() -> None:
    vocab = CategoricalVocab.empty("item_type")
    assert vocab.size == 2
    assert vocab.encode("anything") == VOCAB_UNK_INDEX


def test_vocab_len_matches_size() -> None:
    vocab = CategoricalVocab("item_type").fit(["a", "b"])
    assert len(vocab) == vocab.size


def test_standardizer_fit_transform_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(loc=5.0, scale=2.0, size=(1000, 4)).astype(np.float32)
    standardizer = Standardizer().fit(train)
    transformed = standardizer.transform(train)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-3)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-2)


def test_standardizer_constant_column_does_not_divide_by_zero() -> None:
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


def test_standardizer_to_dict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="unfitted"):
        Standardizer().to_dict()


def test_standardizer_to_dict_from_dict_roundtrip() -> None:
    train = np.random.default_rng(1).normal(size=(50, 3)).astype(np.float32)
    standardizer = Standardizer().fit(train)
    restored = Standardizer.from_dict(standardizer.to_dict())
    assert np.allclose(restored.mean, standardizer.mean)
    assert np.allclose(restored.std, standardizer.std)


def test_standardizer_identity_is_a_no_op() -> None:
    standardizer = Standardizer.identity(num_features=4)
    features = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    assert np.allclose(standardizer.transform(features), features)
