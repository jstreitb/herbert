# SPDX-License-Identifier: MIT
"""Open-vocabulary categorical encoders (hotbar item type, kit type, block type).

These fields are free-form strings emitted by the mod (Minecraft item/block
ids) with no fixed enum in the schema, so we build a vocabulary from the
*training split only* (to avoid val/test leakage) and map unseen values at
inference time to ``<UNK>``.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable

from herbert_nn.constants import (
    NULL_TOKEN,
    UNK_TOKEN,
    VOCAB_NULL_INDEX,
    VOCAB_UNK_INDEX,
)

logger = logging.getLogger(__name__)


class CategoricalVocab:
    """A frequency-capped, JSON-serializable string-to-index vocabulary.

    Index 0 is always reserved for :data:`herbert_nn.constants.NULL_TOKEN`
    (the field was ``null`` / absent) and index 1 for
    :data:`herbert_nn.constants.UNK_TOKEN` (a value not seen often enough
    during fitting, or not seen at all). Real tokens start at index 2,
    ordered by descending training-set frequency (ties broken
    alphabetically for determinism).
    """

    def __init__(self, name: str, max_size: int | None = None) -> None:
        """Initialize an empty vocabulary.

        Args:
            name: Human-readable name used in log messages (e.g. ``"item_type"``).
            max_size: Maximum number of *real* (non-special) tokens to keep.
                Tokens beyond this, ranked by training-set frequency, are
                folded into ``<UNK>``. ``None`` means unbounded.
        """
        self.name = name
        self.max_size = max_size
        self._token_to_index: dict[str, int] = {}
        self._index_to_token: dict[int, str] | None = None
        self._fitted = False

    def fit(self, values: Iterable[str | None]) -> CategoricalVocab:
        """Build the vocabulary from an iterable of (possibly ``None``) strings.

        Args:
            values: Training-split values for this field, one per tick.

        Returns:
            ``self``, for chaining.
        """
        counts: Counter[str] = Counter(v for v in values if v is not None)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if self.max_size is not None:
            ranked = ranked[: self.max_size]
        self._token_to_index = {
            token: index + 2 for index, (token, _count) in enumerate(ranked)
        }
        self._fitted = True
        logger.info(
            "Fitted vocab %r: %d unique tokens kept (out of %d observed).",
            self.name,
            len(self._token_to_index),
            len(counts),
        )
        return self

    @property
    def size(self) -> int:
        """Total vocabulary size including the two reserved special tokens."""
        return len(self._token_to_index) + 2

    def encode(self, value: str | None) -> int:
        """Map a raw value to its integer index.

        Args:
            value: The raw field value, or ``None``.

        Returns:
            ``VOCAB_NULL_INDEX`` if ``value`` is ``None``; the token's index
            if it was seen during :meth:`fit`; otherwise ``VOCAB_UNK_INDEX``.
        """
        if not self._fitted:
            raise RuntimeError(f"CategoricalVocab {self.name!r} used before fit().")
        if value is None:
            return VOCAB_NULL_INDEX
        return self._token_to_index.get(value, VOCAB_UNK_INDEX)

    def decode(self, index: int) -> str:
        """Map an integer index back to its token string (inverse of :meth:`encode`).

        Returns :data:`NULL_TOKEN` / :data:`UNK_TOKEN` for the reserved
        indices, or the original token string for any other valid index.
        """
        if index == VOCAB_NULL_INDEX:
            return NULL_TOKEN
        if index == VOCAB_UNK_INDEX:
            return UNK_TOKEN
        if self._index_to_token is None:
            self._index_to_token = {v: k for k, v in self._token_to_index.items()}
        try:
            return self._index_to_token[index]
        except KeyError as exc:
            raise KeyError(
                f"Index {index} is out of range for vocab {self.name!r} (size={self.size})."
            ) from exc

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for embedding in a cache manifest."""
        return {
            "name": self.name,
            "max_size": self.max_size,
            "token_to_index": self._token_to_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CategoricalVocab:
        """Deserialize a vocab previously produced by :meth:`to_dict`."""
        vocab = cls(name=data["name"], max_size=data.get("max_size"))
        vocab._token_to_index = {
            str(k): int(v) for k, v in data["token_to_index"].items()
        }
        vocab._fitted = True
        return vocab

    def __len__(self) -> int:
        """Return :attr:`size`."""
        return self.size

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        """Return a short debug representation."""
        return f"CategoricalVocab(name={self.name!r}, size={self.size})"


__all__ = [
    "CategoricalVocab",
    "NULL_TOKEN",
    "UNK_TOKEN",
    "VOCAB_NULL_INDEX",
    "VOCAB_UNK_INDEX",
]
