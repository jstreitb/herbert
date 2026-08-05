"""Open-vocabulary categorical encoder -- a hand-synced copy of `herbert_nn.data.vocab`.

See `constants.py` for why this is a copy rather than an import. `to_dict`/`from_dict` are
identical to `/nn`'s format so an `item_type_vocab` / `kit_type_vocab` block inside an `/nn`
preprocessing cache's `manifest.json` loads here unmodified (see `nn_cache.py`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from herbert_rl.constants import NULL_TOKEN, UNK_TOKEN, VOCAB_NULL_INDEX, VOCAB_UNK_INDEX


class CategoricalVocab:
    """A frequency-capped, JSON-serializable string-to-index vocabulary.

    Index 0 is reserved for `<NULL>` (field absent/`None`), index 1 for `<UNK>` (a value not
    seen during fitting). Real tokens start at index 2.
    """

    def __init__(self, name: str, max_size: int | None = None) -> None:
        self.name = name
        self.max_size = max_size
        self._token_to_index: dict[str, int] = {}
        self._index_to_token: dict[int, str] | None = None
        self._fitted = False

    def fit(self, values: Iterable[str | None]) -> CategoricalVocab:
        counts: Counter[str] = Counter(v for v in values if v is not None)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if self.max_size is not None:
            ranked = ranked[: self.max_size]
        self._token_to_index = {token: index + 2 for index, (token, _count) in enumerate(ranked)}
        self._fitted = True
        return self

    @property
    def size(self) -> int:
        return len(self._token_to_index) + 2

    def encode(self, value: str | None) -> int:
        if not self._fitted:
            raise RuntimeError(f"CategoricalVocab {self.name!r} used before fit()/from_dict().")
        if value is None:
            return VOCAB_NULL_INDEX
        return self._token_to_index.get(value, VOCAB_UNK_INDEX)

    def decode(self, index: int) -> str:
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
        return {
            "name": self.name,
            "max_size": self.max_size,
            "token_to_index": self._token_to_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CategoricalVocab:
        vocab = cls(name=data["name"], max_size=data.get("max_size"))
        vocab._token_to_index = {str(k): int(v) for k, v in data["token_to_index"].items()}
        vocab._fitted = True
        return vocab

    @classmethod
    def empty(cls, name: str) -> CategoricalVocab:
        """A vocab with only the two special tokens, for use without a fitted `/nn` cache.

        See `Standardizer.identity` -- same "smoke test only" caveat applies.
        """
        vocab = cls(name=name, max_size=0)
        vocab._fitted = True
        return vocab

    def __len__(self) -> int:
        return self.size
