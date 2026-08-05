# SPDX-License-Identifier: MIT
"""Per-feature z-score standardizer -- a hand-synced copy of `herbert_nn.data.normalization`.

See `constants.py` for why this is a copy rather than an import. The (de)serialization format
(`to_dict`/`from_dict`) is identical to `/nn`'s so that a `standardizer` block inside an `/nn`
preprocessing cache's `manifest.json` loads here unmodified (see `nn_cache.py`).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_MIN_STD = 1e-6


class Standardizer:
    """Per-feature z-score standardizer: ``(x - mean) / std``."""

    def __init__(self) -> None:
        """Initialize an unfitted standardizer (``mean``/``std`` are ``None`` until fit)."""
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> Standardizer:
        """Compute per-column mean/std from training-split feature rows.

        Args:
            features: Array of shape ``(num_ticks, num_features)``, float32/64.

        Returns:
            ``self``, for chaining.
        """
        if features.ndim != 2:
            raise ValueError(
                f"Expected a 2D (num_ticks, num_features) array, got {features.shape}"
            )
        if features.shape[0] == 0:
            # np.mean/np.std of an empty array silently produce NaN rather than raising;
            # fail loudly here instead of letting NaN statistics propagate into training.
            raise ValueError(
                "Cannot fit a Standardizer on zero rows -- the training split has no ticks."
            )
        features64 = features.astype(np.float64)
        mean = features64.mean(axis=0)
        std = features64.std(axis=0)
        std = np.maximum(std, _MIN_STD)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the fitted standardization to a feature array.

        Args:
            features: Array of shape ``(..., num_features)`` matching the dimensionality
                used in :meth:`fit`.

        Returns:
            Standardized array, same shape, dtype ``float32``.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "Standardizer.transform() called before fit()/from_dict()."
            )
        return ((features - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        """Serialize fitted statistics to a JSON-safe dict, in the same format as `/nn`'s."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot serialize an unfitted Standardizer.")
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> Standardizer:
        """Deserialize a standardizer previously produced by :meth:`to_dict` (by this class or `/nn`'s)."""
        standardizer = cls()
        standardizer.mean = np.asarray(data["mean"], dtype=np.float32)
        standardizer.std = np.asarray(data["std"], dtype=np.float32)
        return standardizer

    @classmethod
    def identity(cls, num_features: int) -> Standardizer:
        """Build a no-op standardizer (mean=0, std=1), used only when no `/nn` cache is configured.

        This exists so the environment/smoke test can run without a fitted `/nn` cache on hand
        (e.g. `rl.smoketest` before a BC checkpoint exists), at the cost of feeding the model
        un-normalized features -- never use this for a real training run against a pretrained
        checkpoint, since its weights expect standardized inputs.
        """
        standardizer = cls()
        standardizer.mean = np.zeros(num_features, dtype=np.float32)
        standardizer.std = np.ones(num_features, dtype=np.float32)
        return standardizer
