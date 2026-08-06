# SPDX-License-Identifier: MIT
"""Running-statistics (mean/std) standardization of continuous features.

Per the project spec, normalization statistics are computed **only** on the
training split and then applied unchanged to validation and test data, to
avoid leaking information about held-out sessions into the feature scaling.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Minimum standard deviation used when standardizing, to avoid dividing by
#: (near-)zero for constant-valued feature columns (e.g. a presence flag
#: that never varies within a small dataset).
_MIN_STD = 1e-6


class Standardizer:
    """Per-feature z-score standardizer: ``(x - mean) / std``.

    Attributes:
        mean: Fitted per-feature mean, shape ``(num_features,)``. ``None``
            until :meth:`fit` is called.
        std: Fitted per-feature standard deviation (floored at
            :data:`_MIN_STD`), shape ``(num_features,)``.
    """

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
        # Accumulate in float64 even though the input is typically float32:
        # for a (near-)constant column, a float32 accumulator's rounding
        # error in the mean can be a meaningful fraction of _MIN_STD, which
        # then gets massively amplified by the division below.
        features64 = features.astype(np.float64)
        mean = features64.mean(axis=0)
        std = features64.std(axis=0)
        std = np.maximum(std, _MIN_STD)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        logger.info(
            "Fitted Standardizer on %d rows x %d features (mean/std ranges: "
            "[%.4f, %.4f] / [%.4f, %.4f]).",
            features.shape[0],
            features.shape[1],
            float(mean.min()),
            float(mean.max()),
            float(std.min()),
            float(std.max()),
        )
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the fitted standardization to a feature array.

        Args:
            features: Array of shape ``(..., num_features)`` matching the
                dimensionality used in :meth:`fit`.

        Returns:
            Standardized array, same shape, dtype ``float32``.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer.transform() called before fit().")
        return ((features - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit on ``features`` then immediately transform it."""
        self.fit(features)
        return self.transform(features)

    def to_dict(self) -> dict:
        """Serialize fitted statistics to a JSON-safe dict."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot serialize an unfitted Standardizer.")
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> Standardizer:
        """Deserialize a standardizer previously produced by :meth:`to_dict`."""
        standardizer = cls()
        standardizer.mean = np.asarray(data["mean"], dtype=np.float32)
        standardizer.std = np.asarray(data["std"], dtype=np.float32)
        return standardizer
