# SPDX-License-Identifier: MIT
"""Session-level (never tick-level) train/val/test splitting.

Splitting at the tick level would leak information across the split
boundary (adjacent ticks within a session are highly correlated), making
held-out metrics meaningless. Instead we split whole sessions -- every tick
of a given ``session_id`` ends up in exactly one of train/val/test.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionSplit:
    """Session-id membership for each of the three splits."""

    train: list[str]
    val: list[str]
    test: list[str]

    def split_of(self, session_id: str) -> str:
        """Return which split (``"train"``/``"val"``/``"test"``) owns ``session_id``."""
        if session_id in self.train:
            return "train"
        if session_id in self.val:
            return "val"
        if session_id in self.test:
            return "test"
        raise KeyError(f"session_id {session_id!r} is not part of this split.")


def split_sessions(
    session_ids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> SessionSplit:
    """Deterministically partition session ids into train/val/test.

    Args:
        session_ids: All distinct session ids available (order-independent;
            internally de-duplicated and sorted before shuffling so the
            result is reproducible regardless of input order).
        train_ratio: Fraction of sessions assigned to the training split.
        val_ratio: Fraction of sessions assigned to the validation split.
        test_ratio: Fraction of sessions assigned to the test split.
        seed: Random seed controlling the shuffle, for reproducibility.

    Returns:
        A :class:`SessionSplit` with disjoint, exhaustive session-id lists.

    Raises:
        ValueError: If any ratio is outside ``[0.0, 1.0]``, if the ratios do not sum to
            ~1.0, or there are too few distinct sessions to populate every non-zero-ratio
            split with at least one session.
    """
    for name, ratio in (
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ):
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{name} must be within [0.0, 1.0], got {ratio}")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got {train_ratio} + {val_ratio} + "
            f"{test_ratio} = {total_ratio}"
        )

    unique_ids = sorted(set(session_ids))
    n = len(unique_ids)
    if n == 0:
        raise ValueError("Cannot split an empty list of session ids.")

    rng = random.Random(seed)
    shuffled = unique_ids[:]
    rng.shuffle(shuffled)

    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    # Test gets the remainder so the three counts always sum to n exactly.
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    # (the test split gets whatever remains, so the three counts sum to n exactly)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    for name, ratio, ids in (
        ("train", train_ratio, train_ids),
        ("val", val_ratio, val_ids),
        ("test", test_ratio, test_ids),
    ):
        if ratio > 0 and not ids:
            raise ValueError(
                f"{name} split has ratio {ratio} > 0 but received 0 of {n} sessions; "
                "you need more recorded sessions, or adjust the split ratios."
            )

    logger.info(
        "Session split (seed=%d): train=%d, val=%d, test=%d (of %d total).",
        seed,
        len(train_ids),
        len(val_ids),
        len(test_ids),
        n,
    )
    return SessionSplit(train=train_ids, val=val_ids, test=test_ids)
