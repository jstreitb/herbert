"""Early stopping on a monitored (lower-is-better) validation metric."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training when a monitored value has not improved for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """Args:
        patience: Number of consecutive non-improving epochs to tolerate
            before signaling a stop.
        min_delta: Minimum decrease required to count as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Record one epoch's monitored value.

        Args:
            value: The epoch's validation metric (lower is better).

        Returns:
            ``True`` if this is a new best value (i.e. the caller should save
            a "best" checkpoint), ``False`` otherwise.
        """
        is_best = value < self.best_value - self.min_delta
        if is_best:
            self.best_value = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered: no improvement in %d epochs (best=%.6f).",
                    self.patience,
                    self.best_value,
                )
        return is_best
