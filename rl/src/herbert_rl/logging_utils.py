# SPDX-License-Identifier: MIT
"""Shared structured-logging setup for herbert_rl CLI entry points (copy of `herbert_nn`'s)."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a single stream handler and a consistent format.

    Idempotent: calling this more than once will not duplicate handlers. Verbose (`INFO`) by
    default, per the task's "log verbosely by default" requirement -- the developer debugging
    bridge/IPC timing issues needs to see bridge lifecycle events, reward breakdowns, and
    checkpoint-loading key mismatches without extra flags; pass ``--log-level WARNING`` once
    things are stable.

    Args:
        level: Root logger level, e.g. ``logging.INFO`` or ``logging.DEBUG``.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
