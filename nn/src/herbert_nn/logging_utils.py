# SPDX-License-Identifier: MIT
"""Shared structured-logging setup for herbert_nn CLI entry points."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a single stream handler and a consistent format.

    Idempotent: calling this more than once (e.g. because both a CLI module
    and a library function it calls both configure logging) will not
    duplicate handlers.

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
