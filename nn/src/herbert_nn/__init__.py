# SPDX-License-Identifier: MIT
"""Herbert /nn: behavioral-cloning training pipeline for Hypixel Bridge duels.

This package implements the full data -> model -> training -> evaluation
pipeline for the Herbert project's neural-network component. It consumes
``.jsonl`` session logs produced by the ``/mod`` component (see
:mod:`herbert_nn.schemas` for the versioned data contract) and trains small,
fast-iterating imitation-learning policies (MLP / GRU) on a single consumer
GPU.
"""

from __future__ import annotations

__version__ = "0.1.0"
