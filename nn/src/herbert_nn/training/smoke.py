# SPDX-License-Identifier: MIT
"""Synthetic in-memory dataset used exclusively by ``herbert_nn.smoketest``.

Generates random (but shape/dtype-correct) tensors matching exactly what
:mod:`herbert_nn.data.dataset` produces from real preprocessed data, so the
smoke test exercises the real model/loss/engine code paths without needing
any real recorded session data or a (potentially slow) preprocessing pass.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from herbert_nn.constants import (
    CONTINUOUS_FEATURE_DIM,
    NUM_BLOCK_CELL_TYPES,
    NUM_DISCRETE_ACTIONS,
    NUM_HELD_ITEM_CATEGORIES,
    NUM_HOTBAR_SLOTS,
)
from herbert_nn.models.base import DataMeta

#: Small, fast synthetic dataset shapes -- deliberately tiny so a full epoch
#: over `SMOKE_NUM_SAMPLES` samples finishes in well under a second on CPU.
SMOKE_BLOCK_GRID_SHAPE = (3, 3, 3)
SMOKE_ITEM_TYPE_VOCAB_SIZE = 10
SMOKE_KIT_TYPE_VOCAB_SIZE = 6
SMOKE_PLACE_BLOCK_TYPE_VOCAB_SIZE = 8
SMOKE_WINDOW_LENGTH = 8


def smoke_data_meta() -> DataMeta:
    """Build the tiny :class:`DataMeta` used by the synthetic smoke-test dataset/model."""
    return DataMeta(
        block_grid_shape=SMOKE_BLOCK_GRID_SHAPE,
        item_type_vocab_size=SMOKE_ITEM_TYPE_VOCAB_SIZE,
        kit_type_vocab_size=SMOKE_KIT_TYPE_VOCAB_SIZE,
        place_block_type_vocab_size=SMOKE_PLACE_BLOCK_TYPE_VOCAB_SIZE,
    )


class SyntheticDataset(Dataset):
    """Random tensors matching the real dataset's schema, for pipeline smoke-testing."""

    def __init__(
        self, num_samples: int, data_meta: DataMeta, window_length: int | None
    ) -> None:
        """Initialize the synthetic dataset.

        Args:
            num_samples: Number of synthetic samples to generate.
            data_meta: Determines categorical vocab sizes / block-grid cell count.
            window_length: If ``None``, generate single-tick (MLP) samples; if an
                int, generate windowed (GRU) samples with a leading window
                dimension of this length.
        """
        self.num_samples = num_samples
        self.data_meta = data_meta
        self.window_length = window_length

    def __len__(self) -> int:
        """Return the configured number of synthetic samples."""
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Generate one deterministic-per-index synthetic sample.

        Args:
            index: Sample index, used to seed the per-sample random generator so the same
                index always yields the same tensors.

        Returns:
            A dict of tensors matching the real dataset's schema (see module docstring).
        """
        gen = torch.Generator().manual_seed(
            index
        )  # deterministic per-index, still "random"
        shape_prefix = () if self.window_length is None else (self.window_length,)
        num_cells = self.data_meta.num_block_cells

        sample = {
            "continuous": torch.randn(
                *shape_prefix, CONTINUOUS_FEATURE_DIM, generator=gen
            ),
            "block_grid_cells": torch.randint(
                0, NUM_BLOCK_CELL_TYPES, (*shape_prefix, num_cells), generator=gen
            ),
            "hotbar_slot_index": _randint_scalar_or_seq(
                NUM_HOTBAR_SLOTS, shape_prefix, gen
            ),
            "hotbar_item_type": _randint_scalar_or_seq(
                self.data_meta.item_type_vocab_size, shape_prefix, gen
            ),
            "opponent_held_item_category": _randint_scalar_or_seq(
                NUM_HELD_ITEM_CATEGORIES, shape_prefix, gen
            ),
            "match_kit_type": _randint_scalar_or_seq(
                self.data_meta.kit_type_vocab_size, shape_prefix, gen
            ),
            "mouse_target": torch.randn(2, generator=gen),
            "discrete_target": (
                torch.rand(NUM_DISCRETE_ACTIONS, generator=gen) > 0.5
            ).float(),
            "place_block_type": torch.randint(
                0, self.data_meta.place_block_type_vocab_size, (1,), generator=gen
            ).squeeze(0),
            "place_mask": (torch.rand(1, generator=gen) > 0.5).float().squeeze(0),
        }
        return sample


def _randint_scalar_or_seq(
    high: int, shape_prefix: tuple[int, ...], gen: torch.Generator
) -> torch.Tensor:
    if shape_prefix:
        return torch.randint(0, high, shape_prefix, generator=gen)
    return torch.randint(0, high, (1,), generator=gen).squeeze(0)
