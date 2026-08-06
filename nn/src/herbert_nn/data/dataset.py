# SPDX-License-Identifier: MIT
"""PyTorch ``Dataset`` implementations over cached per-tick tensors.

Two dataset flavors share the same underlying flat, per-tick cached tensors
(see :mod:`herbert_nn.data.cache`):

* :class:`TickDataset` -- one sample per tick, for :class:`MLPPolicy`
  (single-tick state -> single-tick action).
* :class:`WindowDataset` -- one sample per valid sliding window of
  consecutive ticks *within a single session*, for :class:`GRUPolicy`
  (a short state history -> the action at the final tick of the window).

Windows never cross session boundaries: only window end-positions ``t`` with
at least ``window_length`` preceding ticks in the *same* session are used,
so no synthetic padding is needed and no window mixes two sessions.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch.utils.data import Dataset

from herbert_nn.data.cache import SessionBoundary

logger = logging.getLogger(__name__)

#: Tensor keys copied as-is (single tick) by TickDataset, or stacked along a
#: new leading window dimension by WindowDataset, for every FEATURE field.
_FEATURE_KEYS = (
    "continuous",
    "block_grid_cells",
    "hotbar_slot_index",
    "hotbar_item_type",
    "opponent_held_item_category",
    "match_kit_type",
)
#: Tensor keys that are prediction TARGETS -- always taken from the final
#: (most recent) tick of a sample, never windowed.
_TARGET_KEYS = (
    "mouse_target",
    "discrete_target",
    "place_block_type",
    "place_mask",
    "movement_target",
)


class TickDataset(Dataset):
    """Single-tick samples: state at tick ``t`` -> action recorded at tick ``t``."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        """Wrap a split's cached tensor dict as a single-tick dataset.

        Args:
            tensors: The dict produced by :meth:`herbert_nn.data.cache.CacheBundle.load_split`.
        """
        self.tensors = tensors
        self._length = int(tensors["tick"].shape[0])

    def __len__(self) -> int:
        """Return the number of ticks in this split."""
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the single-tick feature/target sample at row ``index``."""
        sample = {key: self.tensors[key][index] for key in _FEATURE_KEYS}
        sample.update({key: self.tensors[key][index] for key in _TARGET_KEYS})
        return sample


class WindowDataset(Dataset):
    """Sliding-window samples: ticks ``[t-W+1, t]`` -> action recorded at tick ``t``."""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        boundaries: list[SessionBoundary],
        window_length: int,
        stride: int = 1,
    ) -> None:
        """Build the index of valid window end-positions.

        Args:
            tensors: The dict produced by :meth:`herbert_nn.data.cache.CacheBundle.load_split`.
            boundaries: Per-session ``[start, end)`` row ranges within ``tensors``,
                in the same row order the tensors were concatenated in.
            window_length: Number of consecutive ticks per window (``W``).
            stride: Step, in ticks, between consecutive window end-positions.

        Raises:
            ValueError: If ``window_length`` or ``stride`` is not positive.
        """
        if window_length <= 0:
            raise ValueError(f"window_length must be positive, got {window_length}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.tensors = tensors
        self.window_length = window_length
        self.stride = stride
        self._ends: list[int] = []
        for boundary in boundaries:
            session_len = boundary.end - boundary.start
            if session_len < window_length:
                continue
            first_end = boundary.start + window_length - 1
            self._ends.extend(range(first_end, boundary.end, stride))
        if not self._ends:
            logger.warning(
                "WindowDataset built 0 windows (window_length=%d longer than every "
                "session in this split?).",
                window_length,
            )

    def __len__(self) -> int:
        """Return the number of valid windows in this split."""
        return len(self._ends)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the windowed feature/target sample ending at the ``index``-th valid window."""
        end = self._ends[index]
        start = end - self.window_length + 1
        sample = {key: self.tensors[key][start : end + 1] for key in _FEATURE_KEYS}
        sample.update({key: self.tensors[key][end] for key in _TARGET_KEYS})
        return sample


ModelFamily = Literal["mlp", "gru"]


def build_dataset(
    tensors: dict[str, torch.Tensor],
    boundaries: list[SessionBoundary],
    family: ModelFamily,
    window_length: int = 32,
    window_stride: int = 1,
) -> TickDataset | WindowDataset:
    """Construct the appropriate dataset flavor for a model family.

    Args:
        tensors: A split's cached tensor dict.
        boundaries: That split's per-session row boundaries (required for
            ``family="gru"``; ignored for ``family="mlp"``).
        family: ``"mlp"`` for :class:`TickDataset`, ``"gru"`` for :class:`WindowDataset`.
        window_length: Window length, used only when ``family="gru"``.
        window_stride: Window stride, used only when ``family="gru"``.

    Returns:
        A :class:`TickDataset` or :class:`WindowDataset` yielding dicts of tensors
        (both expose ``__len__``, unlike the abstract ``torch.utils.data.Dataset``).

    Raises:
        ValueError: If ``family`` is not ``"mlp"`` or ``"gru"``.
    """
    if family == "mlp":
        return TickDataset(tensors)
    if family == "gru":
        return WindowDataset(tensors, boundaries, window_length, window_stride)
    raise ValueError(f"Unknown model family {family!r}, expected 'mlp' or 'gru'.")
