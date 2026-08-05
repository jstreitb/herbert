# SPDX-License-Identifier: MIT
"""Output heads shared by :class:`MLPPolicy` and :class:`GRUPolicy`.

Each head consumes the trunk's final hidden vector (shape ``[batch, trunk_dim]``)
and produces raw (unnormalized) logits/values; losses (see
:mod:`herbert_nn.models.losses`) apply the appropriate activation internally
(``nn.BCEWithLogitsLoss`` / ``nn.CrossEntropyLoss`` both expect raw logits).
"""

from __future__ import annotations

import torch
from torch import nn

from herbert_nn.constants import MOUSE_TARGET_DIM, NUM_DISCRETE_ACTIONS


class MouseHead(nn.Module):
    """Continuous regression head for mouse movement (``d_yaw``, ``d_pitch``)."""

    def __init__(self, trunk_dim: int) -> None:
        """Initialize the head.

        Args:
            trunk_dim: Size of the trunk's final hidden vector this head consumes.
        """
        super().__init__()
        self.linear = nn.Linear(trunk_dim, MOUSE_TARGET_DIM)

    def forward(self, trunk_output: torch.Tensor) -> torch.Tensor:
        """Compute the raw (unnormalized) regression output.

        Args:
            trunk_output: Trunk's final hidden vector, shape ``[batch, trunk_dim]``.

        Returns:
            Raw regression output, shape ``[batch, 2]``.
        """
        return self.linear(trunk_output)


class DiscreteHead(nn.Module):
    """Binary multi-label head for ``jump``, ``sneak``, ``attack``, ``place``."""

    def __init__(self, trunk_dim: int) -> None:
        """Initialize the head.

        Args:
            trunk_dim: Size of the trunk's final hidden vector this head consumes.
        """
        super().__init__()
        self.linear = nn.Linear(trunk_dim, NUM_DISCRETE_ACTIONS)

    def forward(self, trunk_output: torch.Tensor) -> torch.Tensor:
        """Compute the raw (pre-sigmoid) multi-label logits.

        Args:
            trunk_output: Trunk's final hidden vector, shape ``[batch, trunk_dim]``.

        Returns:
            Raw logits (pre-sigmoid), shape ``[batch, 4]``.
        """
        return self.linear(trunk_output)


class BlockPlacementHead(nn.Module):
    """Softmax classification head over the fitted bridgeable block-type vocabulary."""

    def __init__(self, trunk_dim: int, block_type_vocab_size: int) -> None:
        """Initialize the head.

        Args:
            trunk_dim: Size of the trunk's final hidden vector this head consumes.
            block_type_vocab_size: Number of bridgeable block types in the fitted vocabulary.
        """
        super().__init__()
        self.linear = nn.Linear(trunk_dim, block_type_vocab_size)

    def forward(self, trunk_output: torch.Tensor) -> torch.Tensor:
        """Compute the raw (pre-softmax) classification logits.

        Args:
            trunk_output: Trunk's final hidden vector, shape ``[batch, trunk_dim]``.

        Returns:
            Raw logits (pre-softmax), shape ``[batch, block_type_vocab_size]``.
        """
        return self.linear(trunk_output)
