# SPDX-License-Identifier: MIT
"""Tests for `herbert_rl.env.action_wrapper` -- the flat Box <-> ActionCommand encoding."""

from __future__ import annotations

import numpy as np

from herbert_rl.env.action_wrapper import flat_action_to_command


def test_flat_action_decodes_movement_thresholds():
    action = np.array([0.9, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    command = flat_action_to_command(action)
    assert command.forward == 1
    assert command.strafe == -1


def test_flat_action_movement_near_zero_is_neutral():
    action = np.array([0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    command = flat_action_to_command(action)
    assert command.forward == 0
    assert command.strafe == 0


def test_flat_action_decodes_booleans():
    action = np.array([0.0, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0, 0.0], dtype=np.float32)
    command = flat_action_to_command(action)
    assert command.jump is True
    assert command.sneak is False
    assert command.attack is True
    assert command.place is False


def test_flat_action_decodes_and_clips_mouse():
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 250.0, -250.0], dtype=np.float32)
    command = flat_action_to_command(action)
    assert command.delta_yaw == 180.0
    assert command.delta_pitch == -180.0


def test_flat_action_wrong_shape_raises():
    import pytest

    with pytest.raises(ValueError):
        flat_action_to_command(np.zeros(5, dtype=np.float32))
