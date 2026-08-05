"""Tests for `herbert_rl.env.ipc` -- JSON-lines IPC serialization/deserialization round-trips."""

from __future__ import annotations

import json

import pytest

from factories import make_tick_record
from herbert_rl.env.ipc import (
    ActionCommand,
    LifecycleEvent,
    action_dict_to_command,
    close_command_line,
    parse_bridge_line,
    reset_command_line,
    try_parse_bridge_line,
)
from herbert_rl.schema import TickRecordRL


def test_action_command_round_trips_through_json():
    command = ActionCommand(
        forward=1,
        strafe=-1,
        jump=True,
        sneak=False,
        delta_yaw=12.5,
        delta_pitch=-3.25,
        attack=True,
        place=False,
    )
    line = command.to_json_line()
    decoded = json.loads(line)
    assert decoded == {
        "cmd": "action",
        "forward": 1,
        "strafe": -1,
        "jump": True,
        "sneak": False,
        "delta_yaw": 12.5,
        "delta_pitch": -3.25,
        "attack": True,
        "place": False,
    }


def test_reset_and_close_command_lines():
    assert json.loads(reset_command_line()) == {"cmd": "reset"}
    assert json.loads(close_command_line()) == {"cmd": "close"}


def test_action_dict_to_command_decodes_discrete_movement_axes():
    action = {
        "move_forward": 2,  # Discrete index 2 -> axis value +1
        "strafe": 0,  # Discrete index 0 -> axis value -1
        "jump": 1,
        "sneak": 0,
        "attack": 1,
        "place": 0,
        "mouse": [10.0, -5.0],
    }
    command = action_dict_to_command(action)
    assert command.forward == 1
    assert command.strafe == -1
    assert command.jump is True
    assert command.sneak is False
    assert command.attack is True
    assert command.place is False
    assert command.delta_yaw == 10.0
    assert command.delta_pitch == -5.0


def test_parse_bridge_line_round_trips_a_tick_observation():
    record = make_tick_record(tick=42)
    line = record.model_dump_json()
    parsed = parse_bridge_line(line)
    assert isinstance(parsed, TickRecordRL)
    assert parsed == record


def test_parse_bridge_line_parses_lifecycle_events():
    ready = parse_bridge_line(json.dumps({"event": "ready"}))
    assert isinstance(ready, LifecycleEvent)
    assert ready.event == "ready"

    error = parse_bridge_line(json.dumps({"event": "error", "message": "boom"}))
    assert isinstance(error, LifecycleEvent)
    assert error.event == "error"
    assert error.message == "boom"


def test_parse_bridge_line_rejects_malformed_json():
    with pytest.raises(json.JSONDecodeError):
        parse_bridge_line("{not valid json")


def test_parse_bridge_line_rejects_unrecognized_shape():
    with pytest.raises(ValueError):
        parse_bridge_line(json.dumps({"unexpected": "shape"}))


def test_try_parse_bridge_line_never_raises_on_bad_input():
    message, error = try_parse_bridge_line("{not valid json")
    assert message is None
    assert error is not None
    assert "JSONDecodeError" in error


def test_try_parse_bridge_line_returns_message_on_success():
    record = make_tick_record(tick=1)
    message, error = try_parse_bridge_line(record.model_dump_json())
    assert error is None
    assert isinstance(message, TickRecordRL)
