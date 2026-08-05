"""Tests for `herbert_rl.env.match_coordinator.MatchCoordinator`, with a fake `BridgeProcess`
standing in for a real Mineflayer bridge subprocess (see module docstring on `FakeBridgeProcess`)
so these run without a server, per the task's "mock the Mineflayer bridge process" requirement.
"""

from __future__ import annotations

from factories import (
    make_cache_stats,
    make_input_state,
    make_player_state,
    make_reward_weights,
    make_tick_record,
)
from herbert_rl.env.ipc import ActionCommand
from herbert_rl.env.match_coordinator import NO_OP_ACTION, MatchCoordinator, MatchEndConfig
from herbert_rl.env.reward import RewardFunction
from herbert_rl.features import TickFeatureEncoder
from herbert_rl.schema import MatchState


class FakeBridgeProcess:
    """A minimal stand-in for `herbert_rl.env.bridge_process.BridgeProcess`: implements only the
    three methods `MatchCoordinator` calls (`send_action`, `send_reset`, `read_tick`, `close`),
    returning pre-scripted `TickRecordRL`s instead of talking to a real Node.js/Mineflayer
    process. Each call to `send_action` advances an internal cursor into `scripted_ticks`.
    """

    def __init__(self, initial_tick, scripted_ticks):
        self.initial_tick = initial_tick
        self.scripted_ticks = list(scripted_ticks)
        self._cursor = 0
        self.sent_actions: list[ActionCommand] = []
        self.reset_count = 0
        self.closed = False

    def send_reset(self):
        self.reset_count += 1
        self._cursor = 0
        return self.initial_tick

    def send_action(self, action: ActionCommand) -> None:
        self.sent_actions.append(action)

    def read_tick(self):
        record = self.scripted_ticks[self._cursor]
        self._cursor += 1
        return record

    def close(self) -> None:
        self.closed = True


def _build_coordinator(bridge_a, bridge_b, window_length=1, match_end=None):
    cache_stats = make_cache_stats()
    weights_a = make_reward_weights()
    weights_b = make_reward_weights(own_goal_forward_sign=-weights_a.own_goal_forward_sign)
    return MatchCoordinator(
        bridge_a=bridge_a,
        bridge_b=bridge_b,
        reward_a=RewardFunction(weights_a),
        reward_b=RewardFunction(weights_b),
        feature_encoder_a=TickFeatureEncoder(cache_stats),
        feature_encoder_b=TickFeatureEncoder(cache_stats),
        window_length=window_length,
        match_end=match_end or MatchEndConfig(),
    )


def test_reset_returns_windowed_obs_for_both_sides():
    initial_a = make_tick_record(tick=0, player=make_player_state(x=1.0))
    initial_b = make_tick_record(tick=0, player=make_player_state(x=2.0))
    bridge_a = FakeBridgeProcess(initial_a, [])
    bridge_b = FakeBridgeProcess(initial_b, [])
    coordinator = _build_coordinator(bridge_a, bridge_b, window_length=4)

    obs_a, obs_b = coordinator.reset()

    assert obs_a["continuous"].shape[0] == 4
    assert obs_b["continuous"].shape[0] == 4
    assert bridge_a.reset_count == 1
    assert bridge_b.reset_count == 1


def test_advance_sends_actions_to_both_bridges_and_returns_rewards():
    initial = make_tick_record(tick=0)
    tick_a = make_tick_record(tick=1, player=make_player_state(vx=1.0))
    tick_b = make_tick_record(tick=1, player=make_player_state(vx=1.0))
    bridge_a = FakeBridgeProcess(initial, [tick_a])
    bridge_b = FakeBridgeProcess(initial, [tick_b])
    coordinator = _build_coordinator(bridge_a, bridge_b)
    coordinator.reset()

    action_a = ActionCommand(1, 0, False, False, 0.0, 0.0, False, False)
    action_b = ActionCommand(-1, 0, False, False, 0.0, 0.0, False, False)
    result_a, result_b = coordinator.advance(action_a, action_b)

    assert bridge_a.sent_actions == [action_a]
    assert bridge_b.sent_actions == [action_b]
    assert result_a.terminated is False
    assert result_b.terminated is False
    assert isinstance(result_a.reward, float)


def test_advance_detects_match_end_via_score_threshold():
    initial = make_tick_record(tick=0, match=MatchState(own_score=2, opponent_score=0))
    winning_tick = make_tick_record(
        tick=1, match=MatchState(own_score=3, opponent_score=0), player=make_player_state(vx=1.0)
    )
    bridge_a = FakeBridgeProcess(initial, [winning_tick])
    bridge_b = FakeBridgeProcess(initial, [initial])
    coordinator = _build_coordinator(
        bridge_a, bridge_b, match_end=MatchEndConfig(score_threshold=3, chat_patterns=[])
    )
    coordinator.reset()

    result_a, result_b = coordinator.advance(NO_OP_ACTION, NO_OP_ACTION)

    assert result_a.terminated is True
    assert result_b.terminated is True


def test_advance_detects_match_end_via_chat_pattern():
    initial = make_tick_record(tick=0)
    ending_tick = make_tick_record(
        tick=1, player=make_player_state(vx=1.0), chat=["RedTeam has won the game!"]
    )
    bridge_a = FakeBridgeProcess(initial, [ending_tick])
    bridge_b = FakeBridgeProcess(initial, [initial])
    coordinator = _build_coordinator(
        bridge_a,
        bridge_b,
        match_end=MatchEndConfig(score_threshold=None, chat_patterns=[r"has won the game"]),
    )
    coordinator.reset()

    result_a, result_b = coordinator.advance(NO_OP_ACTION, NO_OP_ACTION)

    assert result_a.terminated is True
    assert result_b.terminated is True


def test_advance_handles_disconnected_tick_without_crashing():
    initial = make_tick_record(tick=0)
    disconnected_tick = make_tick_record(tick=1, disconnected=True, input=make_input_state())
    bridge_a = FakeBridgeProcess(initial, [disconnected_tick])
    bridge_b = FakeBridgeProcess(initial, [initial])
    coordinator = _build_coordinator(bridge_a, bridge_b)
    coordinator.reset()

    result_a, result_b = coordinator.advance(NO_OP_ACTION, NO_OP_ACTION)

    assert result_a.truncated is True
    assert result_b.truncated is True
    assert result_a.terminated is False
    assert result_a.reward == 0.0


def test_advance_single_pairs_with_no_op_for_untouched_side():
    initial = make_tick_record(tick=0)
    tick_a = make_tick_record(tick=1, player=make_player_state(vx=1.0))
    bridge_a = FakeBridgeProcess(initial, [tick_a])
    bridge_b = FakeBridgeProcess(initial, [initial])
    coordinator = _build_coordinator(bridge_a, bridge_b)
    coordinator.reset()

    action_a = ActionCommand(1, 0, False, False, 0.0, 0.0, False, False)
    coordinator.advance_single(0, action_a)

    assert bridge_a.sent_actions == [action_a]
    assert bridge_b.sent_actions == [NO_OP_ACTION]


def test_close_closes_both_bridges():
    initial = make_tick_record(tick=0)
    bridge_a = FakeBridgeProcess(initial, [])
    bridge_b = FakeBridgeProcess(initial, [])
    coordinator = _build_coordinator(bridge_a, bridge_b)
    coordinator.close()
    assert bridge_a.closed is True
    assert bridge_b.closed is True
