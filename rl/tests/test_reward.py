"""Tests for `herbert_rl.env.reward.RewardFunction` -- every reward case, with configurable weights."""

from __future__ import annotations

from factories import make_input_state, make_player_state, make_reward_weights, make_tick_record
from herbert_rl.env.reward import RewardFunction
from herbert_rl.schema import MatchState


def test_goal_scored_awards_positive_weight():
    weights = make_reward_weights(
        goal_scored=2.5, goal_conceded=0.0, bridge_progress=0.0, idle_penalty=0.0
    )
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0, match=MatchState(own_score=0, opponent_score=0))
    curr = make_tick_record(
        tick=1, match=MatchState(own_score=1, opponent_score=0), player=make_player_state(vx=1.0)
    )
    breakdown = reward_fn.compute(prev, curr)
    assert breakdown.goal_scored == 2.5
    assert breakdown.total == 2.5


def test_goal_conceded_awards_negative_weight():
    weights = make_reward_weights(
        goal_scored=0.0, goal_conceded=-3.0, bridge_progress=0.0, idle_penalty=0.0
    )
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0, match=MatchState(own_score=0, opponent_score=0))
    curr = make_tick_record(
        tick=1, match=MatchState(own_score=0, opponent_score=1), player=make_player_state(vx=1.0)
    )
    breakdown = reward_fn.compute(prev, curr)
    assert breakdown.goal_conceded == -3.0
    assert breakdown.total == -3.0


def test_no_score_change_gives_zero_goal_terms():
    weights = make_reward_weights(bridge_progress=0.0, idle_penalty=0.0)
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0, match=MatchState(own_score=1, opponent_score=1))
    curr = make_tick_record(
        tick=1, match=MatchState(own_score=1, opponent_score=1), player=make_player_state(vx=1.0)
    )
    breakdown = reward_fn.compute(prev, curr)
    assert breakdown.goal_scored == 0.0
    assert breakdown.goal_conceded == 0.0
    assert breakdown.total == 0.0


def test_missing_match_state_does_not_award_goal_terms():
    weights = make_reward_weights(bridge_progress=0.0, idle_penalty=0.0)
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0, match=None)
    curr = make_tick_record(tick=1, match=None, player=make_player_state(vx=1.0))
    breakdown = reward_fn.compute(prev, curr)
    assert breakdown.total == 0.0


def test_bridge_progress_rewards_first_forward_placement():
    weights = make_reward_weights(
        goal_scored=0.0,
        goal_conceded=0.0,
        bridge_progress=0.01,
        idle_penalty=0.0,
        bridge_axis="x",
        own_goal_forward_sign=1,
    )
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0)
    curr = make_tick_record(
        tick=1,
        player=make_player_state(vx=1.0),
        input=make_input_state(place_occurred=True, place_x=5, place_y=64, place_z=0),
    )
    breakdown = reward_fn.compute(prev, curr)
    assert breakdown.bridge_progress == 0.01
    assert breakdown.total == 0.01


def test_bridge_progress_penalizes_backward_placement_after_forward_one():
    weights = make_reward_weights(
        goal_scored=0.0,
        goal_conceded=0.0,
        bridge_progress=0.01,
        idle_penalty=0.0,
        bridge_axis="x",
        own_goal_forward_sign=1,
    )
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0)

    forward_tick = make_tick_record(
        tick=1,
        player=make_player_state(vx=1.0),
        input=make_input_state(place_occurred=True, place_x=5, place_y=64, place_z=0),
    )
    forward_breakdown = reward_fn.compute(prev, forward_tick)
    assert forward_breakdown.bridge_progress == 0.01

    backward_tick = make_tick_record(
        tick=2,
        player=make_player_state(vx=1.0),
        input=make_input_state(place_occurred=True, place_x=2, place_y=64, place_z=0),
    )
    backward_breakdown = reward_fn.compute(forward_tick, backward_tick)
    assert backward_breakdown.bridge_progress == -0.01


def test_bridge_progress_zero_when_no_placement():
    weights = make_reward_weights(
        goal_scored=0.0, goal_conceded=0.0, bridge_progress=0.01, idle_penalty=0.0
    )
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0)
    curr = make_tick_record(
        tick=1, player=make_player_state(vx=1.0), input=make_input_state(place_occurred=False)
    )
    breakdown = reward_fn.compute(prev, curr)
    assert breakdown.bridge_progress == 0.0


def test_bridge_progress_falls_back_to_player_position_when_place_coords_null():
    weights = make_reward_weights(
        goal_scored=0.0,
        goal_conceded=0.0,
        bridge_progress=0.01,
        idle_penalty=0.0,
        bridge_axis="x",
        own_goal_forward_sign=1,
    )
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0)
    curr = make_tick_record(
        tick=1,
        player=make_player_state(x=3.0, vx=1.0),
        input=make_input_state(place_occurred=True, place_x=None, place_y=None, place_z=None),
    )
    breakdown = reward_fn.compute(prev, curr)
    assert breakdown.bridge_progress == 0.01


def test_idle_penalty_applies_below_speed_threshold():
    weights = make_reward_weights(
        goal_scored=0.0,
        goal_conceded=0.0,
        bridge_progress=0.0,
        idle_penalty=-0.005,
        idle_speed_threshold=0.02,
    )
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0)
    curr = make_tick_record(tick=1, player=make_player_state(vx=0.0, vz=0.0))
    breakdown = reward_fn.compute(prev, curr)
    assert breakdown.idle_penalty == -0.005
    assert breakdown.total == -0.005


def test_idle_penalty_does_not_apply_above_speed_threshold():
    weights = make_reward_weights(
        goal_scored=0.0,
        goal_conceded=0.0,
        bridge_progress=0.0,
        idle_penalty=-0.005,
        idle_speed_threshold=0.02,
    )
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0)
    curr = make_tick_record(tick=1, player=make_player_state(vx=0.5, vz=0.0))
    breakdown = reward_fn.compute(prev, curr)
    assert breakdown.idle_penalty == 0.0


def test_reset_clears_bridge_progress_high_water_mark():
    weights = make_reward_weights(
        goal_scored=0.0,
        goal_conceded=0.0,
        bridge_progress=0.01,
        idle_penalty=0.0,
        bridge_axis="x",
        own_goal_forward_sign=1,
    )
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0)
    far_forward = make_tick_record(
        tick=1,
        player=make_player_state(vx=1.0),
        input=make_input_state(place_occurred=True, place_x=50, place_y=64, place_z=0),
    )
    reward_fn.compute(prev, far_forward)

    reward_fn.reset()

    new_prev = make_tick_record(tick=0)
    new_curr = make_tick_record(
        tick=1,
        player=make_player_state(vx=1.0),
        input=make_input_state(place_occurred=True, place_x=1, place_y=64, place_z=0),
    )
    breakdown = reward_fn.compute(new_prev, new_curr)
    # After reset, the high-water-mark is cleared, so even a "small" placement counts as new progress.
    assert breakdown.bridge_progress == 0.01


def test_total_combines_all_terms():
    weights = make_reward_weights(
        goal_scored=1.0,
        goal_conceded=-1.0,
        bridge_progress=0.01,
        idle_penalty=-0.005,
        idle_speed_threshold=0.02,
    )
    reward_fn = RewardFunction(weights)
    prev = make_tick_record(tick=0, match=MatchState(own_score=0, opponent_score=0))
    curr = make_tick_record(
        tick=1,
        match=MatchState(own_score=1, opponent_score=0),
        player=make_player_state(vx=0.0, vz=0.0),
        input=make_input_state(place_occurred=True, place_x=5, place_y=64, place_z=0),
    )
    breakdown = reward_fn.compute(prev, curr)
    assert breakdown.total == 1.0 + 0.01 + (-0.005)
