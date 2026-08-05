"""Tests for `herbert_rl.env.spaces` -- observation/action space shape consistency with `/nn`'s schema."""

from __future__ import annotations

from factories import make_cache_stats
from herbert_rl.constants import CONTINUOUS_FEATURE_DIM, NUM_HELD_ITEM_CATEGORIES, NUM_HOTBAR_SLOTS
from herbert_rl.env.spaces import build_action_space, build_observation_space


def test_observation_space_continuous_dim_matches_nn_feature_layout():
    cache_stats = make_cache_stats()
    space = build_observation_space(cache_stats, window_length=1)
    assert space["continuous"].shape == (1, CONTINUOUS_FEATURE_DIM)


def test_observation_space_window_length_applies_to_every_field():
    cache_stats = make_cache_stats()
    window_length = 8
    space = build_observation_space(cache_stats, window_length=window_length)
    assert space["continuous"].shape == (window_length, CONTINUOUS_FEATURE_DIM)
    assert space["block_grid_cells"].shape == (window_length, cache_stats.num_block_cells)
    assert space["hotbar_slot_index"].shape == (window_length,)
    assert space["hotbar_item_type"].shape == (window_length,)
    assert space["opponent_held_item_category"].shape == (window_length,)
    assert space["match_kit_type"].shape == (window_length,)


def test_observation_space_block_grid_cell_count_matches_cache_shape():
    cache_stats = make_cache_stats(block_grid_shape=(7, 3, 7))
    space = build_observation_space(cache_stats, window_length=1)
    assert space["block_grid_cells"].shape == (1, 7 * 3 * 7)


def test_observation_space_vocab_sized_fields_match_cache_vocab_sizes():
    cache_stats = make_cache_stats(item_type_vocab_size=12, kit_type_vocab_size=6)
    space = build_observation_space(cache_stats, window_length=1)
    # Box categorical fields are typed [0, num_categories - 1] -- see spaces.py's docstring for
    # why these are Box, not MultiDiscrete.
    assert space["hotbar_item_type"].high[0] == 11
    assert space["match_kit_type"].high[0] == 5
    assert space["opponent_held_item_category"].high[0] == NUM_HELD_ITEM_CATEGORIES - 1
    assert space["hotbar_slot_index"].high[0] == NUM_HOTBAR_SLOTS - 1


def test_observation_space_rejects_nonpositive_window_length():
    import pytest

    cache_stats = make_cache_stats()
    with pytest.raises(ValueError):
        build_observation_space(cache_stats, window_length=0)


def test_action_space_covers_every_required_field():
    space = build_action_space()
    expected_keys = {"move_forward", "strafe", "jump", "sneak", "attack", "place", "mouse"}
    assert set(space.spaces.keys()) == expected_keys


def test_action_space_movement_axes_are_ternary():
    space = build_action_space()
    assert space["move_forward"].n == 3
    assert space["strafe"].n == 3


def test_action_space_booleans_are_binary():
    space = build_action_space()
    for key in ("jump", "sneak", "attack", "place"):
        assert space[key].n == 2


def test_action_space_mouse_is_continuous_2d():
    space = build_action_space()
    assert space["mouse"].shape == (2,)
