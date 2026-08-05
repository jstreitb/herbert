"""Tests for `herbert_rl.features.TickFeatureEncoder`."""

from __future__ import annotations

import numpy as np
import pytest

from factories import make_cache_stats, make_opponent, make_player_state, make_tick_record
from herbert_rl.constants import CONTINUOUS_FEATURE_DIM
from herbert_rl.features import BlockGridShapeError, TickFeatureEncoder


def test_encode_produces_expected_shapes():
    cache_stats = make_cache_stats()
    encoder = TickFeatureEncoder(cache_stats)
    record = make_tick_record(tick=0)
    features = encoder.encode(record)
    assert features.continuous.shape == (CONTINUOUS_FEATURE_DIM,)
    assert features.continuous.dtype == np.float32
    assert features.block_grid_cells.shape == (cache_stats.num_block_cells,)


def test_position_delta_is_zero_on_first_tick_after_reset():
    cache_stats = make_cache_stats()
    encoder = TickFeatureEncoder(cache_stats)
    record = make_tick_record(tick=0, player=make_player_state(x=100.0, y=64.0, z=-50.0))
    encoder.encode(record)  # first call establishes prev position
    encoder.reset()
    from herbert_rl.constants import CONTINUOUS_FEATURE_NAMES

    dx_index = CONTINUOUS_FEATURE_NAMES.index("player_dx")
    record2 = make_tick_record(tick=0, player=make_player_state(x=100.0, y=64.0, z=-50.0))
    features = encoder.encode(record2)
    assert features.continuous[dx_index] == pytest.approx(0.0)


def test_position_delta_tracks_movement_between_ticks():
    from herbert_rl.constants import CONTINUOUS_FEATURE_NAMES

    cache_stats = make_cache_stats()
    encoder = TickFeatureEncoder(cache_stats)
    dx_index = CONTINUOUS_FEATURE_NAMES.index("player_dx")

    encoder.encode(make_tick_record(tick=0, player=make_player_state(x=0.0)))
    features = encoder.encode(make_tick_record(tick=1, player=make_player_state(x=3.0)))
    assert features.continuous[dx_index] == pytest.approx(3.0)


def test_opponent_absent_leaves_presence_flag_zero():
    from herbert_rl.constants import CONTINUOUS_FEATURE_NAMES

    cache_stats = make_cache_stats()
    encoder = TickFeatureEncoder(cache_stats)
    presence_index = CONTINUOUS_FEATURE_NAMES.index("opponent_present")
    features = encoder.encode(make_tick_record(tick=0, opponent=None))
    assert features.continuous[presence_index] == 0.0


def test_opponent_present_sets_presence_flag_and_relative_position():
    from herbert_rl.constants import CONTINUOUS_FEATURE_NAMES

    cache_stats = make_cache_stats()
    encoder = TickFeatureEncoder(cache_stats)
    presence_index = CONTINUOUS_FEATURE_NAMES.index("opponent_present")
    rel_x_index = CONTINUOUS_FEATURE_NAMES.index("opponent_rel_x")
    features = encoder.encode(make_tick_record(tick=0, opponent=make_opponent(rel_x=4.5)))
    assert features.continuous[presence_index] == 1.0
    assert features.continuous[rel_x_index] == pytest.approx(4.5)


def test_mismatched_block_grid_shape_raises():
    cache_stats = make_cache_stats(block_grid_shape=(7, 3, 7))
    encoder = TickFeatureEncoder(cache_stats)
    from factories import make_block_grid

    bad_record = make_tick_record(tick=0, block_grid=make_block_grid(shape=(5, 3, 5)))
    with pytest.raises(BlockGridShapeError):
        encoder.encode(bad_record)


def test_hotbar_item_type_uses_configured_vocab():
    cache_stats = make_cache_stats(item_type_vocab_size=8)
    encoder = TickFeatureEncoder(cache_stats)
    from herbert_rl.schema import HeldItem

    known_token = next(iter(cache_stats.item_type_vocab._token_to_index))
    record = make_tick_record(
        tick=0, held_item=HeldItem(hotbar_slot=2, item_id=known_token, count=10)
    )
    features = encoder.encode(record)
    assert features.hotbar_item_type == cache_stats.item_type_vocab.encode(known_token)
    assert features.hotbar_item_type != cache_stats.item_type_vocab.encode(None)
