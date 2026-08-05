# SPDX-License-Identifier: MIT
"""Shared setup: build the two bridge-backed envs, the `MatchCoordinator`, and the `PPO` model.

Used by both `train.py` (Hydra-driven, requires a real `/nn` checkpoint) and `smoketest.py`
(argparse-driven, optionally skips the checkpoint for a from-scratch wiring check) so the two
entry points can't drift apart on how the pieces fit together.
"""

from __future__ import annotations

import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from herbert_rl.env.action_wrapper import FlattenHerbertActionWrapper
from herbert_rl.env.herbert_bridge_env import make_duel_envs
from herbert_rl.env.match_coordinator import MatchCoordinator, MatchEndConfig
from herbert_rl.env.reward import RewardWeights
from herbert_rl.nn_cache import NNCacheStats, load_nn_cache_stats
from herbert_rl.policy.backbone import RLDataMeta
from herbert_rl.policy.checkpoint_adapter import (
    LoadedBackbone,
    build_fresh_backbone,
    load_nn_checkpoint,
)
from herbert_rl.policy.sb3_policy import HerbertRLPolicy

logger = logging.getLogger(__name__)


def build_reward_weights(reward_cfg: dict) -> tuple[RewardWeights, RewardWeights]:
    """Build the (side A, side B) `RewardWeights` pair.

    Side B's ``own_goal_forward_sign`` is the negation of side A's -- the two ends of a
    symmetric Bridge map face opposite directions along the configured `bridge_axis` (see
    `env/reward.py`'s docstring on that field).
    """
    weights_a = RewardWeights(**reward_cfg)
    weights_b = RewardWeights(
        **{**reward_cfg, "own_goal_forward_sign": -reward_cfg["own_goal_forward_sign"]}
    )
    return weights_a, weights_b


def build_training_components(
    *,
    host: str,
    port: int,
    username_a: str,
    username_b: str,
    nn_cache_manifest_path: str,
    checkpoint_path: str | None,
    fresh_model_cfg: dict | None,
    window_length: int,
    view_distance: int,
    node_executable: str,
    bridge_log_level: str,
    bridge_startup_timeout_s: float,
    bridge_reset_timeout_s: float,
    bridge_tick_timeout_s: float,
    reward_cfg: dict,
    match_end: MatchEndConfig,
    ppo_kwargs: dict,
    device: str,
    seed: int,
) -> tuple[PPO, MatchCoordinator, dict, dict]:
    """Build everything needed to start collecting rollouts: spawns both bridge processes.

    Args:
        host: Hostname/IP of the private Minecraft server both bots connect to.
        port: Port of the private Minecraft server.
        username_a: Bot account username for side A of the duel.
        username_b: Bot account username for side B of the duel.
        nn_cache_manifest_path: Path to the `/nn` preprocessing cache `manifest.json` (or its
            parent directory) the checkpoint being fine-tuned was trained against.
        checkpoint_path: Path to a `/nn` checkpoint to warm-start the policy from. If ``None``,
            ``fresh_model_cfg`` (a `/nn`-shaped ``model`` config dict, e.g.
            ``{"family": "mlp", "hidden_dims": [64, 32], ...}``) is used to build a randomly
            initialized backbone instead -- only valid for `rl.smoketest`.
        fresh_model_cfg: Fallback model config used only when ``checkpoint_path`` is ``None``.
        window_length: Number of ticks per GRU observation window (``1`` for an MLP family).
        view_distance: Block-grid view distance passed through to the bridge process.
        node_executable: Path/name of the ``node`` executable used to spawn bridge processes.
        bridge_log_level: Log level passed to each spawned `/rl/bridge` Node.js process.
        bridge_startup_timeout_s: Seconds to wait for a bridge process to report ready.
        bridge_reset_timeout_s: Seconds to wait for a bridge process to complete a reset.
        bridge_tick_timeout_s: Seconds to wait for a bridge process to respond to one tick.
        reward_cfg: Reward-weight config dict passed to :func:`build_reward_weights`.
        match_end: Match-end detection thresholds (score/timeout) shared by both sides.
        ppo_kwargs: Extra keyword arguments forwarded to `stable_baselines3.PPO`.
        device: Torch device string (``"auto"``/``"cpu"``/``"cuda"``) for the PPO model.
        seed: RNG seed forwarded to `stable_baselines3.PPO`.

    Returns:
        ``(model, coordinator, obs_a, obs_b)`` -- ``model.rollout_buffer``/``model.policy`` are
        ready to use with `train/rollout.py::collect_rollout`; ``obs_a``/``obs_b`` are the
        post-reset initial observations for the first rollout call.
    """
    cache_stats: NNCacheStats = load_nn_cache_stats(nn_cache_manifest_path)

    loaded_backbone: LoadedBackbone
    if checkpoint_path is not None:
        loaded_backbone = load_nn_checkpoint(checkpoint_path, map_location=device)
    else:
        if fresh_model_cfg is None:
            raise ValueError(
                "Either checkpoint_path or fresh_model_cfg must be provided."
            )
        data_meta = RLDataMeta(
            block_grid_shape=cache_stats.block_grid_shape,
            item_type_vocab_size=cache_stats.item_type_vocab.size,
            kit_type_vocab_size=cache_stats.kit_type_vocab.size,
        )
        loaded_backbone = build_fresh_backbone(fresh_model_cfg, data_meta)

    reward_weights_a, reward_weights_b = build_reward_weights(reward_cfg)

    env_a, env_b, coordinator = make_duel_envs(
        host=host,
        port=port,
        username_a=username_a,
        username_b=username_b,
        cache_stats=cache_stats,
        window_length=window_length,
        reward_weights_a=reward_weights_a,
        reward_weights_b=reward_weights_b,
        match_end=match_end,
        node_executable=node_executable,
        bridge_startup_timeout_s=bridge_startup_timeout_s,
        bridge_reset_timeout_s=bridge_reset_timeout_s,
        bridge_tick_timeout_s=bridge_tick_timeout_s,
        view_distance=view_distance,
        bridge_log_level=bridge_log_level,
    )
    env_a_flat = FlattenHerbertActionWrapper(env_a)
    env_b_flat = FlattenHerbertActionWrapper(env_b)
    # Only used for observation/action-space + n_envs introspection by PPO's constructor -- never
    # stepped directly. See env/match_coordinator.py's module docstring for why.
    vec_env = DummyVecEnv([lambda: env_a_flat, lambda: env_b_flat])

    model = PPO(
        policy=HerbertRLPolicy,
        env=vec_env,
        policy_kwargs={"loaded_backbone": loaded_backbone},
        seed=seed,
        device=device,
        **ppo_kwargs,
    )

    logger.info("Requesting initial reset to start the first duel...")
    obs_a, obs_b = coordinator.reset()
    return model, coordinator, obs_a, obs_b
