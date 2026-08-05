"""Custom PPO rollout collection for genuinely simultaneous two-sided Bridge self-play.

Reuses `stable_baselines3`'s `PPO`/`DictRolloutBuffer`/GAE machinery, but does **not** use
`PPO.learn()` or `OnPolicyAlgorithm.collect_rollouts()` -- those assume a single (possibly
vectorized) environment whose sub-envs can be stepped independently, which is exactly the
assumption that breaks for a genuinely simultaneous two-player match (see
`env/match_coordinator.py`'s module docstring for the full explanation). Instead,
:func:`collect_rollout` drives `MatchCoordinator.advance()` directly -- one call per tick, both
sides' actions computed together, no lag, no deadlock -- and manually fills
`model.rollout_buffer` (an SB3 `DictRolloutBuffer` sized for exactly ``n_envs=2``, one slot per
side of the duel) using the same `.add()` / `.compute_returns_and_advantage()` API
`collect_rollouts()` would have used. `train.py`/`smoketest.py` then call SB3's own `PPO.train()`
unmodified to perform the actual clipped-surrogate PPO update from that buffer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from herbert_rl.env.action_wrapper import flat_action_to_command
from herbert_rl.env.match_coordinator import MatchCoordinator

logger = logging.getLogger(__name__)


@dataclass
class EpisodeStats:
    """One completed episode's summary, for TensorBoard logging."""

    side: int
    total_reward: float
    length: int
    final_own_score: int | None
    final_opponent_score: int | None


@dataclass
class RolloutResult:
    obs_a: dict
    obs_b: dict
    episodes: list[EpisodeStats]
    mean_reward: float
    reward_breakdown_means: dict[str, float]


def collect_rollout(
    model: PPO, coordinator: MatchCoordinator, obs_a: dict, obs_b: dict, n_steps: int
) -> RolloutResult:
    """Fill ``model.rollout_buffer`` with exactly ``n_steps`` joint ticks of self-play.

    Both sides of the duel are controlled by the *same* policy (`model.policy`) -- symmetric
    self-play -- with their transitions stored side-by-side along the buffer's ``n_envs=2`` axis.

    Args:
        model: A `PPO` instance whose `.env` was only used for space/`n_envs` introspection at
            construction (see `train.py`) -- never stepped directly.
        coordinator: The live `MatchCoordinator` to advance.
        obs_a: Side-0's current observation (from the previous call's `RolloutResult.obs_a`, or
            `coordinator.reset()` on the very first call).
        obs_b: Side-1's current observation.
        n_steps: Number of ticks to collect (`model.rollout_buffer`'s configured `buffer_size`).

    Returns:
        A :class:`RolloutResult` with the final observations (to feed into the next call) and
        episode/reward summaries for logging. `model.rollout_buffer` is left populated and ready
        for `model.train()`.
    """
    if model.rollout_buffer.n_envs != 2:
        raise ValueError(
            f"rollout buffer must be sized for exactly 2 sides (n_envs=2), got "
            f"{model.rollout_buffer.n_envs}. Check that PPO was constructed with a 2-env VecEnv."
        )
    model.policy.set_training_mode(False)
    model.rollout_buffer.reset()

    episode_starts = np.array([False, False])
    running_reward = np.zeros(2, dtype=np.float64)
    running_length = np.zeros(2, dtype=np.int64)
    completed_episodes: list[EpisodeStats] = []
    breakdown_sums = {
        "goal_scored": 0.0,
        "goal_conceded": 0.0,
        "bridge_progress": 0.0,
        "idle_penalty": 0.0,
    }
    reward_sum = 0.0
    last_dones = np.array([False, False])

    for _ in range(n_steps):
        stacked_obs = {key: np.stack([obs_a[key], obs_b[key]]) for key in obs_a}
        with torch.no_grad():
            obs_tensor = obs_as_tensor(stacked_obs, model.device)
            actions, values, log_probs = model.policy(obs_tensor)
        actions_np = actions.cpu().numpy()
        clipped = np.clip(actions_np, model.action_space.low, model.action_space.high)

        result_a, result_b = coordinator.advance(
            flat_action_to_command(clipped[0]), flat_action_to_command(clipped[1])
        )
        results = (result_a, result_b)

        rewards = np.array([result_a.reward, result_b.reward], dtype=np.float32)
        terminated = np.array([result_a.terminated, result_b.terminated])
        truncated = np.array([result_a.truncated, result_b.truncated])
        dones = terminated | truncated

        # SB3-style bootstrap for a truncated-but-not-terminated transition (e.g. a bridge
        # disconnect mid-match): add gamma * V(next_obs), same as OnPolicyAlgorithm.collect_rollouts.
        for i in range(2):
            if truncated[i] and not terminated[i]:
                with torch.no_grad():
                    next_obs_tensor = obs_as_tensor(
                        {k: np.expand_dims(v, 0) for k, v in results[i].obs.items()}, model.device
                    )
                    bootstrap_value = model.policy.predict_values(next_obs_tensor)[0]
                rewards[i] += float(model.gamma) * float(bootstrap_value.item())

        model.rollout_buffer.add(stacked_obs, clipped, rewards, episode_starts, values, log_probs)
        model.num_timesteps += 2

        for i in range(2):
            breakdown = results[i].info["reward_breakdown"]
            running_reward[i] += results[i].reward
            running_length[i] += 1
            breakdown_sums["goal_scored"] += breakdown.goal_scored
            breakdown_sums["goal_conceded"] += breakdown.goal_conceded
            breakdown_sums["bridge_progress"] += breakdown.bridge_progress
            breakdown_sums["idle_penalty"] += breakdown.idle_penalty
            reward_sum += results[i].reward

        obs_a, obs_b = result_a.obs, result_b.obs
        last_dones = dones
        episode_starts = dones.astype(bool)

        if dones.any():
            # Match end is a joint fact (see match_coordinator.py) -- both sides finish together.
            for i in range(2):
                completed_episodes.append(
                    EpisodeStats(
                        side=i,
                        total_reward=float(running_reward[i]),
                        length=int(running_length[i]),
                        final_own_score=results[i].info["own_score"],
                        final_opponent_score=results[i].info["opponent_score"],
                    )
                )
                running_reward[i] = 0.0
                running_length[i] = 0
            obs_a, obs_b = coordinator.reset()

    with torch.no_grad():
        stacked_obs = {key: np.stack([obs_a[key], obs_b[key]]) for key in obs_a}
        last_values = model.policy.predict_values(obs_as_tensor(stacked_obs, model.device))
    model.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=last_dones)

    n_ticks = n_steps * 2
    reward_breakdown_means = {k: v / n_ticks for k, v in breakdown_sums.items()}
    return RolloutResult(
        obs_a=obs_a,
        obs_b=obs_b,
        episodes=completed_episodes,
        mean_reward=reward_sum / n_ticks,
        reward_breakdown_means=reward_breakdown_means,
    )
