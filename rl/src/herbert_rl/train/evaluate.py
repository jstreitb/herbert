# SPDX-License-Identifier: MIT
r"""``python -m herbert_rl.train.evaluate`` -- load a PPO checkpoint, run N self-play episodes.

Prints win-rate and average reward.

Since Herbert RL trains via *symmetric* self-play (both sides of the duel share one policy --
see `env/match_coordinator.py`), evaluating "the policy" necessarily means running it against
itself. Side-A/side-B win rate is therefore expected to hover near 50/50 in expectation and is
mostly a sanity signal (are matches ending decisively via score, rather than timing out /
truncating on disconnect); **average per-episode reward is the primary "did this improve over
the BC baseline" signal** -- compare it against earlier checkpoints (including update 1, right
after the BC warm start) to see whether PPO fine-tuning moved the needle at all, per the task's
"observe whether the policy improves at all" framing.

Requires a real server connection, like `rl.train`/`rl.smoketest` (see `rl/server/SETUP.md`).

Example::

    python -m herbert_rl.train.evaluate --checkpoint runs/default/2026-08-05_.../best.zip \
        --host 192.168.1.50 --nn-cache-manifest-path /path/to/nn/data/cache/<hash> --episodes 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from herbert_rl.env.action_wrapper import flat_action_to_command
from herbert_rl.env.herbert_bridge_env import make_duel_envs
from herbert_rl.env.match_coordinator import MatchEndConfig
from herbert_rl.env.reward import RewardWeights
from herbert_rl.logging_utils import configure_logging
from herbert_rl.nn_cache import load_nn_cache_stats
from herbert_rl.train.train import resolve_device

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for this CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained herbert_rl PPO checkpoint via self-play, over N episodes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a herbert_rl PPO .zip checkpoint."
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=25565)
    parser.add_argument("--username-a", default="HerbertBot1")
    parser.add_argument("--username-b", default="HerbertBot2")
    parser.add_argument("--nn-cache-manifest-path", required=True)
    parser.add_argument("--window-length", type=int, default=1)
    parser.add_argument("--view-distance", type=int, default=6)
    parser.add_argument("--node-executable", default="node")
    parser.add_argument("--bridge-log-level", default="info")
    parser.add_argument("--bridge-startup-timeout-s", type=float, default=60.0)
    parser.add_argument("--bridge-reset-timeout-s", type=float, default=120.0)
    parser.add_argument("--bridge-tick-timeout-s", type=float, default=5.0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the deterministic (mean) action.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output", default=None, help="Optional path to also write the summary JSON."
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(level=getattr(logging, args.log_level.upper()))

    device = resolve_device(args.device)
    model = PPO.load(args.checkpoint, device=device)
    logger.info("Loaded PPO checkpoint from %s onto device=%s", args.checkpoint, device)
    if not isinstance(model.action_space, spaces.Box):
        raise TypeError(
            f"Expected a continuous Box action space (see env/action_wrapper.py), got "
            f"{type(model.action_space).__name__}."
        )
    action_space: spaces.Box = model.action_space

    cache_stats = load_nn_cache_stats(args.nn_cache_manifest_path)
    reward_weights_a = RewardWeights()
    reward_weights_b = RewardWeights(
        own_goal_forward_sign=-reward_weights_a.own_goal_forward_sign
    )

    _env_a, _env_b, coordinator = make_duel_envs(
        host=args.host,
        port=args.port,
        username_a=args.username_a,
        username_b=args.username_b,
        cache_stats=cache_stats,
        window_length=args.window_length,
        reward_weights_a=reward_weights_a,
        reward_weights_b=reward_weights_b,
        match_end=MatchEndConfig(),
        node_executable=args.node_executable,
        bridge_startup_timeout_s=args.bridge_startup_timeout_s,
        bridge_reset_timeout_s=args.bridge_reset_timeout_s,
        bridge_tick_timeout_s=args.bridge_tick_timeout_s,
        view_distance=args.view_distance,
        bridge_log_level=args.bridge_log_level,
    )

    episode_rewards_a: list[float] = []
    episode_rewards_b: list[float] = []
    wins_a = wins_b = draws = undecided = 0

    try:
        obs_a, obs_b = coordinator.reset()
        for episode in range(1, args.episodes + 1):
            done = False
            ep_reward_a = ep_reward_b = 0.0
            result_a = result_b = None
            while not done:
                stacked_obs = {key: np.stack([obs_a[key], obs_b[key]]) for key in obs_a}
                with torch.no_grad():
                    actions, _, _ = model.policy(
                        obs_as_tensor(stacked_obs, model.device),
                        deterministic=not args.stochastic,
                    )
                actions_np = np.clip(
                    actions.cpu().numpy(),
                    action_space.low,
                    action_space.high,
                )
                result_a, result_b = coordinator.advance(
                    flat_action_to_command(actions_np[0]),
                    flat_action_to_command(actions_np[1]),
                )
                ep_reward_a += result_a.reward
                ep_reward_b += result_b.reward
                done = result_a.terminated or result_a.truncated
                obs_a, obs_b = result_a.obs, result_b.obs

            assert result_a is not None
            own_a, opp_a = result_a.info["own_score"], result_a.info["opponent_score"]
            if own_a is not None and opp_a is not None and own_a != opp_a:
                if own_a > opp_a:
                    wins_a += 1
                else:
                    wins_b += 1
            elif own_a is not None and opp_a is not None:
                draws += 1
            else:
                undecided += 1
            episode_rewards_a.append(ep_reward_a)
            episode_rewards_b.append(ep_reward_b)
            logger.info(
                "Episode %d/%d: side_a_reward=%.3f side_b_reward=%.3f score=%s-%s%s",
                episode,
                args.episodes,
                ep_reward_a,
                ep_reward_b,
                own_a,
                opp_a,
                " (disconnected)" if result_a.info["disconnected"] else "",
            )
            if episode < args.episodes:
                obs_a, obs_b = coordinator.reset()
    finally:
        coordinator.close()

    n = len(episode_rewards_a)
    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": n,
        "side_a_win_rate": wins_a / n if n else None,
        "side_b_win_rate": wins_b / n if n else None,
        "draw_rate": draws / n if n else None,
        "undecided_rate": undecided / n if n else None,
        "avg_reward_side_a": mean(episode_rewards_a) if episode_rewards_a else None,
        "avg_reward_side_b": mean(episode_rewards_b) if episode_rewards_b else None,
        "avg_reward_combined": (
            mean(episode_rewards_a + episode_rewards_b) if episode_rewards_a else None
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
        logger.info("Wrote evaluation summary to %s", args.output)


if __name__ == "__main__":
    main()
