# SPDX-License-Identifier: MIT
r"""``python -m herbert_rl.train.smoketest`` -- end-to-end IPC loop validation against a real server.

**Unlike `/nn`'s smoke test (`herbert_nn.smoketest`), this one requires a real, reachable
Minecraft 1.8.9 server** (your own private one -- see `rl/server/SETUP.md`) and spawns two real
`/rl/bridge` Node.js processes that actually connect to it. There is no synthetic/offline mode:
`/nn`'s smoke test can fabricate shape-correct tensors because behavioral cloning trains on
static logged data, but `/rl`'s env has no meaning without a live bot-to-server connection to
step. What this smoke test *does* skip is the `/nn` checkpoint and PPO machinery entirely --
it drives both sides of a duel with uniformly random actions (`env.action_space.sample()`)
straight through `MatchCoordinator.advance()`, which is exactly the IPC path real training uses,
just with noise instead of policy-selected actions. If this passes, the env/bridge/IPC
plumbing is confirmed sound and any subsequent `rl.train` failure is a training-specific issue
(checkpoint loading, PPO hyperparameters), not a wiring issue.

Example::

    python -m herbert_rl.train.smoketest --host 192.168.1.50 --port 25565 \
        --nn-cache-manifest-path /path/to/nn/data/cache/<hash>
"""

from __future__ import annotations

import argparse
import logging
import time

from herbert_rl.env.herbert_bridge_env import make_duel_envs
from herbert_rl.env.ipc import action_dict_to_command
from herbert_rl.env.match_coordinator import MatchEndConfig
from herbert_rl.env.reward import RewardWeights
from herbert_rl.logging_utils import configure_logging
from herbert_rl.nn_cache import load_nn_cache_stats

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for this CLI."""
    parser = argparse.ArgumentParser(
        description="End-to-end smoke test of the herbert_rl env/bridge/IPC loop against a real "
        "Minecraft 1.8.9 server, using uniformly random actions (no /nn checkpoint, no PPO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", required=True, help="Your private Minecraft 1.8.9 server host."
    )
    parser.add_argument("--port", type=int, default=25565)
    parser.add_argument("--username-a", default="HerbertBot1")
    parser.add_argument("--username-b", default="HerbertBot2")
    parser.add_argument(
        "--nn-cache-manifest-path",
        required=True,
        help="Path to an /nn preprocessing cache's manifest.json (or its parent directory) -- "
        "even without a checkpoint, the observation space needs vocab sizes/block-grid shape "
        "from a real cache. Run `python -m herbert_nn.preprocess` once against any session data "
        "to produce one if you don't have one yet.",
    )
    parser.add_argument("--window-length", type=int, default=1)
    parser.add_argument("--view-distance", type=int, default=6)
    parser.add_argument("--node-executable", default="node")
    parser.add_argument("--bridge-log-level", default="info")
    parser.add_argument("--bridge-startup-timeout-s", type=float, default=60.0)
    parser.add_argument("--bridge-reset-timeout-s", type=float, default=120.0)
    parser.add_argument("--bridge-tick-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--num-episodes", type=int, default=2, help="Number of episodes to complete."
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=12000,
        help="Safety cap (10 minutes at 20Hz) on total ticks -- if this is hit before "
        "--num-episodes complete, match-end detection is likely misconfigured (see "
        "rl/conf/env/default.yaml and rl/server/SETUP.md).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(level=getattr(logging, args.log_level.upper()))

    cache_stats = load_nn_cache_stats(args.nn_cache_manifest_path)
    reward_weights_a = RewardWeights()
    reward_weights_b = RewardWeights(
        own_goal_forward_sign=-reward_weights_a.own_goal_forward_sign
    )
    match_end = MatchEndConfig()

    env_a, env_b, coordinator = make_duel_envs(
        host=args.host,
        port=args.port,
        username_a=args.username_a,
        username_b=args.username_b,
        cache_stats=cache_stats,
        window_length=args.window_length,
        reward_weights_a=reward_weights_a,
        reward_weights_b=reward_weights_b,
        match_end=match_end,
        node_executable=args.node_executable,
        bridge_startup_timeout_s=args.bridge_startup_timeout_s,
        bridge_reset_timeout_s=args.bridge_reset_timeout_s,
        bridge_tick_timeout_s=args.bridge_tick_timeout_s,
        view_distance=args.view_distance,
        bridge_log_level=args.bridge_log_level,
    )
    env_a.action_space.seed(args.seed)
    env_b.action_space.seed(args.seed + 1)

    start = time.perf_counter()
    total_ticks = 0
    episodes_completed = 0
    try:
        logger.info("Requesting initial reset...")
        coordinator.reset()
        while episodes_completed < args.num_episodes:
            action_a = action_dict_to_command(env_a.action_space.sample())
            action_b = action_dict_to_command(env_b.action_space.sample())
            result_a, result_b = coordinator.advance(action_a, action_b)
            total_ticks += 1

            if result_a.terminated or result_a.truncated:
                episodes_completed += 1
                logger.info(
                    "Episode %d/%d finished (%d total ticks so far; own=%s opp=%s disconnected=%s).",
                    episodes_completed,
                    args.num_episodes,
                    total_ticks,
                    result_a.info["own_score"],
                    result_a.info["opponent_score"],
                    result_a.info["disconnected"],
                )
                if episodes_completed < args.num_episodes:
                    coordinator.reset()

            if total_ticks > args.max_ticks:
                raise RuntimeError(
                    f"Smoke test exceeded --max-ticks={args.max_ticks} without completing "
                    f"{args.num_episodes} episodes ({episodes_completed} done). Match-end "
                    "detection is likely misconfigured for your server plugin -- check "
                    "rl/conf/env/default.yaml's match_end.chat_patterns/score_threshold against "
                    "your server's actual chat output (see rl/server/SETUP.md)."
                )
    finally:
        coordinator.close()

    elapsed = time.perf_counter() - start
    logger.info(
        "SMOKE TEST PASSED: %d episodes completed over %d ticks in %.1fs (%.2f ticks/s).",
        episodes_completed,
        total_ticks,
        elapsed,
        total_ticks / elapsed if elapsed > 0 else float("inf"),
    )


if __name__ == "__main__":
    main()
