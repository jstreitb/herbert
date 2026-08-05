"""``python -m herbert_rl.train.train`` -- Hydra-driven PPO fine-tuning of a `/nn` checkpoint.

Requires two bot accounts able to connect to your own private, self-hosted Minecraft 1.8.9
server (see `rl/server/SETUP.md`) -- this spawns two real `/rl/bridge` Node.js processes and
drives a live, real-time Bridge duel; there is no synthetic/offline mode (unlike `rl.smoketest`,
which still needs a real server but skips the pretrained checkpoint).

Example::

    python -m herbert_rl.train.train \\
        pretrained_checkpoint_path=/path/to/nn/runs/gru_run1/2026-08-04_12-00-00/best.pt \\
        nn_cache_manifest_path=/path/to/nn/data/cache/<hash> \\
        env.host=192.168.1.50 env.port=25565

Every run writes to ``runs/{experiment_name}/{timestamp}/`` (same convention as `/nn`):
a snapshot of the resolved config (``.hydra/``), ``best.zip``/``last.zip`` SB3 model archives,
periodic ``checkpoint_update_{n}.zip`` snapshots, TensorBoard event files, and ``metrics.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.logger import configure as configure_sb3_logger

from herbert_rl.env.match_coordinator import MatchEndConfig
from herbert_rl.logging_utils import configure_logging
from herbert_rl.train.rollout import collect_rollout
from herbert_rl.train.setup import build_training_components

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """Resolve the ``device`` config value ("auto"/"cpu"/"cuda") to a concrete device string."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def run_training(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    """Execute a full RL fine-tuning run described by ``cfg``.

    Args:
        cfg: The fully-composed Hydra config (``conf/config.yaml`` + overrides).
        run_dir: Directory to write checkpoints, TensorBoard logs, and ``metrics.json`` into.

    Returns:
        The summary dict also written to ``run_dir / "metrics.json"``.
    """
    device = resolve_device(str(cfg.device))
    logger.info("Training on device=%s", device)

    env_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.env, resolve=True))
    reward_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.reward, resolve=True))
    ppo_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.ppo, resolve=True))
    num_updates = int(ppo_cfg.pop("num_updates"))
    n_steps = int(ppo_cfg["n_steps"])

    match_end_cfg = env_cfg.pop("match_end")
    match_end = MatchEndConfig(
        score_threshold=match_end_cfg["score_threshold"],
        chat_patterns=list(match_end_cfg["chat_patterns"]),
    )

    model, coordinator, obs_a, obs_b = build_training_components(
        host=env_cfg["host"],
        port=int(env_cfg["port"]),
        username_a=env_cfg["username_a"],
        username_b=env_cfg["username_b"],
        nn_cache_manifest_path=str(cfg.nn_cache_manifest_path),
        checkpoint_path=str(cfg.pretrained_checkpoint_path),
        fresh_model_cfg=None,
        window_length=int(env_cfg["window_length"]),
        view_distance=int(env_cfg["view_distance"]),
        node_executable=env_cfg["node_executable"],
        bridge_log_level=env_cfg["bridge_log_level"],
        bridge_startup_timeout_s=float(env_cfg["bridge_startup_timeout_s"]),
        bridge_reset_timeout_s=float(env_cfg["bridge_reset_timeout_s"]),
        bridge_tick_timeout_s=float(env_cfg["bridge_tick_timeout_s"]),
        reward_cfg=reward_cfg,
        match_end=match_end,
        ppo_kwargs=ppo_cfg,
        device=device,
        seed=int(cfg.seed),
    )

    sb3_logger = configure_sb3_logger(str(run_dir), ["stdout", "tensorboard"])
    model.set_logger(sb3_logger)

    total_timesteps_target = num_updates * n_steps * 2  # 2 sides per tick
    best_mean_episode_reward = float("-inf")
    history: list[dict[str, Any]] = []

    try:
        for update in range(1, num_updates + 1):
            result = collect_rollout(model, coordinator, obs_a, obs_b, n_steps)
            obs_a, obs_b = result.obs_a, result.obs_b

            model._update_current_progress_remaining(model.num_timesteps, total_timesteps_target)
            model.train()

            mean_episode_reward = None
            if result.episodes:
                mean_episode_reward = sum(e.total_reward for e in result.episodes) / len(
                    result.episodes
                )
                mean_episode_length = sum(e.length for e in result.episodes) / len(result.episodes)
                mean_goal_diff = sum(
                    (e.final_own_score or 0) - (e.final_opponent_score or 0)
                    for e in result.episodes
                ) / len(result.episodes)
            else:
                mean_episode_length = None
                mean_goal_diff = None

            if update % int(cfg.log_every_n_updates) == 0:
                model.logger.record("rollout/mean_reward_per_tick", result.mean_reward)
                for k, v in result.reward_breakdown_means.items():
                    model.logger.record(f"rollout/reward_{k}_per_tick", v)
                if mean_episode_reward is not None:
                    model.logger.record("rollout/ep_rew_mean", mean_episode_reward)
                    model.logger.record("rollout/ep_len_mean", mean_episode_length)
                    model.logger.record("rollout/goal_differential_mean", mean_goal_diff)
                model.logger.record("time/update", update)
                model.logger.dump(model.num_timesteps)
                logger.info(
                    "update %d/%d | timesteps=%d | mean_reward_per_tick=%.5f | episodes_completed=%d"
                    + (" | mean_ep_reward=%.4f" if mean_episode_reward is not None else ""),
                    *(
                        (
                            update,
                            num_updates,
                            model.num_timesteps,
                            result.mean_reward,
                            len(result.episodes),
                        )
                        + ((mean_episode_reward,) if mean_episode_reward is not None else ())
                    ),
                )

            model.save(str(run_dir / "last.zip"))
            if mean_episode_reward is not None and mean_episode_reward > best_mean_episode_reward:
                best_mean_episode_reward = mean_episode_reward
                model.save(str(run_dir / "best.zip"))
                logger.info(
                    "New best mean episode reward: %.4f (update %d)",
                    best_mean_episode_reward,
                    update,
                )
            if update % int(cfg.checkpoint_every_n_updates) == 0:
                model.save(str(run_dir / f"checkpoint_update_{update}.zip"))

            history.append(
                {
                    "update": update,
                    "timesteps": model.num_timesteps,
                    "mean_reward_per_tick": result.mean_reward,
                    "episodes_completed": len(result.episodes),
                    "mean_episode_reward": mean_episode_reward,
                    "mean_goal_differential": mean_goal_diff,
                }
            )
    finally:
        coordinator.close()

    summary = {
        "experiment_name": str(cfg.experiment_name),
        "updates_run": num_updates,
        "best_mean_episode_reward": (
            best_mean_episode_reward if best_mean_episode_reward != float("-inf") else None
        ),
        "history": history,
        "run_dir": str(run_dir),
        "pretrained_checkpoint_path": str(cfg.pretrained_checkpoint_path),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info("Training complete. Summary written to %s", run_dir / "metrics.json")
    return summary


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: resolve the run directory and delegate to :func:`run_training`."""
    configure_logging()
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.info("Hydra run dir: %s", run_dir)
    run_training(cfg, run_dir)


if __name__ == "__main__":
    main()
