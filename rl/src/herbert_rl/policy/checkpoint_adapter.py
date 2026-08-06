# SPDX-License-Identifier: MIT
"""Loads a `/nn` behavioral-cloning checkpoint into the RL backbone, without importing `herbert_nn`.

A `/nn` checkpoint (as written by `herbert_nn.training.checkpoint.save_checkpoint`) is a plain
``torch.save`` dict -- ``{"model_state_dict": ..., "model_cfg": ..., "data_meta": ..., ...}`` --
so it can be loaded and its tensors extracted with nothing more than `torch.load` plus the
architecturally-matching module tree in `policy/backbone.py`. No `herbert_nn` import required.

Two things happen here:

1. The encoder + GRU/MLP trunk weights load directly into `RLGRUBackbone`/`RLMLPBackbone` via
   `load_state_dict(strict=False)`, since the submodule names match exactly (see
   `backbone.py`'s docstring).
2. The pretrained `MouseHead`/`DiscreteHead`/`MovementHead` linear layers' weights are pulled
   out of the raw state dict *as tensors* (not reconstructed as modules) so
   `policy/sb3_policy.py` can splice them into specific rows of SB3's own `action_net` linear
   layer -- giving the aim (`d_yaw`/`d_pitch`), `jump`/`sneak`/`attack`/`place`, and
   `move_forward`/`strafe` action dimensions a BC-informed starting point, instead of PPO
   having to relearn "roughly point at the opponent," "occasionally jump," and "move toward
   the bridge" from scratch. Only the value head has no BC equivalent to inherit (behavioral
   cloning never estimated a value function) and starts randomly initialized.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from herbert_rl.policy.backbone import RLBackbone, RLDataMeta, build_backbone

logger = logging.getLogger(__name__)

#: Row layout of `sb3_policy.HerbertRLPolicy`'s 8-dim action_net -- see `env/action_wrapper.py`'s
#: flat encoding docstring for what each index means.
MOVEMENT_ACTION_NET_ROWS = slice(0, 2)  # move_forward, strafe
DISCRETE_ACTION_NET_ROWS = slice(2, 6)  # jump, sneak, attack, place
MOUSE_ACTION_NET_ROWS = slice(6, 8)  # d_yaw, d_pitch


@dataclass
class PretrainedHeads:
    """Raw pretrained `/nn` head weights, extracted for splicing into the PPO action head.

    ``None`` fields mean "no checkpoint was loaded" (see :func:`build_fresh_backbone`, used by
    `rl.smoketest`'s random-policy mode) -- the corresponding action dims stay randomly
    initialized rather than BC-informed.
    """

    mouse_weight: torch.Tensor | None
    mouse_bias: torch.Tensor | None
    discrete_weight: torch.Tensor | None
    discrete_bias: torch.Tensor | None
    movement_weight: torch.Tensor | None
    movement_bias: torch.Tensor | None


@dataclass
class LoadedBackbone:
    """Everything `policy/sb3_policy.py` needs to build the PPO policy's network."""

    backbone: RLBackbone
    trunk_dim: int
    model_family: str
    pretrained_heads: PretrainedHeads
    checkpoint_path: str | None


def load_nn_checkpoint(
    checkpoint_path: str | Path, map_location: str = "cpu"
) -> LoadedBackbone:
    """Load a `/nn` checkpoint and build the matching RL backbone with its weights.

    Args:
        checkpoint_path: Path to a `/nn` ``.pt`` checkpoint (e.g. ``runs/gru_run1/.../best.pt``).
        map_location: Passed to `torch.load`.

    Returns:
        A :class:`LoadedBackbone` with the backbone's weights loaded from the checkpoint and the
        pretrained `MouseHead`/`DiscreteHead` tensors available for splicing.
    """
    path = Path(checkpoint_path)
    logger.info("Loading /nn checkpoint from %s", path)
    checkpoint: dict[str, Any] = torch.load(
        path, map_location=map_location, weights_only=False
    )

    data_meta = RLDataMeta.from_checkpoint_data_meta(checkpoint["data_meta"])
    model_cfg = dict(checkpoint["model_cfg"])
    backbone = build_backbone(model_cfg, data_meta)

    state_dict: dict[str, torch.Tensor] = checkpoint["model_state_dict"]
    result = backbone.load_state_dict(state_dict, strict=False)
    # Expected unexpected keys: mouse_head.*, discrete_head.*, block_placement_head.* (handled
    # below / intentionally not part of the backbone). Anything else showing up here indicates a
    # drift between backbone.py and /nn's actual architecture -- log loudly either way.
    expected_unexpected_prefixes = (
        "mouse_head.",
        "discrete_head.",
        "block_placement_head.",
        "movement_head.",
    )
    surprising_unexpected = [
        k
        for k in result.unexpected_keys
        if not k.startswith(expected_unexpected_prefixes)
    ]
    if result.missing_keys:
        logger.warning(
            "Checkpoint load: %d backbone parameters had no matching checkpoint key (random "
            "init retained): %s",
            len(result.missing_keys),
            result.missing_keys,
        )
    if surprising_unexpected:
        logger.warning(
            "Checkpoint load: %d checkpoint keys did not match any backbone parameter (this "
            "usually means backbone.py has drifted from /nn's model architecture -- see "
            "backbone.py's module docstring): %s",
            len(surprising_unexpected),
            surprising_unexpected,
        )
    logger.info(
        "Loaded /nn %s backbone (%d params) from %s.",
        model_cfg["family"],
        sum(p.numel() for p in backbone.parameters()),
        path,
    )

    pretrained_heads = PretrainedHeads(
        mouse_weight=state_dict.get("mouse_head.linear.weight"),
        mouse_bias=state_dict.get("mouse_head.linear.bias"),
        discrete_weight=state_dict.get("discrete_head.linear.weight"),
        discrete_bias=state_dict.get("discrete_head.linear.bias"),
        movement_weight=state_dict.get("movement_head.linear.weight"),
        movement_bias=state_dict.get("movement_head.linear.bias"),
    )
    if (
        pretrained_heads.mouse_weight is None
        or pretrained_heads.discrete_weight is None
        or pretrained_heads.movement_weight is None
    ):
        logger.warning(
            "Checkpoint at %s is missing mouse_head/discrete_head/movement_head weights -- the "
            "corresponding PPO action dims will start randomly initialized instead of "
            "BC-informed. (movement_head is absent in checkpoints trained before /nn added "
            "MovementHead -- this is expected for old checkpoints, not an error.)",
            path,
        )

    return LoadedBackbone(
        backbone=backbone,
        trunk_dim=backbone.output_dim,
        model_family=model_cfg["family"],
        pretrained_heads=pretrained_heads,
        checkpoint_path=str(path),
    )


def build_fresh_backbone(model_cfg: dict, data_meta: RLDataMeta) -> LoadedBackbone:
    """Build a randomly-initialized backbone with no pretrained weights at all.

    Used only by `rl.smoketest` to validate the env/bridge/PPO wiring without requiring a `/nn`
    checkpoint to exist yet -- never use this for a real training run (see `rl/README.md`'s
    "managing expectations" section: skipping the BC warm start defeats the point of the RL
    fine-tuning phase).
    """
    backbone = build_backbone(model_cfg, data_meta)
    empty_heads = PretrainedHeads(
        mouse_weight=None,
        mouse_bias=None,
        discrete_weight=None,
        discrete_bias=None,
        movement_weight=None,
        movement_bias=None,
    )
    return LoadedBackbone(
        backbone=backbone,
        trunk_dim=backbone.output_dim,
        model_family=model_cfg["family"],
        pretrained_heads=empty_heads,
        checkpoint_path=None,
    )
