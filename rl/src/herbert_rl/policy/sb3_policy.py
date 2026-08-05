"""`HerbertRLPolicy`: a custom `stable_baselines3` `ActorCriticPolicy` built on the `/nn` backbone.

Architecture, mapped onto SB3's `ActorCriticPolicy` building blocks:

- **features_extractor** (`HerbertBackboneExtractor`) = the `/nn`-pretrained
  `RLGRUBackbone`/`RLMLPBackbone` (encoder + GRU-or-MLP trunk). Shared between actor and critic
  (`share_features_extractor=True`, SB3's default) -- exactly the "add a value head on top of
  the existing policy network" the task asks for: the trunk is shared, only `action_net` and
  `value_net` (the two linear heads SB3 builds on top) are network-specific.
- **mlp_extractor**: left as SB3's default `net_arch=[]` (identity), so the actor/critic latent
  fed to `action_net`/`value_net` is exactly the backbone's `trunk_dim` output -- required so the
  pretrained `MouseHead`/`DiscreteHead` weight *rows* (each shaped `[out_dim, trunk_dim]`) can be
  spliced directly into `action_net.weight` (shaped `[8, trunk_dim]`) without any dimension
  mismatch.
- **action_net**: SB3's standard `Box(8,)` Gaussian mean layer (see `env/action_wrapper.py` for
  the 8-dim encoding). Rows 2-5 (jump/sneak/attack/place) and 6-7 (d_yaw/d_pitch) are overwritten
  with the pretrained `DiscreteHead`/`MouseHead` weights after construction; rows 0-1
  (move_forward/strafe) are left at SB3's default init, since `/nn` never modeled movement (see
  `nn/README.md`'s "Known limitations").
- **value_net**: SB3's standard fresh `Linear(trunk_dim, 1)` -- the new value head, with no BC
  equivalent to inherit (behavioral cloning never estimated a value function).

**Why `ortho_init=False` is forced:** SB3's default `ActorCriticPolicy._build()` re-initializes
every submodule (including the features extractor!) with orthogonal init when `ortho_init=True`
(the SB3 default) -- which would silently discard the pretrained backbone weights we just loaded.
This policy always disables it.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from herbert_rl.policy.checkpoint_adapter import (
    DISCRETE_ACTION_NET_ROWS,
    MOUSE_ACTION_NET_ROWS,
    LoadedBackbone,
)

logger = logging.getLogger(__name__)


class HerbertBackboneExtractor(BaseFeaturesExtractor):
    """Wraps the `/nn`-derived backbone as an SB3 `BaseFeaturesExtractor` over a Dict obs space.

    SB3's default observation preprocessing (`stable_baselines3.common.preprocessing.preprocess_obs`)
    one-hot-encodes `spaces.MultiDiscrete` leaves, which would break the backbone's `nn.Embedding`
    lookups (they need raw integer indices, not one-hot vectors). To avoid that, `env/spaces.py`
    deliberately types every categorical observation field as a float32 `Box` holding integer
    values -- SB3's preprocessing passes `Box` observations through as plain floats, and this
    extractor casts them back to `.long()` right before the embedding lookups.
    """

    def __init__(self, observation_space: spaces.Dict, backbone: nn.Module, trunk_dim: int) -> None:
        super().__init__(observation_space, features_dim=trunk_dim)
        self.backbone = backbone

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = {
            "continuous": observations["continuous"],
            "block_grid_cells": observations["block_grid_cells"].long(),
            "hotbar_slot_index": observations["hotbar_slot_index"].long(),
            "hotbar_item_type": observations["hotbar_item_type"].long(),
            "opponent_held_item_category": observations["opponent_held_item_category"].long(),
            "match_kit_type": observations["match_kit_type"].long(),
        }
        return self.backbone(batch)


class HerbertRLPolicy(ActorCriticPolicy):
    """SB3 `ActorCriticPolicy` wrapping a `/nn`-pretrained backbone, for use as `PPO(policy=...)`.

    Must be constructed with ``policy_kwargs={"loaded_backbone": <LoadedBackbone>}`` (see
    `policy/checkpoint_adapter.py::load_nn_checkpoint` / `build_fresh_backbone`) -- `train/train.py`
    and `train/smoketest.py` both do this via `functools.partial` before handing the policy class
    to `stable_baselines3.PPO`.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        loaded_backbone: LoadedBackbone,
        **kwargs: Any,
    ) -> None:
        self._loaded_backbone = loaded_backbone
        kwargs.setdefault("net_arch", [])
        kwargs.setdefault("normalize_images", False)
        kwargs["ortho_init"] = False  # see module docstring -- must not clobber pretrained weights
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=HerbertBackboneExtractor,
            features_extractor_kwargs={
                "backbone": loaded_backbone.backbone,
                "trunk_dim": loaded_backbone.trunk_dim,
            },
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self._splice_pretrained_heads()

    def _splice_pretrained_heads(self) -> None:
        """Overwrite the relevant `action_net` rows with pretrained `/nn` head weights."""
        heads = self._loaded_backbone.pretrained_heads
        assert isinstance(self.action_net, nn.Linear)
        with torch.no_grad():
            if heads.discrete_weight is not None and heads.discrete_bias is not None:
                self.action_net.weight[DISCRETE_ACTION_NET_ROWS].copy_(heads.discrete_weight)
                self.action_net.bias[DISCRETE_ACTION_NET_ROWS].copy_(heads.discrete_bias)
                logger.info(
                    "Spliced pretrained DiscreteHead weights into action_net rows %s.",
                    DISCRETE_ACTION_NET_ROWS,
                )
            if heads.mouse_weight is not None and heads.mouse_bias is not None:
                self.action_net.weight[MOUSE_ACTION_NET_ROWS].copy_(heads.mouse_weight)
                self.action_net.bias[MOUSE_ACTION_NET_ROWS].copy_(heads.mouse_bias)
                logger.info(
                    "Spliced pretrained MouseHead weights into action_net rows %s.",
                    MOUSE_ACTION_NET_ROWS,
                )
