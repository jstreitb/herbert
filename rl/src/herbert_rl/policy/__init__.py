# SPDX-License-Identifier: MIT
"""Policy layer: the `/nn`-derived backbone, checkpoint loading/adaptation, and the custom SB3 policy.

The custom `ActorCriticPolicy` that PPO trains -- see `sb3_policy.py` for the entry point.
"""

from __future__ import annotations
