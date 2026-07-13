# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Configuration for the Unitree Go2 robot (reuses IsaacLab built-in USD).

* :obj:`GO2_CFG`: Unitree Go2 quadruped (12 leg joints, no wheels).
"""

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

__all__ = ["GO2_CFG"]

# Reuse the upstream IsaacLab Go2 articulation cfg verbatim. The locomotion
# checkpoint we converted (policy_go2_jit.pt) was trained on this exact USD
# (Isaac-Velocity-Flat-Unitree-Go2-v0), so joint order and joint count match.
GO2_CFG = UNITREE_GO2_CFG.copy()
"""Configuration of Unitree Go2 robot for navigation tasks."""
