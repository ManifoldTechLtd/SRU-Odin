# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT


from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg
from .goal_commands import RobotNavigationGoalCommand


"""
Base command generator.
"""

@configclass
class RobotNavigationGoalCommandCfg(CommandTermCfg):
    """Configuration for the robot goal command generator."""

    class_type: type = RobotNavigationGoalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    robot_to_goal_line_vis: bool = True
    """If true, visualize the line from the robot to the goal."""

    goal_height_offset_range: tuple[float, float] = (0.2, 0.8)
    """Range (min, max) in meters for the random goal height offset above the ground.

    The goal cube is placed at ``terrain_ground_height + U(min, max)``. The default
    (0.2, 0.8) is tuned for the B2W body height (~0.5-0.6 m). For shorter robots
    such as the Go2 (standing height ~0.3 m), lower this range (e.g. (0.1, 0.4)) so
    the goal marker sits near the robot body. Note: success/termination is purely
    horizontal (xy), so this offset only affects visualization and the minor 3D
    ``reach_goal_xy_soft`` reward term.
    """

