# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""B2W specific configuration for navigation environment."""

import os

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from isaaclab_nav_task.navigation.navigation_env_cfg import NavigationEnvCfg
import isaaclab_nav_task.navigation.mdp as mdp

from isaaclab_nav_task.navigation.assets import B2W_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR  # isort: skip


LEG_JOINT_NAMES = [".*hip_joint", ".*thigh_joint", ".*calf_joint"]
WHEEL_JOINT_NAMES = [".*foot_joint"]

@configclass
class B2WNavigationEnvCfg(NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator
        from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config

        initialize_depth_noise_generator(robot_name="b2w", use_jit_precompiled=False)

        camera_config = get_camera_config("b2w")
        CAMERA_RESOLUTION = camera_config.resolution

        self.scene.robot = B2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.raycast_camera.offset.pos = (0.387, 0.0, 0.28)
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        self.terminations.base_contact.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*hip", ".*thigh"]), "threshold": 1.0}

        self.actions.velocity_command.low_level_position_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"], scale=0.5, use_default_offset=True)
        self.actions.velocity_command.low_level_velocity_action = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*foot_joint"], scale=5.0, use_default_offset=True)
        self.actions.velocity_command.low_level_policy_file = os.path.join(ISAACLAB_NAV_TASKS_ASSETS_DIR, "Policies", "locomotion", "b2w", "policy_b2w_new_2.pt")

        self.rewards.joint_acc_l2_joint.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES+WHEEL_JOINT_NAMES)}

        self.terminations.base_contact.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*hip", ".*thigh"]), "threshold": 1.0}

        self.events.randomize_low_pass_filter_alpha.params = {
            "alpha_range": (0.1, 0.6),
            "action_term": "velocity_command",
            "per_dimension": True,
            "alpha_range_vx": (0.1, 0.6),
            "alpha_range_vy": (0.1, 0.6),
            "alpha_range_omega": (0.1, 0.6),
        }

        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False

@configclass
class B2WNavigationEnvCfg_DEV(B2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False

@configclass
class B2WNavigationEnvCfg_PLAY(B2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

            # ---- PLAY_DIFFICULTY="lo,hi": difficulty band ----
            lo, hi = 0.5, 1.0
            _play_diff = os.environ.get("PLAY_DIFFICULTY", "").strip()
            if _play_diff:
                _parts = _play_diff.replace(" ", "").split(",")
                lo, hi = float(_parts[0]), float(_parts[1])
            self.scene.terrain.terrain_generator.difficulty_range = [lo, hi]
            self.scene.terrain.terrain_generator.curriculum = False
            print(f"[B2W PLAY] terrain difficulty_range = [{lo}, {hi}] (curriculum off)")

            sub_terrains = self.scene.terrain.terrain_generator.sub_terrains

            # PLAY_MAZE_ONLY=1: 100% maze sub-terrain
            _maze_only = os.environ.get("PLAY_MAZE_ONLY", "").strip().lower() in ("1", "true", "yes")
            if _maze_only and "maze" in sub_terrains:
                for name in list(sub_terrains.keys()):
                    sub_terrains[name].proportion = 1.0 if name == "maze" else 0.0
                print(f"[B2W PLAY] PLAY_MAZE_ONLY=1 -> 100% maze sub-terrain")

            # PLAY_SUBTERRAIN_MIX="maze=1,non_maze=0,pits=0"
            _mix = os.environ.get("PLAY_SUBTERRAIN_MIX", "").strip()
            if _mix:
                overrides = {}
                for part in _mix.split(","):
                    if "=" not in part:
                        continue
                    k, v = part.split("=", 1)
                    overrides[k.strip()] = float(v.strip())
                for name, prop in overrides.items():
                    if name in sub_terrains:
                        sub_terrains[name].proportion = prop
                final = {n: sub_terrains[n].proportion for n in sub_terrains}
                print(f"[B2W PLAY] PLAY_SUBTERRAIN_MIX -> {final}")

            # PLAY_CELL_SIZE: meters per maze cell
            _play_cell = os.environ.get("PLAY_CELL_SIZE", "").strip()
            if _play_cell:
                new_cell = float(_play_cell)
                for cfg_sub in sub_terrains.values():
                    cfg_sub.cell_size = new_cell
                tile_m = 15 * new_cell
                self.scene.terrain.terrain_generator.size = (tile_m, tile_m)
                print(f"[B2W PLAY] PLAY_CELL_SIZE={new_cell}m -> tile={tile_m}m")

            # PLAY_CLASSIC_MAZE=1: paper-style uniform walls
            _classic = os.environ.get("PLAY_CLASSIC_MAZE", "").strip().lower() in ("1", "true", "yes")
            if _classic and "maze" in sub_terrains:
                sub_terrains["maze"].randomize_wall = False
                sub_terrains["maze"].random_wall_ratio = 0.0
                print(f"[B2W PLAY] PLAY_CLASSIC_MAZE=1 -> classic uniform walls in maze")

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
