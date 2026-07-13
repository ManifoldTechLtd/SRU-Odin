# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Unitree Go2 specific configuration for navigation environment."""

import os

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from isaaclab_nav_task.navigation.navigation_env_cfg import NavigationEnvCfg
import isaaclab_nav_task.navigation.mdp as mdp

from isaaclab_nav_task.navigation.assets import GO2_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR  # isort: skip


# Go2 has 12 leg joints (no wheels). Joint names use the underscore convention
# from the IsaacLab Go2 USD: FL_hip_joint, FL_thigh_joint, FL_calf_joint, ...
LEG_JOINT_NAMES = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]


@configclass
class Go2NavigationEnvCfg(NavigationEnvCfg):
    """Navigation env for Unitree Go2 (legs-only quadruped).

    The locomotion checkpoint expected at
    ``assets/data/Policies/locomotion/go2/policy_go2_jit.pt`` was converted from
    a rsl_rl checkpoint trained on Isaac-Velocity-Flat-Unitree-Go2-v0:
      - input  : 48-dim observation (matches sru-navigation-sim's
                 LowLevelPolicyCfg)
      - output : 12-dim joint position deltas (one per leg joint)
    """

    def __post_init__(self):
        super().__post_init__()

        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator
        from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config

        initialize_depth_noise_generator(robot_name="go2", use_jit_precompiled=False)
        camera_config = get_camera_config("go2")
        # Camera resolution is consumed implicitly through the encoder; sru-nav
        # always feeds (64, 40) into the VAE regardless of robot.
        _ = camera_config

        # ---- Robot ----
        self.scene.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Go2's body link is named "base" in the IsaacLab USD (not "base_link"
        # like B2W).
        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # Real-robot Odin1 mount on Go2: 25.8 cm forward, 15.4 cm above body
        # center, camera looking straight ahead (no pitch).
        # NOTE: The extrinsic Tcl from the Odin1 calibration is camera-from-lidar,
        # not camera-from-base. Until we have T_base_lidar (lidar pose in Go2 base
        # frame), keep this measured mount offset rather than chaining Tcl.
        self.scene.raycast_camera.offset.pos = (0.258, 0.0, 0.154)
        self.scene.raycast_camera.offset.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # ---- Camera intrinsics: Odin1 (LiDAR-aligned depth) ----
        # Real Odin1 @ 1600x1296:
        #   fx = 737.357, fy = 737.292, cx = 794.372, cy = 666.259
        #   hFOV ~94.67 deg, vFOV ~82.65 deg
        # To preserve the FOV while keeping the upstream downsample_factor=3
        # (raycast at 192x120 then downsample to the VAE-required 64x40), we
        # scale the intrinsics linearly:
        #   fx_192 = 737.357 * 192/1600 = 88.48 px
        #   fy_120 = 737.292 * 120/1296 = 68.27 px
        #   cx_192 = 794.372 * 192/1600 = 95.32 px
        #   cy_120 = 666.259 * 120/1296 = 61.69 px
        # Distortion (k2..k7) is ignored: raycast uses an ideal pinhole. The
        # real-side pipeline must undistort before feeding the network.
        # max_distance raised slightly because Odin1's wider FOV picks up more
        # nearby clutter; keep 11 m to match upstream.
        from isaaclab.sensors import patterns
        self.scene.raycast_camera.max_distance = 11.0
        self.scene.raycast_camera.pattern_cfg = patterns.PinholeCameraPatternCfg.from_ros_camera_info(
            fx=88.48,
            fy=68.27,
            cx=95.32,
            cy=61.69,
            width=192,
            height=120,
            downsample_factor=3,  # 192x120 -> 64x40 to match VAE input
        )

        # ---- Goal placement ----
        # The default goal height offset (0.2-0.8 m above ground) is tuned for
        # B2W's body height (~0.5-0.6 m). Go2 stands at ~0.3 m, so lower the
        # range to (0.1, 0.4) m to keep the goal marker near the Go2 body.
        # NOTE: success/termination is purely horizontal (xy), so this is mostly
        # a visual fix + removes the small constant z-error in the 3D
        # reach_goal_xy_soft reward term.
        self.commands.robot_goal.goal_height_offset_range = (0.1, 0.4)

        # ---- Termination ----
        # Penalize/terminate if the body, hips, thighs, or head links hit
        # something. Go2's main body is "base"; the front "head" is split into
        # two extra rigid links ("Head_upper" / "Head_lower" in the upstream
        # Unitree Go2 USD) which stick forward ~10cm past the base. Without
        # those head links in the watch-list, GUI playback shows the snout
        # visually crashing into walls while the base contact sensor stays
        # silent and the episode does not terminate. Including them closes
        # that loophole.
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base", ".*_hip", ".*_thigh", "Head_upper", "Head_lower"],
            ),
            "threshold": 1.0,
        }

        # Diagnostic-only: tighten `terrain_fall` so it actually fires when the
        # robot drops into the pit terrain. The base default of -2.0m never
        # triggered for Go2 (standing base_z ~+0.4m, typical pit depth 0.75-2.25m
        # -> body z bottoms out around -1.1m, well above -2.0m), making
        # `Episode_Termination/terrain_fall` permanently 0 across phases 1-5.
        # -0.3m means: body must be ~0.7m below normal standing height to fire,
        # which cleanly captures real pit falls without flagging walking dips.
        # NOTE: `terrain_fall` is `time_out=True`, so it does NOT trigger the
        # -50 `episode_termination` penalty -- this is purely a metric fix.
        self.terminations.terrain_fall.params = {"fall_height_threshold": -0.3}

        # ---- Action interface (legs only, no wheels) ----
        self.actions.velocity_command.low_level_position_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEG_JOINT_NAMES,
            scale=0.25,                # MUST match the scale used to train policy_go2_jit.pt
            use_default_offset=True,
        )
        # Disable the velocity-action branch entirely (patched in
        # navigation_se2_actions.py to be optional).
        self.actions.velocity_command.low_level_velocity_action = None
        self.actions.velocity_command.low_level_policy_file = os.path.join(
            ISAACLAB_NAV_TASKS_ASSETS_DIR,
            "Policies", "locomotion", "go2", "policy_go2_jit.pt",
        )
        # SE2 command space scale: Go2 cannot match B2W's wheeled top speed,
        # so we shrink the SRU command range to what Go2 can actually track.
        # The high-level SRU policy was trained on B2W with [1.0, 1.0, 1.0]; we
        # shrink each axis so the same network output produces feasible Go2
        # commands.
        self.actions.velocity_command.scale = [0.6, 0.3, 0.7]

        # ---- Reward shaping ----
        self.rewards.joint_acc_l2_joint.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        }
        # Keep the B2W default termination penalty (-50.0). G0-G2 confirmed the
        # Go2 locomotion JIT is stable enough that we do not need the relaxed
        # -25 hedge anymore; restoring -50 makes wall hits / falls genuinely
        # costly so the policy stops trading collisions for goal reward.
        # (Only fires on `base_contact` + `large_pitch_angle`; `time_out`,
        # `at_goal`, `terrain_fall` are all marked time_out=True and don't
        # trigger this term.)
        self.rewards.episode_termination.weight = -50.0

        # Forward-facing depth camera only senses obstacles ahead of the robot.
        # In G2.0 the agent learned to walk *backwards* (vx < 0) to hide obstacles
        # from the camera and bypass forward-collision termination. Enable the
        # backward-movement penalty (default weight is 0.0 in the base cfg).
        # NOTE: the base CurriculumCfg disables this penalty after 500 global
        # steps; if backward-walking re-emerges late in training, either delete
        # ``curriculum.disable_backward_penalty`` or lift its ``disable_after_steps``.
        self.rewards.backward_movement_penalty.weight = -1.0

        # ---- Domain randomization ----
        # Go2 responds slower than wheels; widen the low-pass filter range.
        self.events.randomize_low_pass_filter_alpha.params = {
            "alpha_range": (0.3, 0.7),
            "action_term": "velocity_command",
            "per_dimension": True,
            "alpha_range_vx": (0.3, 0.7),
            "alpha_range_vy": (0.3, 0.7),
            "alpha_range_omega": (0.3, 0.7),
        }

        # ---- Terrain ----
        # The Go2 locomotion .pt is flat-trained, so start with easier terrain.
        # Defaults assume RESUMING a non-trivial ckpt (max_init level 5, band
        # [0.3, 0.8]). For a COLD START set GO2_DIFFICULTY="0.0,0.4" (and the
        # env var GO2_INIT_LEVEL=0 if you want everyone to start at the easiest
        # row), otherwise the policy gets dropped into mid-difficulty terrain
        # and stalls. Curriculum stays ON in both cases so envs auto-promote
        # toward the high end as they succeed.
        lo, hi = 0.3, 0.8
        _diff = os.environ.get("GO2_DIFFICULTY", "").strip()
        if _diff:
            _parts = _diff.replace(" ", "").split(",")
            lo, hi = float(_parts[0]), float(_parts[1])
        init_level = 5
        _lvl = os.environ.get("GO2_INIT_LEVEL", "").strip()
        if _lvl:
            init_level = int(_lvl)
        self.scene.terrain.max_init_terrain_level = init_level
        self.scene.terrain.terrain_generator.difficulty_range = [lo, hi]
        self.scene.terrain.terrain_generator.curriculum = True
        print(f"[Go2 Mixed] difficulty=[{lo}, {hi}], max_init_terrain_level={init_level} (curriculum on)")

        # Drop the dedicated stairs sub-terrain for Go2: the locomotion JIT is
        # trained on flat ground only, climbing stairs is unreliable, and the
        # platform_mask on stair tops would otherwise bias goal sampling
        # towards stair platforms (see PositionSampler.platform_repeat_count=10
        # in mdp/navigation/goal_commands.py).
        # NOTE: ``add_stairs_to_maze`` flag exists in HfMazeTerrainCfg but is
        # never consumed by hf_terrains_maze.py, so the only real stair source
        # is the ``"stairs"`` sub-terrain. We rebalance the remaining 3 to keep
        # roughly the original maze:non_maze:pits ratio (3:2:2 -> ~0.43/0.29/0.29).
        sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
        if "stairs" in sub_terrains:
            del sub_terrains["stairs"]
        if "maze" in sub_terrains:
            sub_terrains["maze"].proportion = 0.43
        if "non_maze" in sub_terrains:
            sub_terrains["non_maze"].proportion = 0.29
        if "pits" in sub_terrains:
            sub_terrains["pits"].proportion = 0.28

        # ---- Shrink tile size for 12 GB VRAM ----
        # Defaults: 30m x 30m tiles with horizontal_scale=0.1 -> 300x300 cells
        # per tile heightfield, x 6 rows x 30 cols x 3 sub-terrains = ~16M
        # heightfield cells + 3 sub-terrain collision meshes. On a 12 GB 5070
        # that OOMs above ~768 envs. Halve cell_size to 1.0m (matches what the
        # PureMaze runs used, where 1024 envs ran stably): tile shrinks to
        # 15m x 15m, heightfield drops 4x to ~4M cells. Override via env var
        # GO2_CELL_SIZE if you want to go back to the paper-default 2.0m.
        _cell = os.environ.get("GO2_CELL_SIZE", "1.0").strip()
        if _cell:
            new_cell = float(_cell)
            for cfg_sub in sub_terrains.values():
                cfg_sub.cell_size = new_cell
            # grid_size stays (15, 15) -> tile_size = 15 * cell_size meters.
            tile_m = 15 * new_cell
            self.scene.terrain.terrain_generator.size = (tile_m, tile_m)
            print(f"[Go2 Mixed] cell_size={new_cell}m -> tile={tile_m}m "
                  f"(set GO2_CELL_SIZE=2.0 to restore paper defaults)")


@configclass
class Go2NavigationEnvCfg_DEV(Go2NavigationEnvCfg):
    """Development configuration with smaller terrain and lower difficulty."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 30
        # The Go2 locomotion JIT is flat-trained. Starting at difficulty 0.3-0.8
        # with no curriculum produced ~16% tip-overs (large_pitch_angle) and
        # stalled success at ~0.5. Start from the easiest terrain and let the
        # curriculum ramp difficulty up as the policy succeeds.
        #
        # Phase 1/2 trained on [0.0, 0.4] (success ~0.86, depth confirmed used
        # via ablation). Phase 3 raises the band so the avoidance behaviour
        # generalises to denser obstacles / deeper pits. Control the band per
        # run with DEV_DIFFICULTY="lo,hi" (default stays [0.0, 0.4]); e.g.
        #   DEV_DIFFICULTY="0.2,0.6"  (recommended Phase 3 bridge)
        #   DEV_DIFFICULTY="0.3,0.8"  (Phase 4, near paper level)
        # max_init_terrain_level stays 0 so a resumed policy re-enters at the
        # easy end of the new band and the curriculum re-ramps from there.
        lo, hi = 0.0, 0.4
        _dev_diff = os.environ.get("DEV_DIFFICULTY", "").strip()
        if _dev_diff:
            _parts = _dev_diff.replace(" ", "").split(",")
            lo, hi = float(_parts[0]), float(_parts[1])
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [lo, hi]
        self.scene.terrain.terrain_generator.curriculum = True
        print(f"[Go2 DEV] terrain difficulty_range = [{lo}, {hi}] (curriculum on)")


@configclass
class Go2NavigationEnvCfg_PureMaze(Go2NavigationEnvCfg):
    """Pure-maze training configuration.

    All tiles are paper-style mazes:
      * 100% ``maze`` sub-terrain (no ``non_maze`` / ``pits`` sub-terrains).
      * Walls are uniform full-height rectangles (no randomized pillar/bar/cross
        obstacles inserted into the maze pattern).
      * Curriculum on, ramps difficulty from open to dense corridors.

    Env-var overrides:
      * ``PUREMAZE_DIFFICULTY="lo,hi"``: difficulty band (default ``"0.0,1.0"``;
        higher band -> denser corridors / more dead-ends).
      * ``PUREMAZE_CELL_SIZE``: meters per maze cell (default 2.0). Lowering
        tightens corridors; tile size scales accordingly so total terrain area
        stays the same per tile in cells.
    """

    def __post_init__(self):
        super().__post_init__()

        sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
        # Lock to 100% maze, paper-style walls.
        for name in list(sub_terrains.keys()):
            sub_terrains[name].proportion = 1.0 if name == "maze" else 0.0
        if "maze" in sub_terrains:
            sub_terrains["maze"].randomize_wall = False
            sub_terrains["maze"].random_wall_ratio = 0.0

        # Difficulty range (curriculum on, ramps from easy to hard).
        lo, hi = 0.0, 1.0
        _diff = os.environ.get("PUREMAZE_DIFFICULTY", "").strip()
        if _diff:
            _parts = _diff.replace(" ", "").split(",")
            lo, hi = float(_parts[0]), float(_parts[1])
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [lo, hi]
        self.scene.terrain.terrain_generator.curriculum = True

        # Cell size (corridor width).
        _cell = os.environ.get("PUREMAZE_CELL_SIZE", "").strip()
        if _cell:
            new_cell = float(_cell)
            for cfg_sub in sub_terrains.values():
                cfg_sub.cell_size = new_cell
            tile_m = 15 * new_cell
            self.scene.terrain.terrain_generator.size = (tile_m, tile_m)
            print(f"[Go2 PureMaze] cell_size={new_cell}m -> tile={tile_m}m, "
                  f"corridor ~{new_cell * 0.6:.2f}-{new_cell * 0.9:.2f}m")

        # Terrain grid dimensions. Defaults are compact (3×10=30 tiles) to save
        # VRAM; the original 6×30=180 tiles wasted mesh memory when most tiles
        # were never visited (especially with few envs / play). Override via
        # PUREMAZE_NUM_ROWS / PUREMAZE_NUM_COLS for large-scale training.
        _rows = int(os.environ.get("PUREMAZE_NUM_ROWS", "3"))
        _cols = int(os.environ.get("PUREMAZE_NUM_COLS", "10"))
        self.scene.terrain.terrain_generator.num_rows = _rows
        self.scene.terrain.terrain_generator.num_cols = _cols

        print(f"[Go2 PureMaze] 100% maze sub-terrain, classic walls, "
              f"difficulty=[{lo}, {hi}] (curriculum on), "
              f"grid={_rows}x{_cols}={_rows*_cols} tiles")


@configclass
class Go2NavigationEnvCfg_PLAY(Go2NavigationEnvCfg):
    """Evaluation/visualization configuration."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

            # ---- Evaluate on the SAME difficulty the policy was trained on ----
            # The base Go2NavigationEnvCfg uses difficulty_range=[0.3, 0.8], but
            # the Dev curriculum (what model_*.pt was actually trained on) ran on
            # [0.0, 0.4]. Replaying on [0.3, 0.8] shows the policy unseen-hard
            # terrain and makes it look like it "can't avoid" obstacles. Default
            # the PLAY difficulty to the training band for an honest read; allow
            # an explicit stress-test override via PLAY_DIFFICULTY="lo,hi".
            lo, hi = 0.0, 0.4
            _play_diff = os.environ.get("PLAY_DIFFICULTY", "").strip()
            if _play_diff:
                _parts = _play_diff.replace(" ", "").split(",")
                lo, hi = float(_parts[0]), float(_parts[1])
            self.scene.terrain.terrain_generator.difficulty_range = [lo, hi]
            # Fixed difficulty band for evaluation (no curriculum ramping).
            self.scene.terrain.terrain_generator.curriculum = False
            print(f"[Go2 PLAY] terrain difficulty_range = [{lo}, {hi}] (curriculum off)")

            # ---- Stress-test knobs for narrow-passage / dense-scene eval ----
            # PLAY_CELL_SIZE: meters per maze cell (default 2.0). Lowering this
            # tightens corridor width without changing tile size:
            #   2.0m  -> 1.7m-2.0m open corridors (default; very wide for Go2)
            #   1.5m  -> 1.0m-1.3m corridors (Go2 fits comfortably with margin)
            #   1.2m  -> 0.7m-0.9m corridors (tight, real perception test)
            #   1.0m  -> 0.5m-0.7m corridors (Go2 body width ~0.3m -> very tight)
            # NOTE: grid_size stays (15,15) so tile size shrinks proportionally
            # (30m -> 15m at cell_size=1.0). To preserve outer terrain footprint,
            # the tile size is rescaled below.
            _play_cell = os.environ.get("PLAY_CELL_SIZE", "").strip()
            # PLAY_MAZE_ONLY=1: force 100% maze sub-terrain (drop pits/non_maze)
            # so every tile is a corridor navigation challenge -- no easy open
            # tiles diluting the sample.
            _maze_only = os.environ.get("PLAY_MAZE_ONLY", "").strip().lower() in ("1", "true", "yes")

            sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
            if _maze_only and "maze" in sub_terrains:
                for name in list(sub_terrains.keys()):
                    sub_terrains[name].proportion = 1.0 if name == "maze" else 0.0
                print(f"[Go2 PLAY] PLAY_MAZE_ONLY=1 -> 100% maze sub-terrain")

            # PLAY_SUBTERRAIN_MIX="maze=0.5,non_maze=0.3,pits=0.2"
            # Override any subset of sub-terrain proportions. Names not listed
            # are left at their training-cfg default. Values do NOT need to sum
            # to 1.0 (IsaacLab normalizes them internally before sampling).
            # Setting a value to 0 effectively disables that sub-terrain.
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
                    else:
                        print(f"[Go2 PLAY] WARNING: unknown sub-terrain '{name}' "
                              f"(known: {list(sub_terrains.keys())})")
                final = {n: sub_terrains[n].proportion for n in sub_terrains}
                print(f"[Go2 PLAY] PLAY_SUBTERRAIN_MIX -> {final}")

            if _play_cell:
                new_cell = float(_play_cell)
                for cfg_sub in sub_terrains.values():
                    cfg_sub.cell_size = new_cell
                # Keep grid_size=(15,15) -> tile size = 15 * cell_size meters.
                # Match outer tile size so the scene grid stays consistent.
                tile_m = 15 * new_cell
                self.scene.terrain.terrain_generator.size = (tile_m, tile_m)
                print(f"[Go2 PLAY] PLAY_CELL_SIZE={new_cell}m -> tile={tile_m}m, "
                      f"corridor ~{new_cell * 0.6:.2f}-{new_cell * 0.9:.2f}m")

            # PLAY_CLASSIC_MAZE=1: reproduce paper-style mazes with uniform
            # rectangular walls and clear corridors (no randomized pillar / bar
            # / cross obstacles inserted into the maze pattern). Affects only
            # the `maze` sub-terrain since `non_maze` and `pits` rely on the
            # randomized shapes for their gameplay.
            _classic = os.environ.get("PLAY_CLASSIC_MAZE", "").strip().lower() in ("1", "true", "yes")
            if _classic and "maze" in sub_terrains:
                sub_terrains["maze"].randomize_wall = False
                sub_terrains["maze"].random_wall_ratio = 0.0
                print(f"[Go2 PLAY] PLAY_CLASSIC_MAZE=1 -> maze walls are full-height "
                      f"rectangles (no randomized obstacle shapes)")

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Visualize the depth-camera ray hits + height scanner grid in the GUI.
        # The RayCasterCamera draws a point per ray that hits geometry; the
        # density / pattern of these points is the effective FOV / resolution.
        self.scene.raycast_camera.debug_vis = True
        if self.scene.height_scanner_critic is not None:
            self.scene.height_scanner_critic.debug_vis = True
