# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import gymnasium as gym

from . import agents, navigation_env_cfg

##
# Register Gym environments.
##

##############################################################################################################
# PPO

gym.register(
    id="Isaac-Nav-PPO-Go2-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg,
        # MixedCfg is the cold-start tuned variant of the Dev hyperparams;
        # the original B2W-FT ``Go2NavPPORunnerCfg`` collapses exploration on a
        # from-scratch run (init_std=0.5, entropy=0.001, clip=0.1).
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavPPORunnerMixedCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Go2-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Go2-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavPPORunnerDevCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Go2-PureMaze-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg_PureMaze,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavPPORunnerPureMazeCfg,
    },
)
