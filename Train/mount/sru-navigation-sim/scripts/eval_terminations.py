#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Headless termination-breakdown evaluator for a trained navigation policy.

Question this answers: *for a given checkpoint on a given task/difficulty,
what fraction of episodes end via each termination reason* -- including the
re-thresholded ``terrain_fall`` (now -0.3m for Go2) which is the only way to
distinguish *fell into a pit* from *bumped a wall*.

The script mirrors ``play.py`` for env/policy setup, then counts the
TerminationManager's per-term done flags across a fixed number of env steps.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate termination breakdown for a navigation policy.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Gym task id (use a *-Play-v0 variant).")
parser.add_argument("--checkpoint", type=str, default=None, help="In-container path to model_*.pt.")
parser.add_argument("--steps", type=int, default=2000, help="Number of env.step() calls.")
parser.add_argument("--warmup", type=int, default=20, help="Steps to skip before counting (let resets settle).")
parser.add_argument("--seed", type=int, default=42, help="Seed.")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


def main():
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    agent_cfg_class = spec.kwargs.get("rsl_rl_cfg_entry_point")

    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_class()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    if not args_cli.checkpoint:
        raise SystemExit("[eval] --checkpoint is required.")
    print(f"[eval] loading checkpoint: {args_cli.checkpoint}")
    loaded = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"], strict=True)
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded["critic_obs_norm_state_dict"])
    print(f"[eval] checkpoint iter = {loaded.get('iter', '?')}")

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Termination manager exposes per-term done flags as torch tensors.
    tm = env.unwrapped.termination_manager
    term_names = list(tm.active_terms)
    print(f"[eval] active termination terms: {term_names}")

    # Per-term running counters (number of envs that ended via this reason).
    counts = {name: 0 for name in term_names}
    counts["__any__"] = 0  # any termination (sanity)

    # Print the relevant cfg knobs so the run is self-documenting.
    try:
        tf_params = env_cfg.terminations.terrain_fall.params
        print(f"[eval] terrain_fall.fall_height_threshold = "
              f"{tf_params.get('fall_height_threshold', 'default')}")
    except Exception:
        pass

    obs, _ = env.get_observations()
    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if step >= args_cli.warmup:
            # Each per-term tensor is bool[num_envs]; sum gives this-step count.
            for name in term_names:
                term_tensor = tm.get_term(name)
                counts[name] += int(term_tensor.sum().item())
            # Any-termination flag (rsl_rl wrapper resets on time_out OR done).
            any_done = env.unwrapped.termination_manager.dones
            counts["__any__"] += int(any_done.sum().item())
        step += 1

    # -------- summarise --------
    total = counts["__any__"] if counts["__any__"] > 0 else 1
    print("\n" + "=" * 78)
    print(f"[eval] steps counted = {args_cli.steps - args_cli.warmup}, num_envs = {args_cli.num_envs}")
    print(f"[eval] total terminations observed = {counts['__any__']}")
    print("\n  TERM                          COUNT     %-of-terms")
    print("  " + "-" * 50)
    for name in term_names:
        c = counts[name]
        pct = 100.0 * c / total
        marker = ""
        if name == "early_termination":
            marker = "  <- success"
        elif name in ("base_contact", "large_pitch_angle"):
            marker = "  <- collision/tipover"
        elif name == "terrain_fall":
            marker = "  <- PIT FALL (with new threshold)"
        elif name == "time_out":
            marker = "  <- ran out of time"
        print(f"  {name:<28} {c:>6}    {pct:>5.1f}%{marker}")
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
