#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Depth-ablation diagnostic for a trained navigation policy.

Question this answers: *does the policy actually use the depth-camera input,
or has it learned to ignore it (perception-blind)?*

Method
------
1. Load the trained actor-critic exactly like ``play.py``.
2. Locate the ``depth_image`` slice inside the concatenated *policy* observation.
3. Roll the env out with the REAL on-policy actions, and at every step also
   compute a counterfactual action where the depth slice is zeroed. Compare.
4. Also track the depth-feature statistics (std over envs/time) so we can tell
   a *dead sensor* (constant features) apart from a policy that *ignores*
   informative features.

Verdict logic (printed at the end)
----------------------------------
* depth features ~constant (std ~ 0)            -> SENSOR/ENCODER dead (no signal)
* features vary, zeroing barely changes action  -> LEARNED-BLIND (ignores depth)
* features vary, zeroing changes action a lot    -> policy DOES use depth

Usage (inside container, mirrors play.py):
    ./isaaclab.sh -p source/isaaclab_nav_task/scripts/ablate_depth.py \
        --task Isaac-Nav-PPO-Go2-Play-v0 --num_envs 16 \
        --checkpoint <path/model_6998.pt> --steps 300 --headless
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Depth-ablation diagnostic for a navigation policy.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (use a *-Play-v0 variant).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")
parser.add_argument("--steps", type=int, default=300, help="Number of env steps to roll out.")
parser.add_argument("--warmup", type=int, default=20, help="Steps to skip before collecting stats (let resets settle).")
parser.add_argument("--seed", type=int, default=42, help="Seed.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Cameras are mandatory for the depth observation.
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


def _find_depth_slice(env) -> tuple[int, int, list[tuple[str, int]]]:
    """Return (start, end, layout) of the ``depth_image`` term in the policy obs.

    layout is a list of (term_name, flat_dim) in concatenation order, for printing.
    """
    om = env.unwrapped.observation_manager
    names = om.active_terms["policy"]
    dims = om.group_obs_term_dim["policy"]  # list of shape-tuples per term

    layout: list[tuple[str, int]] = []
    offset = 0
    depth_start = depth_end = -1
    for name, shape in zip(names, dims):
        flat = 1
        for s in shape:
            flat *= s
        layout.append((name, flat))
        if name == "depth_image":
            depth_start, depth_end = offset, offset + flat
        offset += flat
    if depth_start < 0:
        raise RuntimeError(
            f"No 'depth_image' term found in policy obs group. Terms = {names}"
        )
    return depth_start, depth_end, layout


def main():
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    agent_cfg_class = spec.kwargs.get("rsl_rl_cfg_entry_point")

    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_class()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # --- load checkpoint (same logic as play.py) ---
    resume_path = args_cli.checkpoint
    if not resume_path:
        raise SystemExit("[ablate] --checkpoint is required for the diagnostic.")
    print(f"[ablate] loading checkpoint: {resume_path}")
    loaded = torch.load(resume_path, map_location="cpu", weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"], strict=True)
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded["critic_obs_norm_state_dict"])
    print(f"[ablate] checkpoint iter = {loaded.get('iter', '?')}")

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    depth_start, depth_end, layout = _find_depth_slice(env)
    print("=" * 78)
    print("[ablate] policy observation layout (term : flat_dim):")
    for name, flat in layout:
        marker = "  <-- DEPTH" if name == "depth_image" else ""
        print(f"    {name:<28} {flat:>5}{marker}")
    print(f"[ablate] depth slice = [{depth_start}:{depth_end}]  ({depth_end - depth_start} dims)")
    print("=" * 78)

    obs, _ = env.get_observations()

    # accumulators
    n = 0
    act_norm_sum = 0.0          # ||a_real||
    diff_norm_sum = 0.0         # ||a_real - a_zero||
    per_dim_abs_diff = None     # mean |a_real - a_zero| per action dim
    per_dim_abs_act = None      # mean |a_real| per action dim
    depth_feat_chunks = []      # collect depth slices to measure variability

    device = env.unwrapped.device
    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        with torch.inference_mode():
            a_real = policy(obs)
            obs_zero = obs.clone()
            obs_zero[:, depth_start:depth_end] = 0.0
            a_zero = policy(obs_zero)

        if step >= args_cli.warmup:
            d = (a_real - a_zero)
            diff_norm_sum += torch.linalg.vector_norm(d, dim=-1).mean().item()
            act_norm_sum += torch.linalg.vector_norm(a_real, dim=-1).mean().item()
            ad = d.abs().mean(dim=0)
            aa = a_real.abs().mean(dim=0)
            per_dim_abs_diff = ad if per_dim_abs_diff is None else per_dim_abs_diff + ad
            per_dim_abs_act = aa if per_dim_abs_act is None else per_dim_abs_act + aa
            depth_feat_chunks.append(obs[:, depth_start:depth_end].detach().clone())
            n += 1

        with torch.inference_mode():
            obs, _, _, _ = env.step(a_real)
        step += 1

    # --- summarise ---
    print("\n" + "=" * 78)
    print(f"[ablate] collected {n} steps (after {args_cli.warmup} warmup), {args_cli.num_envs} envs")
    if n == 0:
        print("[ablate] no samples collected; increase --steps.")
    else:
        mean_act = act_norm_sum / n
        mean_diff = diff_norm_sum / n
        rel = (mean_diff / mean_act) if mean_act > 0 else float("nan")
        per_dim_abs_diff = (per_dim_abs_diff / n).tolist()
        per_dim_abs_act = (per_dim_abs_act / n).tolist()

        feats = torch.cat(depth_feat_chunks, dim=0)  # (n*envs, depth_dim)
        feat_std_over_samples = feats.std(dim=0).mean().item()  # variability across time/envs
        feat_abs_mean = feats.abs().mean().item()

        names_se2 = ["vx", "vy", "omega"]
        print("\n-- ACTION SENSITIVITY TO DEPTH (real vs depth-zeroed) --")
        print(f"  mean ||a_real||              = {mean_act:.4f}")
        print(f"  mean ||a_real - a_zero||     = {mean_diff:.4f}")
        print(f"  RELATIVE action change       = {rel*100:.2f}%")
        print("  per-action-dim |diff| / |act|:")
        for i, v in enumerate(per_dim_abs_diff):
            nm = names_se2[i] if i < len(names_se2) else f"a{i}"
            denom = per_dim_abs_act[i] if per_dim_abs_act[i] > 0 else float("nan")
            print(f"    {nm:<6} |diff|={v:.4f}  |act|={per_dim_abs_act[i]:.4f}  ratio={v/denom*100:.1f}%")

        print("\n-- DEPTH FEATURE SIGNAL (is the camera/encoder alive?) --")
        print(f"  feature |mean|               = {feat_abs_mean:.4f}")
        print(f"  feature std over time/envs   = {feat_std_over_samples:.4f}")

        print("\n-- VERDICT --")
        if feat_std_over_samples < 1e-3:
            print("  >> DEAD SIGNAL: depth features are ~constant. Camera/encoder is")
            print("     not producing a varying signal (sensor/render/orientation bug).")
        elif rel < 0.02:
            print("  >> LEARNED-BLIND: features vary but zeroing them changes the action")
            print(f"     by only {rel*100:.2f}%. The policy effectively IGNORES depth.")
        elif rel < 0.10:
            print(f"  >> WEAK USE: depth changes the action by {rel*100:.2f}% (marginal).")
        else:
            print(f"  >> DEPTH IS USED: zeroing depth changes the action by {rel*100:.2f}%.")
            print("     Perception is wired in; the failure is likely a difficulty/")
            print("     generalisation gap, not blindness.")
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
