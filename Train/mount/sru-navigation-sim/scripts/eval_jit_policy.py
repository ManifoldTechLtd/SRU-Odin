#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Headless evaluator for a *TorchScript JIT* navigation policy.

Mirrors ``eval_terminations.py`` but instead of building an rsl_rl Runner and
loading a ``model_*.pt`` checkpoint, it directly loads a JIT-scripted
``ActorCriticSRU`` policy (e.g. the official upstream deployment policy
``nav_policy_new.pt``) and rolls it out in the IsaacLab navigation env.

The JIT module exposes:
    forward(observations: Tensor, reset: bool=False) -> actions: Tensor
with internal ``hidden_state`` / ``cell_state`` buffers sized
``(num_layers, 1, hidden_size)``. We resize those buffers to
``(num_layers, num_envs, hidden_size)`` after load so a single forward call
handles all envs in parallel, and we per-env zero them on episode reset.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a JIT navigation policy.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, required=True, help="Gym task id (use a *-Play-v0 variant).")
parser.add_argument("--jit_policy", type=str, required=True, help="In-container path to TorchScript .pt.")
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    # ---- env setup ----
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device
    num_envs = args_cli.num_envs

    # ---- policy load ----
    print(f"[eval-jit] loading JIT policy: {args_cli.jit_policy}")
    model = torch.jit.load(args_cli.jit_policy, map_location=device)
    model.eval()
    print(f"[eval-jit] model loaded; type={type(model).__name__}")

    # Resize hidden state buffers from (num_layers, 1, hidden_size) to
    # (num_layers, num_envs, hidden_size). Works because ``rnn`` is a generic
    # LSTM-style cell that accepts any batch dim.
    h = model.hidden_state
    c = model.cell_state
    print(f"[eval-jit] original hidden_state shape: {tuple(h.shape)}")
    model.hidden_state = torch.zeros(h.shape[0], num_envs, h.shape[2], device=device, dtype=h.dtype)
    model.cell_state = torch.zeros(c.shape[0], num_envs, c.shape[2], device=device, dtype=c.dtype)
    print(f"[eval-jit] resized hidden_state shape: {tuple(model.hidden_state.shape)}")

    # ---- termination accounting ----
    tm = env.unwrapped.termination_manager
    term_names = list(tm.active_terms)
    print(f"[eval-jit] active termination terms: {term_names}")
    counts = {name: 0 for name in term_names}
    counts["__any__"] = 0

    # ---- rollout ----
    obs, _ = env.get_observations()
    prev_dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        # Zero hidden state for envs that just terminated (got reset internally
        # by IsaacLab at the start of this step's env.step()).
        if prev_dones.any():
            mask = prev_dones.to(torch.bool)
            # buffer shape: (num_layers, num_envs, hidden_size)
            model.hidden_state[:, mask, :] = 0
            model.cell_state[:, mask, :] = 0

        with torch.inference_mode():
            actions = model(obs, False)
            obs, _, dones, _ = env.step(actions)

        prev_dones = dones.to(torch.bool)

        if step >= args_cli.warmup:
            for name in term_names:
                counts[name] += int(tm.get_term(name).sum().item())
            counts["__any__"] += int(env.unwrapped.termination_manager.dones.sum().item())
        step += 1

    # ---- report ----
    total = counts["__any__"] if counts["__any__"] > 0 else 1
    print("\n" + "=" * 78)
    print(f"[eval-jit] steps counted = {args_cli.steps - args_cli.warmup}, num_envs = {num_envs}")
    print(f"[eval-jit] total terminations observed = {counts['__any__']}")
    print("\n  TERM                          COUNT     %-of-terms")
    print("  " + "-" * 50)
    for name in term_names:
        c_n = counts[name]
        pct = 100.0 * c_n / total
        marker = ""
        if name == "early_termination":
            marker = "  <- success"
        elif name in ("base_contact", "large_pitch_angle"):
            marker = "  <- collision/tipover"
        elif name == "terrain_fall":
            marker = "  <- PIT FALL"
        elif name == "time_out":
            marker = "  <- ran out of time"
        print(f"  {name:<28} {c_n:>6}    {pct:>5.1f}%{marker}")
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
