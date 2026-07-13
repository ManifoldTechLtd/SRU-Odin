#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Instrumented collision analysis for a trained navigation policy.

For every env-step, this script records:
  * base_lin_vel_b[:, 0]   -- forward velocity in body frame (m/s)
  * max contact-force magnitude over the *termination sensor body subset*
    (whatever ``terminations.base_contact.params['sensor_cfg'].body_ids`` is)
  * the per-term ``base_contact`` done flag

It maintains a per-env rolling buffer of the last K=16 steps so that, when a
collision fires for env i, we can dump the *pre-impact trajectory* (velocity +
peak force) for that env. The aggregated data is saved as a single .npz which
the companion script ``plot_collisions.py`` turns into figures.

Outputs (under --out-dir, default ``logs/collisions/<task>__<ckpt>``):
  * collisions.npz           full dump (see save_data())
  * summary.txt              human-readable numbers
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lookback", type=int, default=16,
                    help="How many pre-impact steps to capture per collision.")
parser.add_argument("--out-dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


def main():
    spec = gym.spec(args_cli.task)
    env_cfg: ManagerBasedRLEnvCfg = spec.kwargs["env_cfg_entry_point"]()
    agent_cfg: RslRlOnPolicyRunnerCfg = spec.kwargs["rsl_rl_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device
    N = args_cli.num_envs

    # ---- load policy ----
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"], strict=True)
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded["critic_obs_norm_state_dict"])
    print(f"[collisions] checkpoint iter = {loaded.get('iter', '?')}")
    policy = runner.get_inference_policy(device=device)

    # ---- resolve the *exact* body subset used by the base_contact termination ----
    tm = env.unwrapped.termination_manager
    tf = tm.get_term_cfg("base_contact")
    sensor_cfg: SceneEntityCfg = tf.params["sensor_cfg"]
    threshold = float(tf.params["threshold"])
    print(f"[collisions] base_contact threshold = {threshold} N")
    print(f"[collisions] base_contact body_names spec = {sensor_cfg.body_names}")

    # NOTE: TerminationManager already resolved this SceneEntityCfg at env init,
    # so body_ids is already populated; calling .resolve() again would raise.
    body_ids = sensor_cfg.body_ids
    contact_sensor: ContactSensor = env.unwrapped.scene.sensors[sensor_cfg.name]
    print(f"[collisions] resolved body_ids = {list(body_ids) if not isinstance(body_ids, slice) else body_ids}")

    # episode_termination reward weight (for context in summary)
    try:
        ep_term_w = float(env.unwrapped.reward_manager.get_term_cfg("episode_termination").weight)
    except Exception:
        ep_term_w = float("nan")

    robot = env.unwrapped.scene["robot"]

    # ---- rolling buffers (K x N) for pre-impact dumps ----
    K = args_cli.lookback
    vel_buf = torch.zeros(K, N, device=device)        # base lin vel x (body frame)
    speed_buf = torch.zeros(K, N, device=device)      # |v_xy| (body frame)
    force_buf = torch.zeros(K, N, device=device)      # max contact force on subset
    head = 0  # circular index

    # ---- collected data ----
    coll_traj_vel = []     # list of (K,) numpy arrays at the impact moment
    coll_traj_speed = []
    coll_traj_force = []
    coll_impact_vel = []   # scalar at the IMPACT step (after termination flag flips)
    coll_impact_speed = []
    coll_impact_force = []

    # global histograms over all NON-terminating steps (for context)
    all_vel_chunks = []
    all_force_chunks = []

    n_steps_counted = 0
    n_collisions = 0
    n_success = 0
    n_timeout = 0

    obs, _ = env.get_observations()
    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        # -------------------------------------------------------------
        # CRITICAL: measure state BEFORE env.step(), because IsaacLab
        # auto-resets terminated envs INSIDE env.step() and that wipes
        # both robot.data and contact_sensor.data for the dying envs.
        # We do NOT lose the impact data: the contact force history
        # buffer (decimation samples) accumulates within env.step(),
        # so reading it on the *next* iteration's pre-step pass would
        # be the natural way -- but at that point reset has zeroed it
        # for dying envs.  Compromise: at pre-step iter t+1, read
        # sensor history (last decimation_count physics steps from
        # iter t) for the envs that just died -- still works for envs
        # whose sensor history isn't fully cleared by reset (depends
        # on IsaacLab version).  Velocity, on the other hand, must
        # be sampled BEFORE the impact step; using buffer[head-1]
        # (the prior iteration's measurement) gives us exactly that.
        # -------------------------------------------------------------
        v_b = robot.data.root_lin_vel_b  # (N, 3) -- state at start of iter t
        v_fwd = v_b[:, 0]
        v_xy = torch.norm(v_b[:, :2], dim=-1)

        net_force_hist = contact_sensor.data.net_forces_w_history  # (N, H, B_total, 3)
        f_mags = torch.norm(net_force_hist[:, :, body_ids, :], dim=-1)
        peak_force, _ = f_mags.reshape(N, -1).max(dim=-1)  # (N,)

        # Save what we *had* in the buffer before overwriting, in case
        # we need to look at "1 step ago" specifically.
        prev_head = (head - 1) % K
        prev_vel = vel_buf[prev_head].clone()
        prev_speed = speed_buf[prev_head].clone()
        prev_force = force_buf[prev_head].clone()

        # Push into rolling buffer (newest at index `head`, then advance).
        vel_buf[head] = v_fwd
        speed_buf[head] = v_xy
        force_buf[head] = peak_force
        new_head = head
        head = (head + 1) % K

        # ---- step the env (terminations + auto-reset happen in here) ----
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # ---- post-step: read sensor history AGAIN for envs that just died;
        # this captures the *peak* contact force during the impact step
        # if the reset hasn't yet zeroed the history buffer.
        post_force_hist = contact_sensor.data.net_forces_w_history
        post_f_mags = torch.norm(post_force_hist[:, :, body_ids, :], dim=-1)
        post_peak_force, _ = post_f_mags.reshape(N, -1).max(dim=-1)

        if step >= args_cli.warmup:
            n_steps_counted += 1

            done_bc = tm.get_term("base_contact").to(torch.bool)
            done_to = tm.get_term("time_out").to(torch.bool)
            done_succ = tm.get_term("early_termination").to(torch.bool)

            n_collisions += int(done_bc.sum().item())
            n_success += int(done_succ.sum().item())
            n_timeout += int(done_to.sum().item())

            if done_bc.any():
                idxs = torch.where(done_bc)[0].cpu().tolist()
                # Buffer order: oldest -> newest. Newest = just pushed at
                # `new_head`. For envs that died THIS env.step(), the
                # `new_head` sample is the pre-impact state at iter start.
                order = [(head + i) % K for i in range(K)]
                for i in idxs:
                    vt = vel_buf[order, i].cpu().numpy()
                    st = speed_buf[order, i].cpu().numpy()
                    ft = force_buf[order, i].cpu().numpy()
                    coll_traj_vel.append(vt)
                    coll_traj_speed.append(st)
                    coll_traj_force.append(ft)
                    # Pre-impact velocity = the value we measured BEFORE
                    # this env.step() (which contained the actual impact).
                    coll_impact_vel.append(float(v_fwd[i].item()))
                    coll_impact_speed.append(float(v_xy[i].item()))
                    # Impact force: take max of (a) pre-step reading, (b) post-step
                    # reading, (c) buffer max over the window -- whichever is
                    # largest is the best estimate of true peak force.
                    win_peak = float(force_buf[:, i].max().item())
                    impact_f = max(
                        float(peak_force[i].item()),
                        float(post_peak_force[i].item()),
                        win_peak,
                    )
                    coll_impact_force.append(impact_f)

            # Global samples (alive envs only).
            alive = ~tm.dones.to(torch.bool)
            if alive.any():
                all_vel_chunks.append(v_fwd[alive].cpu().numpy())
                # use post-step force for non-dying envs -- it includes
                # in-step contacts the policy then recovers from.
                all_force_chunks.append(post_peak_force[alive].cpu().numpy())

        step += 1

    # ---- save ----
    out_dir = args_cli.out_dir
    if out_dir is None:
        tag = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
        out_dir = os.path.join("logs", "collisions", f"{args_cli.task}__{tag}")
    os.makedirs(out_dir, exist_ok=True)

    impact_vel_arr = np.array(coll_impact_vel)
    impact_speed_arr = np.array(coll_impact_speed)
    impact_force_arr = np.array(coll_impact_force)
    traj_vel_arr = np.stack(coll_traj_vel) if coll_traj_vel else np.zeros((0, K))
    traj_speed_arr = np.stack(coll_traj_speed) if coll_traj_speed else np.zeros((0, K))
    traj_force_arr = np.stack(coll_traj_force) if coll_traj_force else np.zeros((0, K))
    all_vel_arr = np.concatenate(all_vel_chunks) if all_vel_chunks else np.zeros((0,))
    all_force_arr = np.concatenate(all_force_chunks) if all_force_chunks else np.zeros((0,))

    np.savez_compressed(
        os.path.join(out_dir, "collisions.npz"),
        impact_vel=impact_vel_arr,
        impact_speed=impact_speed_arr,
        impact_force=impact_force_arr,
        traj_vel=traj_vel_arr,
        traj_speed=traj_speed_arr,
        traj_force=traj_force_arr,
        all_vel=all_vel_arr,
        all_force=all_force_arr,
        threshold=np.array(threshold, dtype=np.float32),
        ep_term_weight=np.array(ep_term_w, dtype=np.float32),
        K=np.array(K),
        num_envs=np.array(N),
        steps=np.array(n_steps_counted),
        n_collisions=np.array(n_collisions),
        n_success=np.array(n_success),
        n_timeout=np.array(n_timeout),
    )

    # ---- human summary ----
    def pct(x, total):
        return 100.0 * x / max(total, 1)

    total_term = n_collisions + n_success + n_timeout
    summary = []
    summary.append(f"task                : {args_cli.task}")
    summary.append(f"checkpoint          : {args_cli.checkpoint}")
    summary.append(f"steps counted       : {n_steps_counted}  (warmup={args_cli.warmup})")
    summary.append(f"num envs            : {N}")
    summary.append(f"")
    summary.append(f"base_contact threshold (training) : {threshold} N")
    summary.append(f"episode_termination reward weight : {ep_term_w}")
    summary.append(f"")
    summary.append(f"terminations : success={n_success}  collision={n_collisions}  timeout={n_timeout}")
    summary.append(f"  success %    : {pct(n_success, total_term):.1f}")
    summary.append(f"  collision %  : {pct(n_collisions, total_term):.1f}")
    summary.append(f"  timeout %    : {pct(n_timeout, total_term):.1f}")
    summary.append(f"")
    if impact_vel_arr.size > 0:
        summary.append(f"AT IMPACT (n={impact_vel_arr.size})")
        summary.append(f"  v_fwd (body x)  : mean={impact_vel_arr.mean():+.3f}  median={np.median(impact_vel_arr):+.3f}  "
                       f"p10={np.percentile(impact_vel_arr,10):+.3f}  p90={np.percentile(impact_vel_arr,90):+.3f}  "
                       f"std={impact_vel_arr.std():.3f}  (m/s)")
        summary.append(f"  |v_xy|          : mean={impact_speed_arr.mean():.3f}  median={np.median(impact_speed_arr):.3f}  "
                       f"p10={np.percentile(impact_speed_arr,10):.3f}  p90={np.percentile(impact_speed_arr,90):.3f}  (m/s)")
        summary.append(f"  peak force      : mean={impact_force_arr.mean():.1f}  median={np.median(impact_force_arr):.1f}  "
                       f"p10={np.percentile(impact_force_arr,10):.1f}  p90={np.percentile(impact_force_arr,90):.1f}  "
                       f"max={impact_force_arr.max():.1f}  (N)")
        summary.append(f"  force / threshold ratio   : median {np.median(impact_force_arr)/threshold:.1f}x, "
                       f"max {impact_force_arr.max()/threshold:.1f}x")
        n_slow = int((impact_speed_arr < 0.2).sum())
        n_fast = int((impact_speed_arr > 0.8).sum())
        summary.append(f"  slow impacts (|v_xy|<0.2 m/s)  : {n_slow}  ({pct(n_slow, impact_speed_arr.size):.1f}%)")
        summary.append(f"  fast impacts (|v_xy|>0.8 m/s)  : {n_fast}  ({pct(n_fast, impact_speed_arr.size):.1f}%)")
    else:
        summary.append("NO COLLISIONS in this run.")
    if all_vel_arr.size > 0:
        summary.append(f"")
        summary.append(f"GLOBAL (alive-step samples, n={all_vel_arr.size})")
        summary.append(f"  v_fwd  : mean={all_vel_arr.mean():+.3f}  median={np.median(all_vel_arr):+.3f}  "
                       f"std={all_vel_arr.std():.3f}  (m/s)")
        summary.append(f"  peak F : mean={all_force_arr.mean():.2f}  median={np.median(all_force_arr):.2f}  "
                       f"p99={np.percentile(all_force_arr,99):.2f}  (N)")

    txt = "\n".join(summary)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)
    print(f"\n[collisions] saved to: {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
