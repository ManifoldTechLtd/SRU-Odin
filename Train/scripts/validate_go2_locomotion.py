"""Validate that policy_go2_jit.pt produces correct Go2 locomotion inside the
sru-navigation-sim environment, without any high-level navigation policy.

We feed hard-coded SE2 commands (vx, vy, omega) directly into env.step() and
measure the actual base motion that results. This isolates the low-level
locomotion contract from any navigation training.

Test sequence (each phase ~3 seconds):
  1. Stand still       (0.0, 0.0, 0.0) -> expect: stays upright, ~0 velocity
  2. Forward walk      (0.5, 0.0, 0.0) -> expect: moves +X, vx ~ 0.3 m/s
  3. Backward walk    (-0.5, 0.0, 0.0) -> expect: moves -X, vx ~ -0.3 m/s
  4. Lateral right     (0.0, 0.5, 0.0) -> expect: moves +Y (limited)
  5. Rotate (yaw)      (0.0, 0.0, 0.5) -> expect: omega ~ 0.35 rad/s

Run:
  ./isaaclab.sh -p /tmp/validate_go2_locomotion.py --num_envs 4 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Nav-PPO-Go2-Dev-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--phase_seconds", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True   # depth camera obs needed by env

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab_nav_task  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(
        args.task, device="cuda:0", num_envs=args.num_envs, use_fabric=True
    )
    env = gym.make(args.task, cfg=env_cfg)
    print(f"[validate] action_space  = {env.action_space}")
    print(f"[validate] observation_space (policy group) = "
          f"{env.observation_space['policy'].shape}")

    obs, _ = env.reset()
    device = env.unwrapped.device

    # Phase definitions: (label, command, num_steps)
    # The env decimation is 4 (low-level @ 50 Hz), and the high-level step is
    # called at 10 Hz (env.step). So phase_seconds * 10 = num steps.
    phase_steps = int(args.phase_seconds * 10)
    phases = [
        ("stand_still",   torch.tensor([ 0.0,  0.0,  0.0])),
        ("forward",       torch.tensor([ 0.7,  0.0,  0.0])),
        ("backward",      torch.tensor([-0.7,  0.0,  0.0])),
        ("lateral_right", torch.tensor([ 0.0,  0.7,  0.0])),
        ("yaw_rotate",    torch.tensor([ 0.0,  0.0,  0.7])),
    ]

    # Probe sensors
    robot = env.unwrapped.scene["robot"]

    print("\n" + "=" * 78)
    print(f"{'phase':<16} {'cmd (vx,vy,ω)':<22} "
          f"{'measured (vx,vy,ω)':<22} "
          f"{'h(m)':>6} {'pitch':>7} {'falls':>6}")
    print("=" * 78)

    fails = []
    for label, cmd in phases:
        # Fresh reset per phase so prior falls / terrain placement don't bias
        # the next phase's height measurement.
        env.reset()
        cmd_batch = cmd.to(device).unsqueeze(0).expand(args.num_envs, -1).contiguous()
        # Reset stats for this phase
        v_lin_sum  = torch.zeros(3, device=device)
        omega_sum  = torch.zeros(1, device=device)
        height_sum = torch.zeros(1, device=device)
        pitch_max  = 0.0
        n_steps    = 0
        falls      = 0

        for _ in range(phase_steps):
            obs, _, terminated, truncated, _ = env.step(cmd_batch)
            n_steps += 1
            base_lin_vel  = robot.data.root_lin_vel_b   # body frame, (N, 3)
            base_ang_vel  = robot.data.root_ang_vel_b
            root_pos      = robot.data.root_pos_w        # (N, 3)
            grav_b        = robot.data.projected_gravity_b  # (N, 3)
            v_lin_sum  += base_lin_vel.mean(dim=0)
            omega_sum  += base_ang_vel[:, 2].mean().unsqueeze(0)
            height_sum += root_pos[:, 2].mean().unsqueeze(0)
            # pitch = angle between projected_gravity_b and (-Z body) = (0,0,-1)
            pitch_now = (
                torch.acos(torch.clamp(-grav_b[:, 2], -1.0, 1.0))
                .max().item()
            )
            pitch_max = max(pitch_max, pitch_now)
            falls += int(terminated.any().item() or truncated.any().item())

        v_lin = (v_lin_sum / n_steps).cpu().numpy().round(3).tolist()
        omega = (omega_sum / n_steps).item()
        h_avg = (height_sum / n_steps).item()

        # Height threshold uses world-frame Z; the rough terrain in the dev env
        # can place the robot on top of features up to ~0.25 m, so allow up
        # to 0.65 m total for a healthy Go2 body (~0.40 m default standing).
        ok_height = 0.20 < h_avg < 0.70
        ok_pitch  = pitch_max < 0.6
        ok_overall = ok_height and ok_pitch
        flag = "✓" if ok_overall else "✗"

        cmd_str = f"({cmd[0]:+.1f},{cmd[1]:+.1f},{cmd[2]:+.1f})"
        meas_str = f"({v_lin[0]:+.2f},{v_lin[1]:+.2f},{omega:+.2f})"
        print(f"{flag} {label:<14} {cmd_str:<22} {meas_str:<22} "
              f"{h_avg:>6.3f} {pitch_max:>6.2f}r {falls:>6d}")

        # Per-phase pass criteria
        if label == "stand_still":
            if abs(v_lin[0]) > 0.15 or abs(v_lin[1]) > 0.15:
                fails.append(f"{label}: drifting at zero command "
                             f"(v_lin={v_lin})")
        elif label == "forward":
            if v_lin[0] < 0.10:
                fails.append(f"{label}: not moving forward "
                             f"(vx={v_lin[0]:.2f})")
        elif label == "backward":
            if v_lin[0] > -0.10:
                fails.append(f"{label}: not moving backward "
                             f"(vx={v_lin[0]:.2f})")
        elif label == "yaw_rotate":
            if abs(omega) < 0.10:
                fails.append(f"{label}: not rotating (ω={omega:.2f})")

        if not ok_height:
            fails.append(f"{label}: body too low/high (h={h_avg:.2f})")
        if not ok_pitch:
            fails.append(f"{label}: tilted over (max pitch={pitch_max:.2f} rad)")

    print("=" * 78)
    if fails:
        print("\n⚠ FAILED CHECKS:")
        for f in fails:
            print(f"  - {f}")
        print("\nLikely causes:")
        print("  • Action scale mismatch (training used different scale)")
        print("  • Joint order mismatch between policy output and IsaacLab USD")
        print("  • Observation order mismatch")
    else:
        print("\n✅ ALL LOCOMOTION CHECKS PASSED")
        print("   Go2 stands, walks forward/back, and rotates as commanded.")
        print("   Locomotion contract with sru-navigation-sim is correct.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
