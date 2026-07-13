#!/usr/bin/env python3
"""Render figures from a collisions.npz produced by analyze_collisions.py.

Run on the *host* (matplotlib only, no Isaac). Writes PNGs next to the .npz.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="Path to collisions.npz")
    args = ap.parse_args()

    d = np.load(args.npz)
    out_dir = os.path.dirname(os.path.abspath(args.npz))

    thr = float(d["threshold"])
    ep_w = float(d["ep_term_weight"])
    K = int(d["K"])
    n_coll = int(d["n_collisions"])
    n_succ = int(d["n_success"])
    n_to = int(d["n_timeout"])

    impact_vel = d["impact_vel"]
    impact_speed = d["impact_speed"]
    impact_force = d["impact_force"]
    traj_vel = d["traj_vel"]
    traj_speed = d["traj_speed"]
    traj_force = d["traj_force"]
    all_vel = d["all_vel"]
    all_force = d["all_force"]

    # ===== Fig 1: impact velocity histogram =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    if impact_vel.size > 0:
        ax.hist(impact_vel, bins=40, color="#d9534f", alpha=0.85, edgecolor="white")
        ax.axvline(impact_vel.mean(), color="black", ls="--", lw=1.5,
                   label=f"mean = {impact_vel.mean():+.2f} m/s")
        ax.axvline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("body-frame forward velocity at impact  (m/s)")
    ax.set_ylabel("# collisions")
    ax.set_title(f"Velocity at collision  (n={impact_vel.size})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    if impact_speed.size > 0:
        ax.hist(impact_speed, bins=40, color="#5bc0de", alpha=0.85, edgecolor="white")
        ax.axvline(impact_speed.mean(), color="black", ls="--", lw=1.5,
                   label=f"mean = {impact_speed.mean():.2f} m/s")
        ax.axvline(0.2, color="orange", ls=":", lw=1, label="0.2 m/s (slow-impact cutoff)")
    ax.set_xlabel("planar speed |v_xy| at impact  (m/s)")
    ax.set_ylabel("# collisions")
    ax.set_title("Planar speed at collision")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_impact_velocity.png"), dpi=130)
    plt.close(fig)

    # ===== Fig 2: contact force histogram (log scale) with threshold =====
    fig, ax = plt.subplots(figsize=(8, 5))
    if impact_force.size > 0:
        bins = np.logspace(np.log10(max(thr * 0.5, 0.5)),
                           np.log10(max(impact_force.max() * 1.2, thr * 10)), 50)
        ax.hist(impact_force, bins=bins, color="#f0ad4e", alpha=0.85, edgecolor="white",
                label=f"impact peak force  (n={impact_force.size})")
        ax.axvline(thr, color="red", ls="--", lw=2, label=f"training threshold = {thr:.1f} N")
        ax.axvline(np.median(impact_force), color="black", ls=":", lw=1.5,
                   label=f"median impact = {np.median(impact_force):.0f} N "
                         f"({np.median(impact_force)/thr:.0f}× thr)")
    ax.set_xscale("log")
    ax.set_xlabel("peak contact force on (base, hip, thigh) at impact  (N, log)")
    ax.set_ylabel("# collisions")
    ax.set_title("Impact force vs training termination threshold")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_impact_force.png"), dpi=130)
    plt.close(fig)

    # ===== Fig 3: pre-impact trajectory (median + IQR) =====
    if traj_vel.shape[0] > 0:
        steps = np.arange(-K + 1, 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

        for ax, data, ylabel, title, color in [
            (axes[0], traj_speed, "planar speed |v_xy| (m/s)", "Speed leading up to collision", "#5bc0de"),
            (axes[1], traj_force, "peak contact force (N)", "Force leading up to collision", "#f0ad4e"),
        ]:
            med = np.median(data, axis=0)
            p25 = np.percentile(data, 25, axis=0)
            p75 = np.percentile(data, 75, axis=0)
            ax.fill_between(steps, p25, p75, color=color, alpha=0.35, label="IQR (25-75%)")
            ax.plot(steps, med, color=color, lw=2.2, label="median")
            ax.axvline(0, color="red", ls="--", lw=1.5, label="impact step")
            ax.set_xlabel("env-step relative to impact")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}  (n={data.shape[0]})")
            ax.legend()
            ax.grid(alpha=0.3)
        if axes[1].get_ylim()[1] > 5 * thr:
            axes[1].set_yscale("symlog", linthresh=max(thr, 1.0))
            axes[1].axhline(thr, color="red", ls=":", lw=1)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig3_pre_impact.png"), dpi=130)
        plt.close(fig)

    # ===== Fig 4: global v_fwd distribution (alive steps) vs at-impact =====
    fig, ax = plt.subplots(figsize=(8, 5))
    if all_vel.size > 0:
        ax.hist(all_vel, bins=60, density=True, color="#5cb85c", alpha=0.55,
                edgecolor="white", label=f"alive steps  (n={all_vel.size})")
    if impact_vel.size > 0:
        ax.hist(impact_vel, bins=40, density=True, color="#d9534f", alpha=0.55,
                edgecolor="white", label=f"at impact  (n={impact_vel.size})")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("body-frame forward velocity v_fwd  (m/s)")
    ax.set_ylabel("density")
    ax.set_title("Forward velocity: normal navigation vs at collision")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_vfwd_normal_vs_impact.png"), dpi=130)
    plt.close(fig)

    # ===== short text summary =====
    lines = []
    lines.append(f"threshold = {thr} N   ep_term_w = {ep_w}")
    lines.append(f"terminations: success={n_succ}  collision={n_coll}  timeout={n_to}")
    if impact_force.size > 0:
        lines.append(f"impact force: median {np.median(impact_force):.1f} N  "
                     f"({np.median(impact_force)/thr:.0f}x threshold), "
                     f"p90 {np.percentile(impact_force,90):.1f}, max {impact_force.max():.1f}")
    if impact_speed.size > 0:
        slow = (impact_speed < 0.2).mean() * 100
        fast = (impact_speed > 0.8).mean() * 100
        lines.append(f"impact |v_xy|: median {np.median(impact_speed):.2f} m/s  "
                     f"slow<0.2: {slow:.0f}%   fast>0.8: {fast:.0f}%")
    print("\n".join(lines))
    print(f"\nfigures saved into: {out_dir}")


if __name__ == "__main__":
    main()
