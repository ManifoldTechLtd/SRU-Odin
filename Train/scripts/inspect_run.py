#!/usr/bin/env python3
"""Inspect a SRU-Go2 training run: dump TensorBoard scalars + ckpt summary.

Usage:
    python scripts/inspect_run.py --run outputs/logs/rsl_rl/<exp>/<timestamp_runname>
    python scripts/inspect_run.py --latest                       # auto-pick newest run
    python scripts/inspect_run.py --latest --experiment go2_navigation_ppo_dev
    python scripts/inspect_run.py --run <path> --tags Train/mean_reward Loss/value_function

The script tries `tensorboard` first, falls back to `tbparse`, then to a
minimal pure-protobuf reader. Prints a compact markdown-friendly summary
that can be pasted directly into docs/RUN_LOG.md.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = REPO_ROOT / "outputs" / "logs" / "rsl_rl"

# Tags worth surfacing by default (rsl_rl + custom). The script will keep only
# the ones that actually appear in the events file.
DEFAULT_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Train/learning_rate",
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/entropy",
    "Policy/mean_noise_std",
    "Train/mean_success_rate",
    "Train/mean_collision_rate",
]


def _find_latest_run(experiment: str | None) -> Path:
    if experiment:
        roots = [DEFAULT_LOG_ROOT / experiment]
    else:
        roots = sorted([p for p in DEFAULT_LOG_ROOT.iterdir() if p.is_dir()])
    candidates: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for entry in root.iterdir():
            if entry.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", entry.name):
                candidates.append(entry)
    if not candidates:
        raise SystemExit(f"[inspect_run] No runs under {DEFAULT_LOG_ROOT}")
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _read_scalars(run_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Read all scalar tags from the run's tfevents files.

    Returns mapping: tag -> list[(step, value)] sorted by step.
    Tries tensorboard.EventAccumulator; falls back to tbparse; then raw protobuf.
    """
    files = sorted(glob.glob(str(run_dir / "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"[inspect_run] No tfevents in {run_dir}")

    # --- attempt 1: tensorboard ---
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        acc.Reload()
        out: Dict[str, List[Tuple[int, float]]] = {}
        for tag in acc.Tags().get("scalars", []):
            out[tag] = [(e.step, e.value) for e in acc.Scalars(tag)]
        return out
    except Exception as e:
        print(f"[inspect_run] tensorboard path failed: {e}; trying tbparse", file=sys.stderr)

    # --- attempt 2: tbparse ---
    try:
        from tbparse import SummaryReader  # type: ignore
        reader = SummaryReader(str(run_dir), pivot=False)
        df = reader.scalars
        out2: Dict[str, List[Tuple[int, float]]] = {}
        for tag, group in df.groupby("tag"):
            group = group.sort_values("step")
            out2[tag] = list(zip(group["step"].astype(int), group["value"].astype(float)))
        return out2
    except Exception as e:
        print(f"[inspect_run] tbparse failed: {e}; trying raw protobuf", file=sys.stderr)

    # --- attempt 3: raw protobuf via tensorflow ---
    try:
        import tensorflow as tf  # type: ignore
        out3: Dict[str, List[Tuple[int, float]]] = {}
        for f in files:
            for raw in tf.data.TFRecordDataset(f):
                ev = tf.compat.v1.Event.FromString(raw.numpy())
                if not ev.summary:
                    continue
                for v in ev.summary.value:
                    if v.HasField("simple_value"):
                        out3.setdefault(v.tag, []).append((int(ev.step), float(v.simple_value)))
        for k in out3:
            out3[k].sort()
        return out3
    except Exception as e:
        raise SystemExit(
            f"[inspect_run] All readers failed. Install one of:\n"
            f"  pip install tensorboard\n  pip install tbparse\n"
            f"Last error: {e}"
        )


def _summarize(series: List[Tuple[int, float]]) -> dict:
    if not series:
        return {}
    steps = [s for s, _ in series]
    vals = [v for _, v in series]
    peak_idx = max(range(len(vals)), key=lambda i: vals[i])
    min_idx = min(range(len(vals)), key=lambda i: vals[i])
    # last-decile mean to smooth jitter
    tail = vals[max(0, len(vals) - max(1, len(vals) // 10)) :]
    return {
        "start_step": steps[0],
        "start_val": vals[0],
        "end_step": steps[-1],
        "end_val": vals[-1],
        "peak_step": steps[peak_idx],
        "peak_val": vals[peak_idx],
        "min_step": steps[min_idx],
        "min_val": vals[min_idx],
        "tail_mean": sum(tail) / len(tail),
        "n_points": len(series),
    }


def _ckpt_summary(run_dir: Path) -> dict:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(re.findall(r"model_(\d+)\.pt", p.name)[0]))
    if not ckpts:
        return {}
    latest = ckpts[-1]
    iter_n = int(re.findall(r"model_(\d+)\.pt", latest.name)[0])
    return {
        "count": len(ckpts),
        "latest": latest.name,
        "latest_iter": iter_n,
        "size_mb": latest.stat().st_size / 1024 / 1024,
        "all_iters": [int(re.findall(r"model_(\d+)\.pt", p.name)[0]) for p in ckpts],
    }


def _md_table(rows: List[List[str]], header: List[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run", type=str, help="Path to a run directory (host).")
    g.add_argument("--latest", action="store_true", help="Pick newest run under outputs/logs/rsl_rl/.")
    ap.add_argument("--experiment", type=str, default=None, help="Restrict --latest to this experiment subdir.")
    ap.add_argument("--tags", nargs="*", default=None, help="Override the tag whitelist; pass tag names.")
    ap.add_argument("--all-tags", action="store_true", help="Print every scalar tag found.")
    args = ap.parse_args()

    if args.run:
        run_dir = Path(args.run).resolve()
    else:
        run_dir = _find_latest_run(args.experiment)
    if not run_dir.exists():
        raise SystemExit(f"[inspect_run] Not a dir: {run_dir}")

    print(f"# Run inspection\n")
    print(f"- **dir**: `{run_dir}`")
    ck = _ckpt_summary(run_dir)
    if ck:
        print(f"- **checkpoints**: {ck['count']} files, latest `{ck['latest']}` "
              f"(iter {ck['latest_iter']}, {ck['size_mb']:.1f} MB)")
        print(f"- **all iters saved**: {ck['all_iters']}")
    print()

    scalars = _read_scalars(run_dir)
    tags = args.tags or (list(scalars.keys()) if args.all_tags else DEFAULT_TAGS)
    rows = []
    for tag in tags:
        if tag not in scalars:
            continue
        s = _summarize(scalars[tag])
        if not s:
            continue
        rows.append([
            f"`{tag}`",
            f"{s['start_val']:.3g} @ {s['start_step']}",
            f"{s['end_val']:.3g} @ {s['end_step']}",
            f"{s['peak_val']:.3g} @ {s['peak_step']}",
            f"{s['min_val']:.3g} @ {s['min_step']}",
            f"{s['tail_mean']:.3g}",
            str(s["n_points"]),
        ])

    print("## Scalars\n")
    if rows:
        print(_md_table(rows, ["tag", "start", "end", "peak", "min", "tail-mean(10%)", "n"]))
    else:
        print("_no matching tags found_")
        print(f"\navailable tags: {sorted(scalars.keys())}")
    print()


if __name__ == "__main__":
    main()
