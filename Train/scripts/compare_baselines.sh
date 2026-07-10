#!/usr/bin/env bash
# Apples-to-apples comparison: my-Go2-policy vs official-B2W-policy on the
# SAME pure-maze scene (same seed, same PLAY_* env vars => same maze layouts).
#
# Bodies differ (Go2 quadruped vs B2W wheeled), but the terrain topology is
# identical, so success-rate / failure-mode breakdowns are directly comparable
# as a "policy + body" combo.
#
# Example:
#   ./scripts/compare_baselines.sh \
#     --go2-run 2026-06-23_11-13-55_puremaze_tight_corridors \
#     --go2-iter 32200 \
#     --b2w-jit assets/baselines/b2w_nav_policy_new.pt \
#     --cell-size 2.0 \
#     --difficulty "0.8,1.0" \
#     --envs 32 --steps 2000 --seed 42
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- defaults ----
GO2_EXPERIMENT="go2_navigation_ppo_puremaze"
GO2_RUN=""
GO2_ITER=""
B2W_JIT=""
CELL_SIZE="2.0"
DIFFICULTY="0.8,1.0"
NUM_ENVS=32
STEPS=2000
SEED=42
OUT_DIR="$REPO_ROOT/outputs/compare/$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<EOF
Usage: $(basename "$0") --go2-run DIR --go2-iter N --b2w-jit PATH [options]

Required (Go2 side):
  --go2-run DIR        Go2 run-dir name (under outputs/logs/rsl_rl/<experiment>/).
  --go2-iter N         Go2 model_<N>.pt iter.

Required (B2W side):
  --b2w-jit PATH       Host path to TorchScript B2W policy (e.g. nav_policy_new.pt).

Options:
  --go2-experiment N   Go2 experiment subdir (default: $GO2_EXPERIMENT).
  --cell-size F        PLAY_CELL_SIZE in meters (default: $CELL_SIZE).
                       NOTE: B2W body is large; cell<1.5 usually impossible.
  --difficulty "l,h"   PLAY_DIFFICULTY band (default: "$DIFFICULTY").
  --envs N             Parallel envs (default: $NUM_ENVS).
  --steps N            env.step() count (default: $STEPS).
  --seed N             Shared seed for both runs (default: $SEED).
  --out-dir PATH       Where to write logs+diff (default: outputs/compare/<ts>).
  -h, --help

Both runs share:
  PLAY_MAZE_ONLY=1                    (100% maze sub-terrain)
  PLAY_CLASSIC_MAZE=1                 (paper-style uniform walls)
  PLAY_DIFFICULTY=\$DIFFICULTY
  PLAY_CELL_SIZE=\$CELL_SIZE
  --seed \$SEED, --num_envs \$NUM_ENVS, --steps \$STEPS
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --go2-run)        GO2_RUN="$2"; shift 2 ;;
    --go2-iter)       GO2_ITER="$2"; shift 2 ;;
    --go2-experiment) GO2_EXPERIMENT="$2"; shift 2 ;;
    --b2w-jit)        B2W_JIT="$2"; shift 2 ;;
    --cell-size)      CELL_SIZE="$2"; shift 2 ;;
    --difficulty)     DIFFICULTY="$2"; shift 2 ;;
    --envs)           NUM_ENVS="$2"; shift 2 ;;
    --steps)          STEPS="$2"; shift 2 ;;
    --seed)           SEED="$2"; shift 2 ;;
    --out-dir)        OUT_DIR="$2"; shift 2 ;;
    -h|--help)        usage ;;
    *) echo "[error] Unknown arg: $1" >&2; usage ;;
  esac
done

[[ -z "$GO2_RUN" || -z "$GO2_ITER" || -z "$B2W_JIT" ]] && \
  { echo "[error] --go2-run, --go2-iter, --b2w-jit are required." >&2; usage; }

mkdir -p "$OUT_DIR"
GO2_LOG="$OUT_DIR/go2_eval.log"
B2W_LOG="$OUT_DIR/b2w_eval.log"
DIFF_TABLE="$OUT_DIR/comparison.md"

# Shared env vars for both runs.
export PLAY_MAZE_ONLY=1
export PLAY_CLASSIC_MAZE=1
export PLAY_DIFFICULTY="$DIFFICULTY"
export PLAY_CELL_SIZE="$CELL_SIZE"

cat <<EOF | tee "$OUT_DIR/setup.txt"
================ COMPARE SETUP ================
out_dir       = $OUT_DIR
seed          = $SEED
envs          = $NUM_ENVS
steps         = $STEPS
PLAY_MAZE_ONLY     = 1
PLAY_CLASSIC_MAZE  = 1
PLAY_DIFFICULTY    = $DIFFICULTY
PLAY_CELL_SIZE     = $CELL_SIZE
----- GO2 -----
  experiment = $GO2_EXPERIMENT
  run-dir    = $GO2_RUN
  iter       = $GO2_ITER
----- B2W -----
  jit policy = $B2W_JIT
================================================
EOF

# ---- 1. Go2 (my policy) on maze ----
echo "[compare] >>> running Go2 eval ..."
"$REPO_ROOT/scripts/eval_terminations.sh" \
  --task Isaac-Nav-PPO-Go2-Play-v0 \
  --experiment "$GO2_EXPERIMENT" \
  --run-dir "$GO2_RUN" \
  --from-iter "$GO2_ITER" \
  --envs "$NUM_ENVS" --steps "$STEPS" --seed "$SEED" \
  2>&1 | tee "$GO2_LOG"

# ---- 2. B2W (official JIT) on maze ----
echo "[compare] >>> running B2W eval ..."
"$REPO_ROOT/scripts/eval_jit_baseline.sh" \
  --task Isaac-Nav-PPO-B2W-Play-v0 \
  --jit-policy "$B2W_JIT" \
  --envs "$NUM_ENVS" --steps "$STEPS" --seed "$SEED" \
  2>&1 | tee "$B2W_LOG"

# ---- 3. Parse both logs and produce a side-by-side table ----
python3 - "$GO2_LOG" "$B2W_LOG" "$DIFF_TABLE" <<'PY'
import re, sys, pathlib
go2_log, b2w_log, out_md = sys.argv[1], sys.argv[2], sys.argv[3]

def parse(path):
    """Extract per-term counts from an eval-style log."""
    text = pathlib.Path(path).read_text()
    rows = {}
    # lines look like: "  early_termination          1234     45.6%  <- success"
    for m in re.finditer(r"^\s{2}([a-z_]+)\s+(\d+)\s+([\d.]+)%", text, re.M):
        rows[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    return rows

g = parse(go2_log)
b = parse(b2w_log)
all_terms = sorted(set(g) | set(b))

# Ordering: success first, then collisions, then time_out, then rest.
priority = {"early_termination": 0, "base_contact": 1, "large_pitch_angle": 2,
            "terrain_fall": 3, "time_out": 4}
all_terms.sort(key=lambda n: (priority.get(n, 99), n))

lines = []
lines.append("# Go2-policy vs official-B2W-policy on identical maze scenes\n")
lines.append("| termination          | Go2 count | Go2 %  | B2W count | B2W %  | meaning |")
lines.append("|----------------------|-----------|--------|-----------|--------|---------|")
meaning = {
    "early_termination": "**success** (reached goal)",
    "base_contact":      "collision (body hit wall)",
    "large_pitch_angle": "tipped over",
    "terrain_fall":      "fell into pit",
    "time_out":          "ran out of time",
}
for t in all_terms:
    gc, gp = g.get(t, (0, 0.0))
    bc, bp = b.get(t, (0, 0.0))
    lines.append(f"| `{t:<20}` | {gc:>9} | {gp:>5.1f}% | {bc:>9} | {bp:>5.1f}% | {meaning.get(t,'')} |")

# Summary delta on success rate.
g_succ = g.get("early_termination", (0, 0.0))[1]
b_succ = b.get("early_termination", (0, 0.0))[1]
lines.append("")
lines.append(f"**Success rate:** Go2 = **{g_succ:.1f}%**, B2W = **{b_succ:.1f}%**, "
             f"Δ = **{g_succ - b_succ:+.1f}** pp (Go2 - B2W)")

out = "\n".join(lines) + "\n"
pathlib.Path(out_md).write_text(out)
print("\n" + "=" * 78)
print(out)
print(f"[compare] wrote {out_md}")
PY

echo ""
echo "[compare] all done."
echo "  setup:      $OUT_DIR/setup.txt"
echo "  go2 log:    $GO2_LOG"
echo "  b2w log:    $B2W_LOG"
echo "  table:      $DIFF_TABLE"
