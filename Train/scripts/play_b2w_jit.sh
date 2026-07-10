#!/usr/bin/env bash
# Visualize / evaluate a TorchScript JIT navigation policy (e.g. the official
# upstream B2W baseline ``nav_policy_new.pt``) in the IsaacLab pure-maze env.
#
# Examples:
#   # GUI viewer, 4 envs, paper-style classic maze
#   PLAY_CLASSIC_MAZE=1 PLAY_DIFFICULTY="0.8,1.0" \
#   ./scripts/play_b2w_jit.sh --jit-policy assets/baselines/b2w_nav_policy_new.pt
#
#   # Headless stats run (no GUI)
#   PLAY_CLASSIC_MAZE=1 PLAY_DIFFICULTY="0.8,1.0" \
#   ./scripts/play_b2w_jit.sh \
#     --jit-policy assets/baselines/b2w_nav_policy_new.pt \
#     --headless --envs 32 --steps 2000
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- defaults (GUI-friendly) ----
TASK="Isaac-Nav-PPO-B2W-Play-v0"
JIT_HOST=""
NUM_ENVS=4
STEPS=100000        # huge so the viewer just keeps running until user closes it
SEED=42
HEADLESS=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --jit-policy PATH [options]

Required:
  --jit-policy PATH    Host path to TorchScript .pt policy
                       (mounted into /workspace/baseline_policy.pt).

Options:
  --task NAME          Gym task id (default: $TASK).
  --envs N             Parallel envs (default: $NUM_ENVS — GUI-friendly).
  --steps N            env.step() calls (default: $STEPS).
  --seed N             (default: $SEED).
  --headless           Run without GUI (faster, no X11).
  -h, --help           Show this help.

Forwarded env vars (set on host):
  PLAY_DIFFICULTY="lo,hi"        terrain difficulty band
  PLAY_SUBTERRAIN_MIX=...        e.g. "maze=1,non_maze=0,pits=0"
  PLAY_MAZE_ONLY=1               100% maze sub-terrain
  PLAY_CELL_SIZE=2.0             m per maze cell (B2W needs >= ~1.5)
  PLAY_CLASSIC_MAZE=1            paper-style uniform walls
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jit-policy) JIT_HOST="$2"; shift 2 ;;
    --task)       TASK="$2"; shift 2 ;;
    --envs)       NUM_ENVS="$2"; shift 2 ;;
    --steps)      STEPS="$2"; shift 2 ;;
    --seed)       SEED="$2"; shift 2 ;;
    --headless)   HEADLESS=1; shift ;;
    -h|--help)    usage ;;
    *) echo "[error] Unknown arg: $1" >&2; usage ;;
  esac
done

[[ -z "$JIT_HOST" ]] && { echo "[error] --jit-policy is required." >&2; exit 1; }
if [[ "$JIT_HOST" != /* ]]; then JIT_HOST="$REPO_ROOT/$JIT_HOST"; fi
[[ -f "$JIT_HOST" ]] || { echo "[error] JIT file not found: $JIT_HOST" >&2; exit 1; }

JIT_CTR="/workspace/baseline_policy.pt"

# ---- GUI plumbing (same as play_go2.sh) ----
EXTRA_DOCKER_ARGS=()
HEADLESS_FLAG=""
if [[ "$HEADLESS" == "0" ]]; then
  xhost +local:root >/dev/null 2>&1 || xhost +local:docker >/dev/null 2>&1 || true
  EXTRA_DOCKER_ARGS+=(-e "DISPLAY=${DISPLAY:-:1}"
                     -v /tmp/.X11-unix:/tmp/.X11-unix:rw
                     -v "$HOME/.Xauthority:/root/.Xauthority:ro")
else
  HEADLESS_FLAG="--headless"
fi

cat <<EOF
[play_b2w_jit] task        = $TASK
[play_b2w_jit] jit (host)  = $JIT_HOST
[play_b2w_jit] jit (ctr)   = $JIT_CTR
[play_b2w_jit] envs        = $NUM_ENVS
[play_b2w_jit] steps       = $STEPS
[play_b2w_jit] headless    = $HEADLESS
[play_b2w_jit] PLAY_DIFFICULTY      = ${PLAY_DIFFICULTY:-(default)}
[play_b2w_jit] PLAY_SUBTERRAIN_MIX  = ${PLAY_SUBTERRAIN_MIX:-(default)}
[play_b2w_jit] PLAY_MAZE_ONLY       = ${PLAY_MAZE_ONLY:-(default)}
[play_b2w_jit] PLAY_CELL_SIZE       = ${PLAY_CELL_SIZE:-(default)}
[play_b2w_jit] PLAY_CLASSIC_MAZE    = ${PLAY_CLASSIC_MAZE:-(default)}
EOF

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PLAY_DIFFICULTY="${PLAY_DIFFICULTY:-}" \
  -e PLAY_SUBTERRAIN_MIX="${PLAY_SUBTERRAIN_MIX:-}" \
  -e PLAY_MAZE_ONLY="${PLAY_MAZE_ONLY:-}" \
  -e PLAY_CELL_SIZE="${PLAY_CELL_SIZE:-}" \
  -e PLAY_CLASSIC_MAZE="${PLAY_CLASSIC_MAZE:-}" \
  --shm-size=16g --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  "${EXTRA_DOCKER_ARGS[@]}" \
  -v "$REPO_ROOT/outputs/logs:/workspace/IsaacLab/logs" \
  -v "$REPO_ROOT/mount/sru-navigation-sim:/workspace/IsaacLab/source/isaaclab_nav_task" \
  -v "$REPO_ROOT/mount/rsl_rl:/workspace/rsl_rl" \
  -v "$JIT_HOST:$JIT_CTR:ro" \
  --entrypoint bash sru-nav:latest -lc \
  "./isaaclab.sh -p source/isaaclab_nav_task/scripts/eval_jit_policy.py \
      --task $TASK \
      --num_envs $NUM_ENVS \
      --steps $STEPS \
      --seed $SEED \
      --jit_policy $JIT_CTR \
      $HEADLESS_FLAG"
