#!/usr/bin/env bash
# Evaluate a TorchScript JIT navigation policy (e.g. the official B2W baseline
# nav_policy_new.pt) on the IsaacLab navigation env, with the same PLAY_*
# terrain env vars as eval_terminations.sh.
#
# Example: baseline on dense paper-style maze
#   PLAY_DIFFICULTY="0.8,1.0" \
#   PLAY_SUBTERRAIN_MIX="maze=1,non_maze=0,pits=0" \
#   PLAY_CLASSIC_MAZE=1 \
#   ./scripts/eval_jit_baseline.sh \
#     --task Isaac-Nav-PPO-B2W-Play-v0 \
#     --jit-policy assets/baselines/b2w_nav_policy_new.pt \
#     --envs 32 --steps 2000
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TASK="Isaac-Nav-PPO-B2W-Play-v0"
JIT_HOST=""
NUM_ENVS=32
STEPS=2000
SEED=42

usage() {
  cat <<EOF
Usage: $(basename "$0") --jit-policy PATH [options]

Required:
  --jit-policy PATH      Host path to a TorchScript .pt policy.
                         (will be mounted into /workspace at /jit_policy.pt)

Options:
  --task NAME            Gym task id (default: $TASK).
  --envs N               Parallel envs (default: $NUM_ENVS).
  --steps N              env.step() calls (default: $STEPS).
  --seed N               (default: $SEED).
  -h, --help             Show this help.

Forwarded env vars (set on host):
  PLAY_DIFFICULTY="lo,hi"
  PLAY_SUBTERRAIN_MIX="maze=1,non_maze=0,pits=0"
  PLAY_MAZE_ONLY=1
  PLAY_CELL_SIZE=2.0
  PLAY_CLASSIC_MAZE=1
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
    -h|--help)    usage ;;
    *) echo "[error] Unknown arg: $1" >&2; usage ;;
  esac
done

[[ -z "$JIT_HOST" ]] && { echo "[error] --jit-policy is required." >&2; exit 1; }
# Resolve to absolute path
if [[ "$JIT_HOST" != /* ]]; then JIT_HOST="$REPO_ROOT/$JIT_HOST"; fi
[[ -f "$JIT_HOST" ]] || { echo "[error] JIT file not found: $JIT_HOST" >&2; exit 1; }

JIT_CTR="/workspace/baseline_policy.pt"

cat <<EOF
[eval-jit] task        = $TASK
[eval-jit] jit (host)  = $JIT_HOST
[eval-jit] jit (ctr)   = $JIT_CTR
[eval-jit] envs        = $NUM_ENVS
[eval-jit] steps       = $STEPS
[eval-jit] PLAY_DIFFICULTY      = ${PLAY_DIFFICULTY:-(default)}
[eval-jit] PLAY_SUBTERRAIN_MIX  = ${PLAY_SUBTERRAIN_MIX:-(default)}
[eval-jit] PLAY_MAZE_ONLY       = ${PLAY_MAZE_ONLY:-(default)}
[eval-jit] PLAY_CELL_SIZE       = ${PLAY_CELL_SIZE:-(default)}
[eval-jit] PLAY_CLASSIC_MAZE    = ${PLAY_CLASSIC_MAZE:-(default)}
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
      --headless"
