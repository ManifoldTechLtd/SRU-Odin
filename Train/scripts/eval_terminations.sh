#!/usr/bin/env bash
# Headless termination breakdown evaluator. Counts per-term terminations over
# N steps so we can read the *true* pit-fall rate (terrain_fall) now that the
# Go2 cfg has tightened fall_height_threshold to -0.3m.
#
# Examples:
#   # Latest run, latest ckpt, 64 envs, 2000 steps
#   ./scripts/eval_terminations.sh
#
#   # Specific run + iter + difficulty band
#   PLAY_DIFFICULTY="0.3,0.8" ./scripts/eval_terminations.sh \
#     --run-dir 2026-06-20_12-07-08_phase5_refine_0.3_0.8 --from-iter 27800
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TASK="Isaac-Nav-PPO-Go2-Play-v0"
EXPERIMENT="go2_navigation_ppo_dev"
NUM_ENVS=64
STEPS=2000
FROM_ITER=""
RUN_DIR=""
CHECKPOINT=""
SEED=42

usage() { sed -n '1,15p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-iter)   FROM_ITER="$2"; shift 2 ;;
    --run-dir)     RUN_DIR="$2"; shift 2 ;;
    --checkpoint)  CHECKPOINT="$2"; shift 2 ;;
    --task)        TASK="$2"; shift 2 ;;
    --experiment)  EXPERIMENT="$2"; shift 2 ;;
    --envs)        NUM_ENVS="$2"; shift 2 ;;
    --steps)       STEPS="$2"; shift 2 ;;
    --seed)        SEED="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *) echo "[error] Unknown arg: $1" >&2; usage ;;
  esac
done

HOST_LOG_ROOT="$REPO_ROOT/outputs/logs/rsl_rl/$EXPERIMENT"

if [[ -z "$CHECKPOINT" ]]; then
  if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$(ls -1 "$HOST_LOG_ROOT" 2>/dev/null | sort | tail -n 1 || true)"
    [[ -z "$RUN_DIR" ]] && { echo "[error] No runs under $HOST_LOG_ROOT" >&2; exit 1; }
  fi
  RUN_PATH="$HOST_LOG_ROOT/$RUN_DIR"
  [[ -d "$RUN_PATH" ]] || { echo "[error] Run dir not found: $RUN_PATH" >&2; exit 1; }

  if [[ -n "$FROM_ITER" ]]; then
    HOST_CKPT="$RUN_PATH/model_${FROM_ITER}.pt"
    [[ -f "$HOST_CKPT" ]] || { echo "[error] Checkpoint not found: $HOST_CKPT" >&2; exit 1; }
  else
    HOST_CKPT="$(ls -1 "$RUN_PATH"/model_*.pt 2>/dev/null | sort -V | tail -n 1)"
    [[ -z "$HOST_CKPT" ]] && { echo "[error] No model_*.pt in $RUN_PATH" >&2; exit 1; }
  fi
  CHECKPOINT="${HOST_CKPT/$REPO_ROOT\/outputs\/logs/\/workspace\/IsaacLab\/logs}"
else
  HOST_CKPT="${CHECKPOINT/\/workspace\/IsaacLab\/logs/$REPO_ROOT\/outputs\/logs}"
fi

cat <<EOF
[eval] task        = $TASK
[eval] checkpoint  = $CHECKPOINT
[eval] host file   = $HOST_CKPT
[eval] envs        = $NUM_ENVS
[eval] steps       = $STEPS
[eval] PLAY_DIFFICULTY = ${PLAY_DIFFICULTY:-(default)}
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
  --entrypoint bash sru-nav:latest -lc \
  "./isaaclab.sh -p source/isaaclab_nav_task/scripts/eval_terminations.py \
      --task $TASK \
      --num_envs $NUM_ENVS \
      --steps $STEPS \
      --seed $SEED \
      --checkpoint $CHECKPOINT \
      --headless"
