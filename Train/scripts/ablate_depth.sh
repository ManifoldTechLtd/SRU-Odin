#!/usr/bin/env bash
# Depth-ablation diagnostic: does the trained Go2 policy actually USE the depth
# camera, or has it learned to ignore it (perception-blind)?
#
# Runs headless inside the container, rolls out the policy, and compares the
# action with vs without the depth slice zeroed. Prints a verdict.
#
# Examples:
#   # Latest run, latest checkpoint
#   ./scripts/ablate_depth.sh
#
#   # Specific run + iter
#   ./scripts/ablate_depth.sh --run-dir 2026-06-16_09-24-44_phase2 --from-iter 6998
#
#   # More steps / envs for tighter stats
#   ./scripts/ablate_depth.sh --from-iter 6998 --envs 32 --steps 400
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- defaults ----
TASK="Isaac-Nav-PPO-Go2-Play-v0"
EXPERIMENT="go2_navigation_ppo_dev"
NUM_ENVS=16
STEPS=300
FROM_ITER=""
RUN_DIR=""
CHECKPOINT=""
SEED=42

usage() { sed -n '1,18p' "$0"; exit 0; }

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

# ---- resolve checkpoint (same scheme as play_go2.sh) ----
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
[ablate] task        = $TASK
[ablate] checkpoint  = $CHECKPOINT
[ablate] host file   = $HOST_CKPT
[ablate] envs        = $NUM_ENVS
[ablate] steps       = $STEPS
EOF

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --shm-size=16g --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO_ROOT/outputs/logs:/workspace/IsaacLab/logs" \
  -v "$REPO_ROOT/mount/sru-navigation-sim:/workspace/IsaacLab/source/isaaclab_nav_task" \
  -v "$REPO_ROOT/mount/rsl_rl:/workspace/rsl_rl" \
  --entrypoint bash sru-nav:latest -lc \
  "./isaaclab.sh -p source/isaaclab_nav_task/scripts/ablate_depth.py \
      --task $TASK \
      --num_envs $NUM_ENVS \
      --steps $STEPS \
      --seed $SEED \
      --checkpoint $CHECKPOINT \
      --headless"
