#!/usr/bin/env bash
# Continue training the Go2 SRU navigation policy from the latest dev checkpoint.
#
# Defaults: headless (no GUI for speed), Dev task (same cfg as the previous
# 33-iter run), warm-start with optimizer state so PPO truly resumes.
#
# Usage:
#   ./scripts/train_go2_continue.sh                          # 1000 iter, 24 envs
#   NUM_ENVS=32 MAX_ITER=2000 ./scripts/train_go2_continue.sh
#   CHECKPOINT=/workspace/IsaacLab/logs/.../model_33.pt ./scripts/train_go2_continue.sh
#   GUI=1 ./scripts/train_go2_continue.sh                    # show Isaac Sim GUI
#   TASK=Isaac-Nav-PPO-Go2-v0 ./scripts/train_go2_continue.sh # full PPO task (5000 iter cap)
#
# Notes:
#   * CHECKPOINT must be the IN-CONTAINER path under /workspace/IsaacLab/logs/...
#   * --load_optimizer keeps PPO momentum / adaptive-LR state, so loss curves
#     look continuous instead of restarting cold.
#   * RTX 5070 has 12 GB VRAM. 24 envs headless is conservative; tune up if free.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

NUM_ENVS="${NUM_ENVS:-24}"
MAX_ITER="${MAX_ITER:-1000}"
TASK="${TASK:-Isaac-Nav-PPO-Go2-Dev-v0}"
GUI="${GUI:-0}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-resume_from_33}"

# Auto-pick the latest dev checkpoint if none provided.
if [[ -z "${CHECKPOINT:-}" ]]; then
  HOST_LOG_ROOT="$REPO_ROOT/outputs/logs/rsl_rl/go2_navigation_ppo_dev"
  LATEST_RUN="$(ls -1 "$HOST_LOG_ROOT" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$LATEST_RUN" ]]; then
    echo "[error] No previous runs under $HOST_LOG_ROOT and no CHECKPOINT given." >&2
    exit 1
  fi
  LATEST_CKPT="$(ls -1 "$HOST_LOG_ROOT/$LATEST_RUN"/model_*.pt 2>/dev/null | sort -V | tail -n 1)"
  if [[ -z "$LATEST_CKPT" ]]; then
    echo "[error] No model_*.pt found in $HOST_LOG_ROOT/$LATEST_RUN" >&2
    exit 1
  fi
  CHECKPOINT="${LATEST_CKPT/$REPO_ROOT\/outputs\/logs/\/workspace\/IsaacLab\/logs}"
fi

HEADLESS_FLAG="--headless"
EXTRA_DOCKER_ARGS=()
if [[ "$GUI" == "1" ]]; then
  HEADLESS_FLAG=""
  xhost +local:root >/dev/null 2>&1 || xhost +local:docker >/dev/null 2>&1 || true
  EXTRA_DOCKER_ARGS+=(-e "DISPLAY=${DISPLAY:-:1}"
                     -v /tmp/.X11-unix:/tmp/.X11-unix:rw
                     -v "$HOME/.Xauthority:/root/.Xauthority:ro")
fi

echo "[host] task=$TASK  num_envs=$NUM_ENVS  max_iter=$MAX_ITER  gui=$GUI"
echo "[host] checkpoint (in container) = $CHECKPOINT"
echo "[host] run_name=$RUN_NAME  seed=$SEED"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --shm-size=16g --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  "${EXTRA_DOCKER_ARGS[@]}" \
  -v "$REPO_ROOT/outputs/logs:/workspace/IsaacLab/logs" \
  -v "$REPO_ROOT/mount/sru-navigation-sim:/workspace/IsaacLab/source/isaaclab_nav_task" \
  -v "$REPO_ROOT/mount/rsl_rl:/workspace/rsl_rl" \
  --entrypoint bash sru-nav:latest -lc \
  "./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
      --task $TASK \
      --num_envs $NUM_ENVS \
      --max_iterations $MAX_ITER \
      --seed $SEED \
      --run_name $RUN_NAME \
      --checkpoint $CHECKPOINT \
      --load_optimizer \
      $HEADLESS_FLAG"
