#!/usr/bin/env bash
# Replay a trained Go2 navigation policy in the Isaac Sim GUI (X11 forwarded).
#
# Usage:
#   ./scripts/play_go2_gui.sh                                  # latest dev run, 4 envs
#   NUM_ENVS=8 ./scripts/play_go2_gui.sh
#   CHECKPOINT=/workspace/IsaacLab/logs/rsl_rl/go2_navigation_ppo_dev/2026-06-12_06-25-41/model_33.pt \
#       ./scripts/play_go2_gui.sh
#
# The CHECKPOINT path must be the IN-CONTAINER path (under /workspace/IsaacLab/logs/...).
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

NUM_ENVS="${NUM_ENVS:-4}"
TASK="${TASK:-Isaac-Nav-PPO-Go2-Play-v0}"
# Which experiment subdir to scan when CHECKPOINT not given. Override with
# EXPERIMENT=go2_navigation_ppo_puremaze for P0 / maze runs.
EXPERIMENT="${EXPERIMENT:-go2_navigation_ppo_dev}"

# Auto-pick the latest checkpoint under <EXPERIMENT> if none was provided.
if [[ -z "${CHECKPOINT:-}" ]]; then
  HOST_LOG_ROOT="$REPO_ROOT/outputs/logs/rsl_rl/$EXPERIMENT"
  LATEST_RUN="$(ls -1 "$HOST_LOG_ROOT" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$LATEST_RUN" ]]; then
    echo "[error] No runs found under $HOST_LOG_ROOT and no CHECKPOINT given." >&2
    exit 1
  fi
  LATEST_CKPT="$(ls -1 "$HOST_LOG_ROOT/$LATEST_RUN"/model_*.pt 2>/dev/null | sort -V | tail -n 1)"
  if [[ -z "$LATEST_CKPT" ]]; then
    echo "[error] No model_*.pt found in $HOST_LOG_ROOT/$LATEST_RUN" >&2
    exit 1
  fi
  # Map host path -> in-container path.
  CHECKPOINT="${LATEST_CKPT/$REPO_ROOT\/outputs\/logs/\/workspace\/IsaacLab\/logs}"
fi

xhost +local:root >/dev/null 2>&1 || xhost +local:docker >/dev/null 2>&1 || true

echo "[host] DISPLAY=${DISPLAY:-:1}"
echo "[host] task=$TASK  num_envs=$NUM_ENVS"
echo "[host] checkpoint (in container) = $CHECKPOINT"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DISPLAY="${DISPLAY:-:1}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PLAY_DIFFICULTY="${PLAY_DIFFICULTY:-}" \
  -e PLAY_CELL_SIZE="${PLAY_CELL_SIZE:-}" \
  -e PLAY_MAZE_ONLY="${PLAY_MAZE_ONLY:-}" \
  -e PLAY_CLASSIC_MAZE="${PLAY_CLASSIC_MAZE:-}" \
  -e PLAY_SUBTERRAIN_MIX="${PLAY_SUBTERRAIN_MIX:-}" \
  --shm-size=16g --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOME/.Xauthority:/root/.Xauthority:ro" \
  -v "$REPO_ROOT/outputs/logs:/workspace/IsaacLab/logs" \
  -v "$REPO_ROOT/mount/sru-navigation-sim:/workspace/IsaacLab/source/isaaclab_nav_task" \
  -v "$REPO_ROOT/mount/rsl_rl:/workspace/rsl_rl" \
  --entrypoint bash sru-nav:latest -lc \
  "./isaaclab.sh -p source/isaaclab_nav_task/scripts/play.py \
      --task $TASK \
      --num_envs $NUM_ENVS \
      --checkpoint $CHECKPOINT"
