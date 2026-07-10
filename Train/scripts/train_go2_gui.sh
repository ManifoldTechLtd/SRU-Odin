#!/usr/bin/env bash
# Train Go2 nav with the Isaac Sim GUI visible (X11 forwarded from host).
# Default: 30 iterations, small env count, so you can watch the dogs walking.
#
# Usage:
#   ./scripts/train_go2_gui.sh                # 30 iter, 32 envs, GUI
#   NUM_ENVS=16 MAX_ITER=100 ./scripts/train_go2_gui.sh
#
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

NUM_ENVS="${NUM_ENVS:-32}"
MAX_ITER="${MAX_ITER:-30}"
TASK="${TASK:-Isaac-Nav-PPO-Go2-Dev-v0}"

# Allow the docker user (any UID) to talk to the host X server.
# This affects only the local user's X server.
xhost +local:root >/dev/null 2>&1 || xhost +local:docker >/dev/null 2>&1 || true

echo "[host] DISPLAY=${DISPLAY:-:1}"
echo "[host] task=$TASK  num_envs=$NUM_ENVS  max_iterations=$MAX_ITER"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DISPLAY="${DISPLAY:-:1}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --shm-size=16g --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOME/.Xauthority:/root/.Xauthority:ro" \
  -v "$REPO_ROOT/outputs/logs:/workspace/IsaacLab/logs" \
  -v "$REPO_ROOT/mount/sru-navigation-sim:/workspace/IsaacLab/source/isaaclab_nav_task" \
  -v "$REPO_ROOT/mount/rsl_rl:/workspace/rsl_rl" \
  --entrypoint bash sru-nav:latest -lc \
  "./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
      --task $TASK \
      --num_envs $NUM_ENVS \
      --max_iterations $MAX_ITER"
