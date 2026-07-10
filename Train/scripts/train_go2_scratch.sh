#!/usr/bin/env bash
# Train the Go2 SRU navigation policy FROM SCRATCH (no checkpoint).
#
# Defaults: headless, 24 envs, 1000 iter, Dev task, seed=42.
# Use this when you intentionally want a cold start (e.g. after sweeping
# reward / action / terrain shaping where the previous policy mean has
# collapsed into a degenerate solution and resuming would just re-learn it).
#
# Usage:
#   ./scripts/train_go2_scratch.sh                          # 1000 iter, 24 envs, headless
#   NUM_ENVS=16 MAX_ITER=500 ./scripts/train_go2_scratch.sh
#   GUI=1 NUM_ENVS=4 MAX_ITER=200 ./scripts/train_go2_scratch.sh
#   TASK=Isaac-Nav-PPO-Go2-v0 RUN_NAME=scratch_full ./scripts/train_go2_scratch.sh
#
# Notes:
#   * No --checkpoint flag is passed, so the agent + critic are re-initialised.
#   * RTX 5070 (12 GB VRAM): 24 envs headless is the safe default. GUI + 32
#     envs OOMs (G0.0 lesson).
#   * Output goes to outputs/logs/rsl_rl/<EXPERIMENT>/<timestamp>_<RUN_NAME>/.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

NUM_ENVS="${NUM_ENVS:-24}"
MAX_ITER="${MAX_ITER:-1000}"
TASK="${TASK:-Isaac-Nav-PPO-Go2-Dev-v0}"
GUI="${GUI:-0}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-scratch_reward_fix_v1}"
# GPU selection. Set GPU=0 / GPU=1 / GPU=0,1 to pin specific cards.
# Empty (default) = all visible GPUs (--gpus all). The container sees the selected
# GPU(s) as cuda:0,cuda:1,... so the training process always uses cuda:0 inside.
GPU="${GPU:-}"

GPU_FLAG="--gpus all"
if [[ -n "$GPU" ]]; then
  GPU_FLAG="--gpus device=${GPU}"
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

cat <<EOF
[train_go2_scratch] task      = $TASK
[train_go2_scratch] num_envs  = $NUM_ENVS
[train_go2_scratch] max_iter  = $MAX_ITER
[train_go2_scratch] seed      = $SEED
[train_go2_scratch] run_name  = $RUN_NAME
[train_go2_scratch] gui       = $GUI
[train_go2_scratch] gpu       = ${GPU:-all}
[train_go2_scratch] (no --checkpoint: cold start)
EOF

docker run --rm $GPU_FLAG \
  --name "go2_train_gpu${GPU:-all}_${RUN_NAME}_$$" \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e DEV_DIFFICULTY="${DEV_DIFFICULTY:-}" \
  -e GO2_DIFFICULTY="${GO2_DIFFICULTY:-}" \
  -e GO2_INIT_LEVEL="${GO2_INIT_LEVEL:-}" \
  -e GO2_CELL_SIZE="${GO2_CELL_SIZE:-}" \
  -e PUREMAZE_DIFFICULTY="${PUREMAZE_DIFFICULTY:-}" \
  -e PUREMAZE_CELL_SIZE="${PUREMAZE_CELL_SIZE:-}" \
  -e PUREMAZE_NUM_ROWS="${PUREMAZE_NUM_ROWS:-}" \
  -e PUREMAZE_NUM_COLS="${PUREMAZE_NUM_COLS:-}" \
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
      $HEADLESS_FLAG"
