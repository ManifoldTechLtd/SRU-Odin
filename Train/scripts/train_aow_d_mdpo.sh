#!/usr/bin/env bash
set -e
cd /workspace/IsaacLab
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-MDPO-AoW-D-v0 \
    --num_envs ${NUM_ENVS:-2048} \
    --max_iterations ${MAX_ITERS:-10000} \
    --headless \
    "$@"
