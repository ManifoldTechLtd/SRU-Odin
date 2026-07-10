#!/usr/bin/env bash
# Run inside the container.
set -e
cd /workspace/IsaacLab
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-v0 \
    --num_envs ${NUM_ENVS:-4096} \
    --headless \
    "$@"
