#!/usr/bin/env bash
# Quick dev/smoke-test (300 iters, tensorboard).
set -e
cd /workspace/IsaacLab
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-Dev-v0 \
    --num_envs ${NUM_ENVS:-32} \
    --headless \
    "$@"
