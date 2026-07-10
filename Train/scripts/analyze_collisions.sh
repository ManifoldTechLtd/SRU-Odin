#!/usr/bin/env bash
# Run instrumented collision analysis inside the Isaac container, then plot
# figures on the host.
#
# Example:
#   PLAY_MAZE_ONLY=1 PLAY_CLASSIC_MAZE=1 PLAY_DIFFICULTY="0.8,1.0" PLAY_CELL_SIZE=1.0 \
#   ./scripts/analyze_collisions.sh \
#     --task Isaac-Nav-PPO-Go2-Play-v0 \
#     --experiment go2_navigation_ppo_puremaze \
#     --run-dir 2026-06-23_11-13-55_puremaze_tight_corridors \
#     --from-iter 32200 \
#     --envs 32 --steps 4000
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TASK="Isaac-Nav-PPO-Go2-Play-v0"
EXPERIMENT="go2_navigation_ppo_puremaze"
RUN_DIR=""
FROM_ITER=""
CHECKPOINT=""
NUM_ENVS=32
STEPS=4000
SEED=42

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  --task NAME           (default: $TASK)
  --experiment NAME     (default: $EXPERIMENT)
  --run-dir NAME        run dir under outputs/logs/rsl_rl/<experiment>/
  --from-iter N         model_<N>.pt
  --checkpoint PATH     explicit in-container ckpt (overrides above)
  --envs N              (default: $NUM_ENVS)
  --steps N             (default: $STEPS)
  --seed N              (default: $SEED)

Forwarded env vars: PLAY_DIFFICULTY PLAY_SUBTERRAIN_MIX PLAY_MAZE_ONLY
                    PLAY_CELL_SIZE PLAY_CLASSIC_MAZE
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)        TASK="$2"; shift 2 ;;
    --experiment)  EXPERIMENT="$2"; shift 2 ;;
    --run-dir)     RUN_DIR="$2"; shift 2 ;;
    --from-iter)   FROM_ITER="$2"; shift 2 ;;
    --checkpoint)  CHECKPOINT="$2"; shift 2 ;;
    --envs)        NUM_ENVS="$2"; shift 2 ;;
    --steps)       STEPS="$2"; shift 2 ;;
    --seed)        SEED="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *) echo "[error] $1" >&2; usage ;;
  esac
done

HOST_LOG_ROOT="$REPO_ROOT/outputs/logs/rsl_rl/$EXPERIMENT"
if [[ -z "$CHECKPOINT" ]]; then
  [[ -z "$RUN_DIR" || -z "$FROM_ITER" ]] && { echo "[error] need --run-dir + --from-iter or --checkpoint" >&2; exit 1; }
  HOST_CKPT="$HOST_LOG_ROOT/$RUN_DIR/model_${FROM_ITER}.pt"
  [[ -f "$HOST_CKPT" ]] || { echo "[error] not found: $HOST_CKPT" >&2; exit 1; }
  CHECKPOINT="${HOST_CKPT/$REPO_ROOT\/outputs\/logs/\/workspace\/IsaacLab\/logs}"
fi

TAG="${RUN_DIR:-$(basename "$(dirname "$CHECKPOINT")")}_iter${FROM_ITER:-unknown}"
HOST_OUT="$REPO_ROOT/outputs/collisions/$TAG"
CTR_OUT="/workspace/IsaacLab/logs/collisions/$TAG"
mkdir -p "$HOST_OUT"

echo "[collisions] task=$TASK"
echo "[collisions] ckpt(ctr)=$CHECKPOINT"
echo "[collisions] out(host)=$HOST_OUT"

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
  "./isaaclab.sh -p source/isaaclab_nav_task/scripts/analyze_collisions.py \
      --task $TASK \
      --num_envs $NUM_ENVS \
      --steps $STEPS \
      --seed $SEED \
      --checkpoint $CHECKPOINT \
      --out-dir $CTR_OUT \
      --headless && \
   ./isaaclab.sh -p source/isaaclab_nav_task/scripts/plot_collisions.py $CTR_OUT/collisions.npz"
echo "[collisions] done -> $HOST_OUT"
ls -la "$HOST_OUT"
