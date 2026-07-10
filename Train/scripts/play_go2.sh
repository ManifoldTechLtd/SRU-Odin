#!/usr/bin/env bash
# Replay a trained Go2 SRU navigation policy in the Isaac Sim GUI.
#
# Examples:
#   # Latest run, latest checkpoint, 4 envs (default)
#   ./scripts/play_go2.sh
#
#   # Specific iter from latest run
#   ./scripts/play_go2.sh --from-iter 1000
#
#   # Specific iter from a specific run dir
#   ./scripts/play_go2.sh \
#       --run-dir 2026-06-12_09-32-44_resume_from_1000 \
#       --from-iter 50999
#
#   # Pin an exact in-container checkpoint path
#   ./scripts/play_go2.sh --checkpoint \
#       /workspace/IsaacLab/logs/rsl_rl/go2_navigation_ppo_dev/<run>/model_500.pt
#
#   # Headless replay (no GUI), e.g. on a server
#   ./scripts/play_go2.sh --from-iter 800 --headless --envs 1
#
#   # Browse a different experiment dir (e.g. full PPO)
#   ./scripts/play_go2.sh --experiment go2_navigation_ppo_ft_from_b2w --from-iter 5000
#
# Notes:
#   * --run-dir is the timestamped folder name under
#     outputs/logs/rsl_rl/<experiment>/.
#   * If neither --run-dir nor --checkpoint is given, the latest run is picked.
#   * If --from-iter is omitted, the latest model_*.pt in the run dir is used.
#   * The Play task disables corruption / pushes / large terrain to make the
#     viewer behave well; that is decided by --task (default: Play-v0).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- defaults ----
TASK="Isaac-Nav-PPO-Go2-Play-v0"
EXPERIMENT="go2_navigation_ppo_dev"
NUM_ENVS=4
FROM_ITER=""
RUN_DIR=""
CHECKPOINT=""
HEADLESS=0
SEED=42
EXPORT_ONNX=0
EXPORT_JIT=0

usage() {
  sed -n '1,30p' "$0"
  echo
  echo "Flags:"
  echo "  --from-iter N        Replay model_<N>.pt within the chosen run dir."
  echo "  --run-dir NAME       Run dir name under outputs/logs/rsl_rl/<experiment>/."
  echo "  --checkpoint PATH    Explicit IN-CONTAINER path (overrides --run-dir/--from-iter)."
  echo "  --task NAME          Gym task id (default: $TASK)."
  echo "  --experiment NAME    Log subdir (default: $EXPERIMENT)."
  echo "  --envs N             Parallel envs (default: $NUM_ENVS)."
  echo "  --seed N             Seed (default: $SEED)."
  echo "  --headless           Run without GUI (faster, no X11)."
  echo "  --export-onnx        Also export policy to <run-dir>/export/policy.onnx."
  echo "  --export-jit         Also export policy to <run-dir>/export/policy.pt (TorchScript)."
  echo "  -h, --help           Show this help."
  exit 0
}

# ---- arg parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-iter)   FROM_ITER="$2"; shift 2 ;;
    --run-dir)     RUN_DIR="$2"; shift 2 ;;
    --checkpoint)  CHECKPOINT="$2"; shift 2 ;;
    --task)        TASK="$2"; shift 2 ;;
    --experiment)  EXPERIMENT="$2"; shift 2 ;;
    --envs)        NUM_ENVS="$2"; shift 2 ;;
    --seed)        SEED="$2"; shift 2 ;;
    --headless)    HEADLESS=1; shift ;;
    --export-onnx) EXPORT_ONNX=1; shift ;;
    --export-jit)  EXPORT_JIT=1; shift ;;
    -h|--help)     usage ;;
    *) echo "[error] Unknown arg: $1" >&2; usage ;;
  esac
done

HOST_LOG_ROOT="$REPO_ROOT/outputs/logs/rsl_rl/$EXPERIMENT"

# ---- resolve checkpoint ----
if [[ -z "$CHECKPOINT" ]]; then
  if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$(ls -1 "$HOST_LOG_ROOT" 2>/dev/null | sort | tail -n 1 || true)"
    [[ -z "$RUN_DIR" ]] && { echo "[error] No runs under $HOST_LOG_ROOT" >&2; exit 1; }
  fi
  RUN_PATH="$HOST_LOG_ROOT/$RUN_DIR"
  [[ -d "$RUN_PATH" ]] || { echo "[error] Run dir not found: $RUN_PATH" >&2; exit 1; }

  if [[ -n "$FROM_ITER" ]]; then
    HOST_CKPT="$RUN_PATH/model_${FROM_ITER}.pt"
    if [[ ! -f "$HOST_CKPT" ]]; then
      echo "[error] Checkpoint not found: $HOST_CKPT" >&2
      echo "[hint] Available iters in $RUN_DIR:" >&2
      ls -1 "$RUN_PATH"/model_*.pt 2>/dev/null \
        | sed -nE 's/.*model_([0-9]+)\.pt/  \1/p' \
        | paste -sd ' ' >&2
      exit 1
    fi
  else
    HOST_CKPT="$(ls -1 "$RUN_PATH"/model_*.pt 2>/dev/null | sort -V | tail -n 1)"
    [[ -z "$HOST_CKPT" ]] && { echo "[error] No model_*.pt in $RUN_PATH" >&2; exit 1; }
  fi
  CHECKPOINT="${HOST_CKPT/$REPO_ROOT\/outputs\/logs/\/workspace\/IsaacLab\/logs}"
else
  HOST_CKPT="${CHECKPOINT/\/workspace\/IsaacLab\/logs/$REPO_ROOT\/outputs\/logs}"
fi

# ---- GUI plumbing ----
EXTRA_DOCKER_ARGS=()
HEADLESS_FLAG=""
if [[ "$HEADLESS" == "0" ]]; then
  xhost +local:root >/dev/null 2>&1 || xhost +local:docker >/dev/null 2>&1 || true
  EXTRA_DOCKER_ARGS+=(-e "DISPLAY=${DISPLAY:-:1}"
                     -v /tmp/.X11-unix:/tmp/.X11-unix:rw
                     -v "$HOME/.Xauthority:/root/.Xauthority:ro")
else
  HEADLESS_FLAG="--headless"
fi

cat <<EOF
[play_go2] task        = $TASK
[play_go2] experiment  = $EXPERIMENT
[play_go2] checkpoint  = $CHECKPOINT
[play_go2] host file   = $HOST_CKPT
[play_go2] envs        = $NUM_ENVS
[play_go2] headless    = $HEADLESS
EOF

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PLAY_DIFFICULTY="${PLAY_DIFFICULTY:-}" \
  -e PLAY_SUBTERRAIN_MIX="${PLAY_SUBTERRAIN_MIX:-}" \
  -e PLAY_MAZE_ONLY="${PLAY_MAZE_ONLY:-}" \
  -e PLAY_CELL_SIZE="${PLAY_CELL_SIZE:-}" \
  -e PLAY_CLASSIC_MAZE="${PLAY_CLASSIC_MAZE:-}" \
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
  "./isaaclab.sh -p source/isaaclab_nav_task/scripts/play.py \
      --task $TASK \
      --num_envs $NUM_ENVS \
      --seed $SEED \
      --checkpoint $CHECKPOINT \
      $HEADLESS_FLAG \
      $([[ $EXPORT_ONNX == 1 ]] && echo --export_onnx) \
      $([[ $EXPORT_JIT == 1 ]] && echo --export_jit)"
