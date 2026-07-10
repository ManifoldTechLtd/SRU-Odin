#!/usr/bin/env bash
# Continue training the Go2 SRU navigation policy with explicit CLI flags.
#
# Examples:
#   # Continue from the latest checkpoint of the latest run, train 1000 more iters.
#   ./scripts/continue_go2.sh --iters 1000
#
#   # Resume from a specific iteration of the latest run.
#   ./scripts/continue_go2.sh --iters 2000 --from-iter 1000
#
#   # Resume from a specific run directory + iteration.
#   ./scripts/continue_go2.sh --iters 500 \
#       --run-dir 2026-06-12_08-16-50_resume_from_33 --from-iter 600
#
#   # Pin an exact checkpoint file (in-container path).
#   ./scripts/continue_go2.sh --iters 500 \
#       --checkpoint /workspace/IsaacLab/logs/rsl_rl/go2_navigation_ppo_dev/2026-06-12_08-16-50_resume_from_33/model_1000.pt
#
#   # GUI mode, fewer envs.
#   ./scripts/continue_go2.sh --iters 200 --envs 8 --gui
#
#   # Switch to the full-PPO task (different experiment_name dir).
#   ./scripts/continue_go2.sh --iters 2000 --task Isaac-Nav-PPO-Go2-v0 \
#       --experiment go2_navigation_ppo_ft_from_b2w
#
# Behavior:
#   - --load-optimizer is ON by default (true PPO resume); disable with --no-load-optimizer.
#   - A new timestamped run directory is created; old runs are preserved.
#   - The new run's --max_iterations is the *absolute* PPO iteration target,
#     so set it to (start_iter + extra_iters). For convenience you can pass
#     --extra-iters N which will be added to the resumed start_iter.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- defaults ----
TASK="Isaac-Nav-PPO-Go2-Dev-v0"
EXPERIMENT="go2_navigation_ppo_dev"   # log subdir under outputs/logs/rsl_rl/
NUM_ENVS=1024
ITERS=""        # absolute target iteration count
EXTRA_ITERS=""  # alternative: N more iters past the resumed checkpoint
FROM_ITER=""    # specific iteration number to resume from
RUN_DIR=""      # specific run dir name under <experiment>/
CHECKPOINT=""   # explicit in-container checkpoint path (overrides everything)
SEED=42
RUN_NAME=""     # auto-generated if empty
GUI=0
LOAD_OPTIMIZER=1

usage() {
  sed -n '1,30p' "$0"
  echo
  echo "Flags:"
  echo "  --iters N                Absolute target PPO iteration count (overrides max_iterations)."
  echo "  --extra-iters N          Iters to train past the resumed start (alternative to --iters)."
  echo "  --from-iter N            Resume from model_<N>.pt within the chosen run dir."
  echo "  --run-dir NAME           Run dir name under outputs/logs/rsl_rl/<experiment>/."
  echo "  --checkpoint PATH        Explicit in-container checkpoint path (overrides above)."
  echo "  --task NAME              Gym task id (default: $TASK)."
  echo "  --experiment NAME        Log subdir to scan (default: $EXPERIMENT)."
  echo "  --envs N                 Parallel envs (default: $NUM_ENVS)."
  echo "  --seed N                 Seed (default: $SEED)."
  echo "  --run-name STR           Suffix for the new run dir (default: resume_from_<iter>)."
  echo "  --gui                    Show Isaac Sim GUI (X11)."
  echo "  --no-load-optimizer      Load model only (cold optimizer)."
  echo "  -h, --help               Show this help."
  exit 0
}

# ---- arg parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iters)            ITERS="$2"; shift 2 ;;
    --extra-iters)      EXTRA_ITERS="$2"; shift 2 ;;
    --from-iter)        FROM_ITER="$2"; shift 2 ;;
    --run-dir)          RUN_DIR="$2"; shift 2 ;;
    --checkpoint)       CHECKPOINT="$2"; shift 2 ;;
    --task)             TASK="$2"; shift 2 ;;
    --experiment)       EXPERIMENT="$2"; shift 2 ;;
    --envs)             NUM_ENVS="$2"; shift 2 ;;
    --seed)             SEED="$2"; shift 2 ;;
    --run-name)         RUN_NAME="$2"; shift 2 ;;
    --gui)              GUI=1; shift ;;
    --no-load-optimizer) LOAD_OPTIMIZER=0; shift ;;
    -h|--help)          usage ;;
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
    [[ -f "$HOST_CKPT" ]] || { echo "[error] Checkpoint not found: $HOST_CKPT" >&2; exit 1; }
  else
    HOST_CKPT="$(ls -1 "$RUN_PATH"/model_*.pt 2>/dev/null | sort -V | tail -n 1)"
    [[ -z "$HOST_CKPT" ]] && { echo "[error] No model_*.pt in $RUN_PATH" >&2; exit 1; }
  fi
  CHECKPOINT="${HOST_CKPT/$REPO_ROOT\/outputs\/logs/\/workspace\/IsaacLab\/logs}"
else
  HOST_CKPT="${CHECKPOINT/\/workspace\/IsaacLab\/logs/$REPO_ROOT\/outputs\/logs}"
fi

# ---- derive start iter from filename if possible ----
START_ITER="$(basename "$HOST_CKPT" | sed -nE 's/model_([0-9]+)\.pt/\1/p')"
[[ -z "$START_ITER" ]] && START_ITER=0

# ---- resolve target iteration count ----
if [[ -z "$ITERS" && -n "$EXTRA_ITERS" ]]; then
  ITERS=$(( START_ITER + EXTRA_ITERS ))
fi
if [[ -z "$ITERS" ]]; then
  echo "[error] Must provide --iters N (absolute) or --extra-iters N (relative)." >&2
  exit 1
fi

[[ -z "$RUN_NAME" ]] && RUN_NAME="resume_from_${START_ITER}"

# ---- GUI plumbing ----
HEADLESS_FLAG="--headless"
EXTRA_DOCKER_ARGS=()
if [[ "$GUI" == "1" ]]; then
  HEADLESS_FLAG=""
  xhost +local:root >/dev/null 2>&1 || xhost +local:docker >/dev/null 2>&1 || true
  EXTRA_DOCKER_ARGS+=(-e "DISPLAY=${DISPLAY:-:1}"
                     -v /tmp/.X11-unix:/tmp/.X11-unix:rw
                     -v "$HOME/.Xauthority:/root/.Xauthority:ro")
fi

OPT_FLAG=""
[[ "$LOAD_OPTIMIZER" == "1" ]] && OPT_FLAG="--load_optimizer"

cat <<EOF
[continue_go2] task         = $TASK
[continue_go2] experiment   = $EXPERIMENT
[continue_go2] checkpoint   = $CHECKPOINT
[continue_go2] start iter   = $START_ITER
[continue_go2] target iter  = $ITERS  (= $((ITERS - START_ITER)) more)
[continue_go2] envs         = $NUM_ENVS
[continue_go2] seed         = $SEED
[continue_go2] run_name     = $RUN_NAME
[continue_go2] gui          = $GUI
[continue_go2] load_opt     = $LOAD_OPTIMIZER
EOF

docker run --rm --gpus all \
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
      --max_iterations $ITERS \
      --seed $SEED \
      --run_name $RUN_NAME \
      --checkpoint $CHECKPOINT \
      $OPT_FLAG \
      $HEADLESS_FLAG"
