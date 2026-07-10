#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-shot launcher for the SRU navigation node on Go2 NX.
#
# Combines a conda python env (onnxruntime, scipy, ...) with the system
# ROS Noetic installation (rospy, cv_bridge). The trick:
#   1) source /opt/ros/noetic/setup.bash       -> ROS env
#   2) source <catkin_ws>/devel/setup.bash     -> our package
#   3) conda activate sru_nav                  -> python with onnxruntime
#   4) export PYTHONPATH=<ros dist-packages>:$PYTHONPATH so conda python
#      can still import rospy / cv_bridge / sensor_msgs / etc.
#
# Usage:
#   bash launch_sru_nav.sh                # full launch (joy + static TF + node)
#   bash launch_sru_nav.sh --no-joy       # if /joy is already provided elsewhere
#   bash launch_sru_nav.sh --no-deadman   # TESTING: skip joystick deadman entirely
#                                         # (cmd_vel_ratio=1.0, no /joy needed).
#                                         # Robot WILL move on /goal_pose alone.
#   bash launch_sru_nav.sh --node-only    # run just sru_nav_node (assume joy + TF already up)
# -----------------------------------------------------------------------------

set -eu

ENV_NAME="${ENV_NAME:-sru_nav}"
ROS_DISTRO="${ROS_DISTRO:-noetic}"
CATKIN_WS="${CATKIN_WS:-$HOME/code/odin_sru_nav}"

EXTRA_ARGS=""
NODE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --no-joy)         EXTRA_ARGS="$EXTRA_ARGS launch_joy:=false" ;;
    --no-tf)          EXTRA_ARGS="$EXTRA_ARGS launch_static_tf:=false" ;;
    --no-deadman)     EXTRA_ARGS="$EXTRA_ARGS require_joystick:=false launch_joy:=false" ;;
    --node-only)      NODE_ONLY=1 ;;
    *) EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
  esac
done

# ---- 1) ROS env -------------------------------------------------------------
if [ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  echo "[ERROR] /opt/ros/${ROS_DISTRO}/setup.bash not found"; exit 1
fi
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO}/setup.bash"

if [ -f "${CATKIN_WS}/devel/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "${CATKIN_WS}/devel/setup.bash"
else
  echo "[WARN] ${CATKIN_WS}/devel/setup.bash not found. Did you catkin_make?"
fi

# ---- 2) conda env -----------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Run setup_conda_env.sh first."; exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ---- 3) Make rospy / cv_bridge importable from the conda python -------------
ROS_PY_DIST="/opt/ros/${ROS_DISTRO}/lib/python3/dist-packages"
export PYTHONPATH="${ROS_PY_DIST}:${PYTHONPATH:-}"

# Fix conda<->system libffi ABI conflict that breaks cv_bridge:
#   "libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0"
# Force the system libffi (used to build libp11-kit) to load first.
for _sys_libffi in \
    /lib/aarch64-linux-gnu/libffi.so.7 \
    /usr/lib/aarch64-linux-gnu/libffi.so.7 \
    /lib/x86_64-linux-gnu/libffi.so.7 \
    /usr/lib/x86_64-linux-gnu/libffi.so.7; do
  if [ -f "${_sys_libffi}" ]; then
    export LD_PRELOAD="${_sys_libffi}${LD_PRELOAD:+:${LD_PRELOAD}}"
    echo "[INFO] LD_PRELOAD   += ${_sys_libffi}"
    break
  fi
done

echo "[INFO] python       = $(which python)"
echo "[INFO] PYTHONPATH   = ${PYTHONPATH}"
echo "[INFO] ROS_DISTRO   = ${ROS_DISTRO}"
echo "[INFO] CATKIN_WS    = ${CATKIN_WS}"

# ---- Sanity: catkin wrapper must call conda's python (not system /usr/bin) ---
WRAPPER="${CATKIN_WS}/devel/lib/sru_nav_go2_ros1/sru_nav_node"
if [ -f "${WRAPPER}" ]; then
  WRAPPER_SHEBANG="$(head -n1 "${WRAPPER}")"
  CONDA_PY="$(which python)"
  case "${WRAPPER_SHEBANG}" in
    *"${CONDA_PY}"*|*"/usr/bin/env python"*)
      : # OK
      ;;
    *)
      cat >&2 <<EOF
[WARN] Catkin-generated wrapper shebang is NOT the conda python:
         wrapper : ${WRAPPER}
         shebang : ${WRAPPER_SHEBANG}
         conda py: ${CONDA_PY}
       It will likely fail with 'No module named cv2 / onnxruntime'.
       Fix:
         conda activate ${ENV_NAME}
         cd ${CATKIN_WS}
         catkin_make clean
         catkin_make -DPYTHON_EXECUTABLE=\$(which python3)
EOF
      ;;
  esac
fi

# ---- 4) Launch --------------------------------------------------------------
if [ "$NODE_ONLY" -eq 1 ]; then
  exec rosrun sru_nav_go2_ros1 sru_nav_node \
       __ns:= \
       _vae_model_path:="" _policy_model_path:=""
else
  # shellcheck disable=SC2086
  exec roslaunch sru_nav_go2_ros1 sru_nav_go2.launch $EXTRA_ARGS
fi
