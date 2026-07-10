#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# verify_port.sh — automated acceptance checks for sru_nav_go2_ros1
#
# Run after generating (or updating) the package to confirm the port is
# self-consistent and ready for hardware testing.
#
# Usage:
#   bash scripts/verify_port.sh                      # full check
#   bash scripts/verify_port.sh --skip-roslaunch     # skip the live node test
#   ENV_NAME=my_env CATKIN_WS=/path bash scripts/verify_port.sh
#
# Exit code: 0 if everything passes, 1 otherwise.
# -----------------------------------------------------------------------------

set -u

ENV_NAME="${ENV_NAME:-sru_nav}"
ROS_DISTRO="${ROS_DISTRO:-noetic}"
CATKIN_WS="${CATKIN_WS:-$HOME/code/odin_sru_nav}"

PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SKIP_ROSLAUNCH=0
for arg in "$@"; do
  case "$arg" in
    --skip-roslaunch) SKIP_ROSLAUNCH=1 ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
  esac
done

PASS=0
FAIL=0
WARN=0

c_red()    { printf "\033[31m%s\033[0m" "$1"; }
c_green()  { printf "\033[32m%s\033[0m" "$1"; }
c_yellow() { printf "\033[33m%s\033[0m" "$1"; }

ok()    { echo "  [$(c_green PASS)] $1"; PASS=$((PASS+1)); }
fail()  { echo "  [$(c_red   FAIL)] $1"; FAIL=$((FAIL+1)); }
warn()  { echo "  [$(c_yellow WARN)] $1"; WARN=$((WARN+1)); }
info()  { echo "  [INFO] $1"; }

section() { echo; echo "=== $1 ==="; }

# -----------------------------------------------------------------------------
section "0. Environment"
# -----------------------------------------------------------------------------

info "PKG_DIR    = ${PKG_DIR}"
info "CATKIN_WS  = ${CATKIN_WS}"
info "ENV_NAME   = ${ENV_NAME}"
info "ROS_DISTRO = ${ROS_DISTRO}"

# OS
if grep -q "Ubuntu 20.04" /etc/os-release 2>/dev/null; then
  ok "Ubuntu 20.04 (Noetic-compatible)"
else
  warn "OS is not Ubuntu 20.04 — Noetic may not run cleanly"
fi

# ROS
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  ok "ROS ${ROS_DISTRO} installed"
else
  fail "/opt/ros/${ROS_DISTRO}/setup.bash not found"
fi

# Conda
if command -v conda >/dev/null 2>&1; then
  ok "conda available ($(conda --version))"
else
  fail "conda not found in PATH"
fi

# System clock
year=$(date +%Y)
if [ "$year" -ge 2024 ] && [ "$year" -le 2099 ]; then
  ok "system clock looks sane ($(date '+%Y-%m-%d %H:%M:%S'))"
else
  fail "system clock implausible ($(date)) — pip TLS will fail"
fi

# libffi
if ls /lib/*-linux-gnu/libffi.so.7 2>/dev/null >/dev/null; then
  ok "system libffi.so.7 present (LD_PRELOAD target)"
elif ls /lib/*-linux-gnu/libffi.so.8 2>/dev/null >/dev/null; then
  warn "only libffi.so.8 found — update launch_sru_nav.sh accordingly"
else
  fail "no system libffi.so.* — cv_bridge will likely fail"
fi

# -----------------------------------------------------------------------------
section "1. Package layout"
# -----------------------------------------------------------------------------

req_files=(
  "package.xml"
  "CMakeLists.txt"
  "setup.py"
  "launch/sru_nav_go2.launch"
  "config/sru_nav.yaml"
  "scripts/sru_nav_node"
  "scripts/launch_sru_nav.sh"
  "scripts/setup_conda_env.sh"
  "src/sru_nav_go2/__init__.py"
  "src/sru_nav_go2/constants.py"
  "src/sru_nav_go2/model.py"
  "src/sru_nav_go2/utils.py"
  "src/sru_nav_go2/visualization.py"
  "src/sru_nav_go2/waypoint_manager.py"
  "src/sru_nav_go2/navigation_policy_node.py"
  "models/vae_encoder.onnx"
  "models/nav_policy.onnx"
)

for f in "${req_files[@]}"; do
  if [ -e "${PKG_DIR}/${f}" ]; then
    ok "${f}"
  else
    fail "missing: ${f}"
  fi
done

# -----------------------------------------------------------------------------
section "2. YAML config — required parameters"
# -----------------------------------------------------------------------------

YAML="${PKG_DIR}/config/sru_nav.yaml"
if [ -f "$YAML" ]; then
  for key in depth_topic odom_topic joy_topic goal_topic cmd_vel_topic \
             min_depth max_depth control_frequency policy_scale \
             use_sim require_joystick; do
    if grep -E "^[[:space:]]*${key}[[:space:]]*:" "$YAML" >/dev/null; then
      ok "yaml has ${key}"
    else
      fail "yaml missing ${key}"
    fi
  done
else
  fail "yaml not found"
fi

# -----------------------------------------------------------------------------
section "3. Constants — training-aligned numerics"
# -----------------------------------------------------------------------------

CONSTS="${PKG_DIR}/src/sru_nav_go2/constants.py"
if [ -f "$CONSTS" ]; then
  declare -A want=(
    [DEFAULT_CONTROL_FREQUENCY]="5"
    [DEFAULT_MIN_DEPTH]="0.25"
    [DEFAULT_MAX_DEPTH]="10"
    [JOYSTICK_TIMEOUT]="15"
  )
  for k in "${!want[@]}"; do
    if grep -E "^${k}\s*=\s*${want[$k]}" "$CONSTS" >/dev/null; then
      ok "${k} = ${want[$k]}"
    else
      warn "${k} not exactly ${want[$k]} — verify against training cfg"
    fi
  done
else
  fail "constants.py not found"
fi

# -----------------------------------------------------------------------------
section "4. Launch script — coexistence fixes"
# -----------------------------------------------------------------------------

LAUNCH_SH="${PKG_DIR}/scripts/launch_sru_nav.sh"
if [ -f "$LAUNCH_SH" ]; then
  grep -q "LD_PRELOAD"  "$LAUNCH_SH" && ok "LD_PRELOAD libffi fix present" \
                                     || fail "no LD_PRELOAD libffi fix"
  grep -q "PYTHONPATH"  "$LAUNCH_SH" && ok "ROS dist-packages injected to PYTHONPATH" \
                                     || fail "no PYTHONPATH injection"
  grep -q "conda activate" "$LAUNCH_SH" && ok "conda activate in launcher" \
                                        || fail "no conda activate"
  grep -q -- "--no-deadman" "$LAUNCH_SH" && ok "--no-deadman bypass present" \
                                         || warn "no --no-deadman flag"
fi

SETUP_SH="${PKG_DIR}/scripts/setup_conda_env.sh"
if [ -f "$SETUP_SH" ]; then
  for pkg in onnxruntime netifaces defusedxml opencv-python rospkg empy; do
    grep -q "$pkg" "$SETUP_SH" \
      && ok "setup installs ${pkg}" \
      || fail "setup script missing ${pkg}"
  done
fi

# -----------------------------------------------------------------------------
section "5. ONNX models — load test"
# -----------------------------------------------------------------------------

if command -v python >/dev/null && python -c "import onnxruntime" 2>/dev/null; then
  for f in "models/vae_encoder.onnx" "models/nav_policy.onnx"; do
    full="${PKG_DIR}/${f}"
    if [ -f "$full" ]; then
      if python - <<EOF 2>/dev/null
import onnxruntime as ort, sys
s = ort.InferenceSession("${full}", providers=["CPUExecutionProvider"])
print("  ${f} inputs:", [(i.name, i.shape) for i in s.get_inputs()])
print("  ${f} outputs:", [(o.name, o.shape) for o in s.get_outputs()])
EOF
      then
        ok "${f} loads in onnxruntime"
      else
        fail "${f} failed to load (corrupt or wrong opset?)"
      fi
    fi
  done
else
  warn "onnxruntime not importable in current shell — skipping load test"
  warn "  hint: 'conda activate ${ENV_NAME}' before running verify_port.sh"
fi

# -----------------------------------------------------------------------------
section "6. catkin build artifacts (optional)"
# -----------------------------------------------------------------------------

WRAPPER="${CATKIN_WS}/devel/lib/sru_nav_go2_ros1/sru_nav_node"
if [ -f "$WRAPPER" ]; then
  ok "catkin wrapper exists: ${WRAPPER}"
  shebang=$(head -n1 "$WRAPPER")
  if echo "$shebang" | grep -qE "miniconda|anaconda|envs/${ENV_NAME}"; then
    ok "wrapper shebang points at conda python"
    info "  shebang: ${shebang}"
  else
    fail "wrapper shebang is NOT conda python: ${shebang}"
    info "  fix: 'conda activate ${ENV_NAME} && cd ${CATKIN_WS} && \\"
    info "        catkin_make clean && catkin_make -DPYTHON_EXECUTABLE=\$(which python3)'"
  fi
else
  warn "no catkin build artifact — run catkin_make in conda env first"
fi

# -----------------------------------------------------------------------------
section "7. Live roslaunch smoke test"
# -----------------------------------------------------------------------------

if [ "$SKIP_ROSLAUNCH" -eq 1 ]; then
  info "skipped (--skip-roslaunch)"
elif [ ! -f "${CATKIN_WS}/devel/setup.bash" ]; then
  warn "catkin workspace not built — skipping roslaunch test"
else
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash" || true
  source "${CATKIN_WS}/devel/setup.bash"     || true

  if ! command -v roslaunch >/dev/null; then
    fail "roslaunch not on PATH after sourcing"
  else
    LOG=$(mktemp)
    timeout 12 roslaunch sru_nav_go2_ros1 sru_nav_go2.launch \
        launch_joy:=false launch_static_tf:=false \
        require_joystick:=false >"$LOG" 2>&1 &
    RPID=$!
    sleep 8
    if grep -q "Navigation policy node is ready" "$LOG"; then
      ok "node became ready within 8 s"
    else
      fail "node did not reach 'ready' state within 8 s"
      info "  last 20 log lines:"
      tail -n 20 "$LOG" | sed 's/^/    /'
    fi
    kill "$RPID" 2>/dev/null
    wait "$RPID" 2>/dev/null
    rm -f "$LOG"
  fi
fi

# -----------------------------------------------------------------------------
section "Summary"
# -----------------------------------------------------------------------------

echo "  PASS=${PASS}  FAIL=${FAIL}  WARN=${WARN}"
if [ "$FAIL" -gt 0 ]; then
  echo
  c_red "RESULT: FAIL"; echo
  exit 1
elif [ "$WARN" -gt 0 ]; then
  echo
  c_yellow "RESULT: PASS WITH WARNINGS"; echo
  exit 0
else
  echo
  c_green "ALL CHECKS PASSED"; echo
  exit 0
fi
