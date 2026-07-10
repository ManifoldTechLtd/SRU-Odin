#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Conda environment bootstrap for sru_nav_go2_ros1 on Go2 NX (Jetson Xavier NX)
# -----------------------------------------------------------------------------
# Goal: create a python env containing onnxruntime + cv2 + scipy + numpy that
# works alongside system ROS Noetic (which provides rospy / cv_bridge).
#
# We do NOT install rospy / cv_bridge into conda. They come from /opt/ros/noetic
# (system python3.8). To make them importable from the conda env we expose the
# ROS site-packages via PYTHONPATH at run time (see launch_sru_nav.sh).
#
# Usage:
#   bash setup_conda_env.sh                      # create env and install deps
#   bash setup_conda_env.sh --gpu                # try to install Jetson onnxruntime-gpu wheel
#   bash setup_conda_env.sh --check              # just run sanity checks
# -----------------------------------------------------------------------------

set -eu

ENV_NAME="${ENV_NAME:-sru_nav}"
PY_VER="${PY_VER:-3.8}"      # match Noetic's system python so rospy works
USE_GPU=0
CHECK_ONLY=0
SKIP_TIME_CHECK=0

# Silence anaconda.org's noisy "aau_token_host" telemetry warnings.
export ANACONDA_ANON_USAGE=false

for arg in "$@"; do
  case "$arg" in
    --gpu)             USE_GPU=1 ;;
    --check)           CHECK_ONLY=1 ;;
    --skip-time-check) SKIP_TIME_CHECK=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------------
# 0. System clock sanity. SSL fails with "certificate is not yet valid" when
#    the Jetson clock is rolled back (no RTC battery). Detect early.
# -----------------------------------------------------------------------------
if [ "$SKIP_TIME_CHECK" -eq 0 ]; then
  NOW_EPOCH=$(date +%s)
  # 2025-01-01 UTC = 1735689600. Treat anything before that as suspicious.
  if [ "$NOW_EPOCH" -lt 1735689600 ]; then
    cat >&2 <<EOF
[ERROR] System clock looks wrong:  $(date)
        This will break HTTPS/pip with "certificate is not yet valid".
        Fix it before continuing:

          sudo timedatectl set-ntp true
          sudo systemctl restart systemd-timesyncd
          # or, offline:
          sudo date -s "$(date -u +'%Y-%m-%d %H:%M:%S' -d '@1748400000')"
          sudo hwclock --systohc

        Then re-run this script. To override (NOT recommended) pass:
          bash $0 --skip-time-check
EOF
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# 1. Locate conda
# -----------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Install miniforge (recommended for aarch64):"
  echo "  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
  echo "  bash Miniforge3-Linux-aarch64.sh -b -p \$HOME/miniforge3"
  echo "  source \$HOME/miniforge3/etc/profile.d/conda.sh"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# -----------------------------------------------------------------------------
# 2. Create env if missing
# -----------------------------------------------------------------------------
if [ "$CHECK_ONLY" -eq 0 ]; then
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] Conda env '${ENV_NAME}' already exists. Skipping creation."
  else
    echo "[INFO] Creating conda env '${ENV_NAME}' with python=${PY_VER}"
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}" pip
  fi
fi

conda activate "${ENV_NAME}"

if [ "$CHECK_ONLY" -eq 0 ]; then
  echo "[INFO] Installing Python dependencies into '${ENV_NAME}'"
  # Optional: use a faster mirror (Tsinghua) if PIP_INDEX_URL is not already set.
  : "${PIP_INDEX_URL:=}"
  export PIP_INDEX_URL
  echo "[INFO] PIP_INDEX_URL=${PIP_INDEX_URL}"

  python -m pip install --upgrade pip
  # Pin versions that are known to work on python 3.8 + ARM.
  # netifaces / defusedxml are needed by rospy at runtime (the system ROS
  # python gets them via apt 'python3-netifaces / python3-defusedxml', but
  # the conda python does not).
  python -m pip install \
      "numpy<2.0" \
      "scipy>=1.7,<1.11" \
      "opencv-python>=4.5,<5.0" \
      "rospkg" \
      "pyyaml" \
      "empy==3.3.4" \
      "netifaces" \
      "defusedxml"

  if [ "$USE_GPU" -eq 1 ]; then
    cat <<'EOF'
[INFO] --gpu requested. Standard pip onnxruntime-gpu does NOT work on Jetson.
       Download the JetPack-matched wheel from NVIDIA's Jetson Zoo:
           https://elinux.org/Jetson_Zoo#ONNX_Runtime
       Then install with e.g.:
           pip install ./onnxruntime_gpu-<ver>-cp38-cp38-linux_aarch64.whl
       Skipping automatic GPU wheel install.
EOF
    python -m pip install "onnxruntime>=1.15,<1.18"     # CPU fallback so node still runs
  else
    python -m pip install "onnxruntime>=1.15,<1.18"
  fi
fi

# -----------------------------------------------------------------------------
# 3. Sanity checks
# -----------------------------------------------------------------------------
echo
echo "============================================================"
echo "  Sanity check: conda env '${ENV_NAME}'"
echo "============================================================"
python - <<'PYEOF'
import sys, platform
print(f"python      : {sys.version.split()[0]}  ({platform.machine()})")
fail = []

def check(name, importer):
    try:
        mod = importer()
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK] {name:18s} {ver}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        fail.append(name)

check("numpy",        lambda: __import__("numpy"))
check("scipy",        lambda: __import__("scipy"))
check("cv2",          lambda: __import__("cv2"))
check("rospkg",       lambda: __import__("rospkg"))
check("netifaces",    lambda: __import__("netifaces"))
check("defusedxml",   lambda: __import__("defusedxml"))
check("onnxruntime",  lambda: __import__("onnxruntime"))

try:
    import onnxruntime as ort
    print(f"  ORT providers   : {ort.get_available_providers()}")
except Exception:
    pass

# rospy / cv_bridge come from /opt/ros/noetic; only succeed when PYTHONPATH is set.
import os
ros_path = "/opt/ros/noetic/lib/python3/dist-packages"
if os.path.isdir(ros_path) and ros_path not in sys.path:
    sys.path.insert(0, ros_path)

check("rospy",        lambda: __import__("rospy"))
check("cv_bridge",    lambda: __import__("cv_bridge"))

if fail:
    print(f"\n[WARN] Missing modules: {fail}")
    sys.exit(1)
print("\n[OK] All imports passed.")
PYEOF

echo
echo "[INFO] Try loading the policy ONNX files (if present)..."
# Resolve the package's models/ directory relative to THIS script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_MODELS_DIR="$(cd "${SCRIPT_DIR}/../models" 2>/dev/null && pwd || echo '')"
export PKG_MODELS_DIR
python - <<'PYEOF'
import os, sys
candidates = [
    os.environ.get("PKG_MODELS_DIR", ""),
    os.path.expanduser("~/sru_ws/src/sru_nav_go2_ros1/models"),
]
candidates = [c for c in candidates if c]
import onnxruntime as ort
for d in candidates:
    vae = os.path.join(d, "vae_encoder.onnx")
    pol = os.path.join(d, "nav_policy.onnx")
    if os.path.exists(vae) and os.path.exists(pol):
        print(f"  Found models in {d}")
        try:
            s1 = ort.InferenceSession(vae, providers=["CPUExecutionProvider"])
            s2 = ort.InferenceSession(pol, providers=["CPUExecutionProvider"])
            print(f"    vae inputs  : {[i.name+str(i.shape) for i in s1.get_inputs()]}")
            print(f"    vae outputs : {[o.name+str(o.shape) for o in s1.get_outputs()]}")
            print(f"    pol inputs  : {[i.name+str(i.shape) for i in s2.get_inputs()]}")
            print(f"    pol outputs : {[o.name+str(o.shape) for o in s2.get_outputs()]}")
        except Exception as e:
            print(f"    [FAIL] {e}")
        break
else:
    print("  (No models found; skipping ORT smoke test.)")
PYEOF

echo
echo "[DONE] Activate with:  conda activate ${ENV_NAME}"
