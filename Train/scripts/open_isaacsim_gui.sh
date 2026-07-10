#!/usr/bin/env bash
# Launch the Isaac Sim full GUI (no task, no policy) in the project's docker.
# Use this to manually inspect USDs (e.g. drag the Go2 USD into the stage,
# expand link tree, view collision meshes).
#
# Usage:
#   ./scripts/open_isaacsim_gui.sh
#
# To open the Go2 USD inside the GUI:
#   1. File -> Open  ->  paste the URL printed below into the file picker
#   2. Or drag it from the Content browser (left panel):
#        Robots/Unitree/Go2/go2.usd
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

xhost +local:root >/dev/null 2>&1 || xhost +local:docker >/dev/null 2>&1 || true

cat <<EOF
[host] DISPLAY=${DISPLAY:-:1}
[host] Launching Isaac Sim GUI from sru-nav:latest ...

To inspect the Go2 USD once GUI is up, look under the Content browser:
    omniverse://localhost/NVIDIA/Assets/Isaac/<ver>/Isaac/Robots/Unitree/Go2/go2.usd
or via the cloud asset path used by IsaacLab:
    \$ISAACLAB_NUCLEUS_DIR/Robots/Unitree/Go2/go2.usd

EOF

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DISPLAY="${DISPLAY:-:1}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --shm-size=16g --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOME/.Xauthority:/root/.Xauthority:ro" \
  -v "$REPO_ROOT/outputs/logs:/workspace/IsaacLab/logs" \
  -v "$REPO_ROOT/mount/sru-navigation-sim:/workspace/IsaacLab/source/isaaclab_nav_task" \
  --entrypoint bash sru-nav:latest -lc \
  "/isaac-sim/isaac-sim.sh"
