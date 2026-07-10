#!/usr/bin/env bash
# Run probe_pt.py inside the sru-nav container against the Go2 locomotion .pt.
# Usage:
#   ./scripts/probe_pt.sh                       # uses default path under mount/
#   ./scripts/probe_pt.sh /custom/path/in/host  # absolute host path
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_PT="${1:-$REPO_ROOT/mount/sru-navigation-sim/isaaclab_nav_task/navigation/assets/data/Policies/locomotion/go2/policy_go2.pt}"

if [[ ! -f "$HOST_PT" ]]; then
    echo "ERROR: .pt not found at: $HOST_PT"
    echo "       Copy your Go2 locomotion .pt there first."
    exit 1
fi

echo "[host] probing: $HOST_PT"
echo "[host] file size: $(du -h "$HOST_PT" | cut -f1)"

# Translate the host path to the corresponding container path (the mount).
HOST_NAV_SIM="$REPO_ROOT/mount/sru-navigation-sim"
CONTAINER_NAV_SIM="/workspace/IsaacLab/source/isaaclab_nav_task"
if [[ "$HOST_PT" == "$HOST_NAV_SIM"/* ]]; then
    CONTAINER_PT="${HOST_PT/$HOST_NAV_SIM/$CONTAINER_NAV_SIM}"
else
    echo "ERROR: .pt must live under $HOST_NAV_SIM (the live-mounted folder)."
    echo "       Got: $HOST_PT"
    exit 1
fi
echo "[host] container path: $CONTAINER_PT"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  --shm-size=4g --ipc host \
  -v "$REPO_ROOT/scripts/probe_pt.py:/tmp/probe_pt.py:ro" \
  -v "$REPO_ROOT/mount/sru-navigation-sim:/workspace/IsaacLab/source/isaaclab_nav_task" \
  --entrypoint bash sru-nav:latest -lc \
  "/workspace/IsaacLab/_isaac_sim/python.sh /tmp/probe_pt.py '$CONTAINER_PT'"
