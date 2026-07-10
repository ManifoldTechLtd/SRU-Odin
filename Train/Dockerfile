# =====================================================================
# SRU Navigation Sim - Reproducible Training Image
# Base: NVIDIA Isaac Sim 4.5.0 (matches IsaacLab v2.1.1 requirement)
# =====================================================================
ARG ISAAC_SIM_VERSION=4.5.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1

# ---------- System deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl wget vim ca-certificates build-essential \
        cmake ninja-build libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libglu1-mesa libglib2.0-0 ncurses-term tmux htop \
        python3-pip && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Avoid "dubious ownership" inside containers
RUN git config --global --add safe.directory '*'

# ---------- Configure pip mirror (optional) ----------
# Default behavior uses the official PyPI. The global pip config is written so
# ALL python interpreters in the image (root, kit-bundled
# `_isaac_sim/kit/python`, system python3) pick it up.
#
# Override pip index if needed: --build-arg PIP_INDEX_URL=https://your-mirror/simple --build-arg PIP_TRUSTED_HOST=your-mirror-host
ARG PIP_INDEX_URL=
ARG PIP_TRUSTED_HOST=
RUN if [ -n "${PIP_INDEX_URL}" ]; then \
        mkdir -p /etc && printf "[global]\nindex-url = %s\ntrusted-host = %s\ntimeout = 120\nretries = 5\n" \
            "${PIP_INDEX_URL}" "${PIP_TRUSTED_HOST}" > /etc/pip.conf; \
        mkdir -p /root/.config/pip && cp /etc/pip.conf /root/.config/pip/pip.conf; \
        echo "[build] pip index -> ${PIP_INDEX_URL}"; \
    fi

# Workspace
ARG ISAACLAB_VERSION=v2.1.1
ENV ISAACLAB_PATH=/workspace/IsaacLab
WORKDIR /workspace

# Optional build-time proxy (pass via: --build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=...)
# Useful when the host is behind GFW / corporate firewall and bare `git clone github.com`
# hits `GnuTLS recv error (-110)` / TLS resets. Leave empty to disable.
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY=
ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY}

# Optional GitHub mirror prefix (e.g. https://ghproxy.com/ or https://gh-proxy.com/).
# When set, all subsequent github.com clones are rewritten through it via git insteadOf.
ARG GITHUB_MIRROR=
RUN if [ -n "${GITHUB_MIRROR}" ]; then \
        git config --global url."${GITHUB_MIRROR}https://github.com/".insteadOf "https://github.com/"; \
        echo "[build] github.com clones rewritten via ${GITHUB_MIRROR}"; \
    fi

# Harden git for flaky networks (large postBuffer, low-speed timeout, no compression,
# disable smart-http v2 which is the layer that surfaces the GnuTLS reset on long clones).
RUN git config --global http.postBuffer 1048576000 && \
    git config --global http.lowSpeedLimit 1000 && \
    git config --global http.lowSpeedTime 600 && \
    git config --global core.compression 0 && \
    git config --global http.version HTTP/1.1

# ---------- Bring IsaacLab v2.1.1 into the image ----------
# We COPY from the local build context (./mount/IsaacLab) instead of `git clone`,
# because on GFW-affected servers `git clone github.com` repeatedly fails with
# GnuTLS resets / connection timeouts. To set this up on the host:
#
#     # one-shot: clone IsaacLab anywhere reachable from your build context
#     git clone --depth 1 --branch v2.1.1 \
#         https://github.com/isaac-sim/IsaacLab.git <repo>/mount/IsaacLab
#
# If mount/IsaacLab is missing the build will abort here with a clear error.
COPY mount/IsaacLab ${ISAACLAB_PATH}
RUN test -d ${ISAACLAB_PATH}/source/isaaclab && \
    echo "[build] using local IsaacLab from mount/IsaacLab"

# Symlink Isaac Sim into IsaacLab as required by isaaclab.sh
RUN ln -s /isaac-sim ${ISAACLAB_PATH}/_isaac_sim

# ---------- Install IsaacLab (uses bundled python from isaac-sim) ----------
WORKDIR ${ISAACLAB_PATH}
RUN ./isaaclab.sh --install

# Workaround: the official --install loop builds extensions in PEP-517 isolation,
# which fails for source/isaaclab because the kit-python build env lacks pkg_resources.
# Reinstall every source/* extension with --no-build-isolation to guarantee `import isaaclab` works.
RUN for d in ${ISAACLAB_PATH}/source/*/; do \
        if [ -f "$d/setup.py" ]; then \
            ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install --no-build-isolation --timeout 180 --retries 5 -e "$d" || exit 1; \
        fi; \
    done && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -c "import isaaclab; print('isaaclab OK:', isaaclab.__file__)"

# ---------- Replace bundled rsl_rl with SRU-enhanced fork ----------
# sru-navigation-learning provides ActorCriticSRU + MDPO/SPO/PPO required by sru-navigation-sim.
# It installs under the package name `rsl_rl`, replacing any prior install.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall rsl-rl-lib -y || true
RUN rm -rf ${ISAACLAB_PATH}/_isaac_sim/kit/python/lib/python3.10/site-packages/rsl_rl
WORKDIR /workspace
RUN for i in 1 2 3 4 5; do \
        git clone --depth 1 https://github.com/leggedrobotics/sru-navigation-learning.git /workspace/rsl_rl && break; \
        echo "[retry $i] sru-navigation-learning clone failed, retrying in 10s..."; \
        rm -rf /workspace/rsl_rl; \
        sleep 10; \
    done && test -d /workspace/rsl_rl/.git
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-build-isolation --timeout 180 --retries 5 -e /workspace/rsl_rl
RUN ${ISAACLAB_PATH}/_isaac_sim/python.sh -c "from rsl_rl.modules import ActorCriticSRU; print('SRU OK')"

# ---------- Clone & install sru-navigation-sim extension ----------
ARG SRU_NAV_REPO=https://github.com/leggedrobotics/sru-navigation-sim.git
ARG SRU_NAV_REF=main
# Note: git network hardening already applied above (postBuffer / lowSpeedTime /
# HTTP/1.1 / optional GITHUB_MIRROR), so this block only retries.
RUN for i in 1 2 3 4 5; do \
        git clone --depth 1 --branch ${SRU_NAV_REF} \
            ${SRU_NAV_REPO} ${ISAACLAB_PATH}/source/isaaclab_nav_task && break; \
        echo "[retry $i] sru-navigation-sim clone failed, cleaning up and retrying in 10s..."; \
        rm -rf ${ISAACLAB_PATH}/source/isaaclab_nav_task; \
        sleep 10; \
    done && \
    test -d ${ISAACLAB_PATH}/source/isaaclab_nav_task
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e \
        ${ISAACLAB_PATH}/source/isaaclab_nav_task

# Useful Python packages for logging/visualization
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install \
        wandb tensorboard tensorboardX

# Common runtime envs
ENV PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all

WORKDIR ${ISAACLAB_PATH}

# Sanity check (non-fatal): list registered tasks
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip show isaaclab_nav_task || true

CMD ["/bin/bash"]
