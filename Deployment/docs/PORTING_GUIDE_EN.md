# Porting the SRU navigation policy to Odin1 + ROS1: full reproduction guide

> Starting from the paper *Spatially-Enhanced Recurrent Memory for Long-Range
> Mapless Navigation via End-to-End Reinforcement Learning* and the
> open-source repositories indexed at
> [`sru-project-website`](https://michaelfyang.github.io/sru-project-website/),
> this guide explains how to deploy the SRU end-to-end navigation policy on
> **the Odin1 depth camera + ROS1 Noetic + any robot that consumes
> `geometry_msgs/Twist` on `/cmd_vel`** (the reference platform is the
> Unitree Go2). The package `sru_nav_go2_ros1` in this repository is the
> output of that effort.
>
> **中文版**: see `PORTING_GUIDE.md`.

---

## 0. What this document is — and is not

**Is**: a three-in-one guide combining (a) the porting decision log, (b) a
ready-to-use AI prompt that lets a coding agent regenerate this package
from upstream, and (c) an automated acceptance script. After reading you
should be able to:

1. understand which upstream pieces were changed and why,
2. run a slash command in Cascade / Claude Code to recreate the package
   from the open-source repos and the paper,
3. run a single script to verify the result.

**Is not**: an algorithm tutorial. For the model and reward design, read
the paper and the upstream code directly.

**Audience**:
- comfortable with Ubuntu + ROS1 + Python/conda,
- has a Unitree Go2 (or any robot subscribing to `/cmd_vel`) plus an
  Odin1 depth camera,
- wants to run SRU on their own robot, or to verify whether an AI agent
  can reproduce a real-robot deployment package from the paper + open
  code.

> Zero-barrier promise: every path, IP, robot model, and camera model is
> overridable via environment variable or ROS parameter. No source code
> edits required to retarget.

---

## 1. Upstream sources

| Asset | Link / path | Role in this port |
|---|---|---|
| Paper | *Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning* | Defines the observation/action space, rewards, and network (depth VAE + LSTM-SRU + actor head) |
| Project page | <https://michaelfyang.github.io/sru-project-website/> | Index of repos, videos, paper |
| `sru-navigation-learning` | RL training code (rsl_rl-style); contains PPO/MDPO algorithms, reward terms, observation manager | Source of truth for **reward design, action squashing, observation order**, plus the ONNX export entry point |
| `sru-navigation-sim` | IsaacLab task with B2W / AOW training configs | Provides `policy_scaling`, randomization ranges, observation shapes (depth `64×5×8`) |
| `sru-robot-deployment` | Original B2W + ZED-X real-robot deployment by the authors | The **direct rewrite target**: original uses PyTorch + DLIO odom + a custom SDK bridge; this port uses ONNX + Odin1 odom + a Go2 bridge |
| `sru-pytorch-spatial-learning` | PyTorch implementation of the SRU / LSTM-SRU cell | Explains RNN hidden-state shape (`rnn_hidden_size=512, num_layers=1`) and how `(h, c)` are exposed when exporting to ONNX |
| `sru-depth-pretraining` | Pretraining of the depth VAE encoder | Provides the input shape (`H×W` normalized depth, single channel) and latent dim of `vae_encoder.onnx` |

If you only want to run, not reproduce: drop the two trained ONNX files
(`vae_encoder.onnx` + `nav_policy.onnx`) into `sru_nav_go2_ros1/models/`
and skip to §6.

---

## 2. Porting target matrix

| Aspect | Author deployment (`sru-robot-deployment`) | This package (`sru_nav_go2_ros1`) | Abstracted interface |
|---|---|---|---|
| Robot | Unitree **B2W** (wheeled-leg) | Unitree **Go2** (pure-leg) | Any robot consuming `/cmd_vel` (`geometry_msgs/Twist`, body frame) |
| Depth sensor | ZED-X | **Odin1** (`32FC1`, ~10 Hz) | `~depth_topic` (`sensor_msgs/Image`, 32FC1, meters) |
| Odometry | DLIO (custom LIO) | Odin1 `odometry_highfreq` | `~odom_topic` (`nav_msgs/Odometry`, world frame) |
| Inference backend | PyTorch JIT | **ONNX Runtime** | `models/{vae_encoder,nav_policy}.onnx` |
| Compute platform | x86 industrial PC | **Jetson Orin NX (aarch64) + conda py3.8** | Any Linux + ROS1 Noetic + conda |
| Control rate | ~10 Hz | **5 Hz** (matches training; Odin depth ~10 Hz, naturally decimated) | `~control_frequency` |
| Operator joystick | PS5 PerceptiveNavigationSE2 | Same PS5 mapping, **plus `require_joystick: false`** for headless | `--no-deadman` / yaml `require_joystick` |
| Install / launch | rosinstall full stack | **`setup_conda_env.sh` + `launch_sru_nav.sh`** | env vars `ENV_NAME / ROS_DISTRO / CATKIN_WS`, all overridable |

---

## 3. Core porting decisions (motivation + reference)

### 3.1 Inference: PyTorch → ONNX Runtime

- **Why**: stacking PyTorch + cv_bridge + onnxruntime on a Jetson Orin NX
  rapidly snowballs into a dependency mess; onnxruntime ships a single
  shared-object loader, swaps CPU/GPU EP cleanly, and has prebuilt ARM
  wheels.
- **Interface**: `@/sru_nav_go2_ros1/src/sru_nav_go2/model.py` implements
  `LearningModel`, which loads `vae_encoder.onnx` (depth → latent) and
  `nav_policy.onnx` (latent + state vector + last action + LSTM hidden →
  action mean + new hidden) as two separate sessions.
- **Hidden-state contract**: training uses `rnn_hidden_size=512,
  num_layers=1`; the ONNX export carries `(h, c)` as input/output tensors
  and the deployment side holds them between frames. `reset_hidden_state`
  matches the training `is_first` flag.
- **Action squash**: the network already includes `tanh`, so the output is
  in `(-1, 1)`. The deployment then multiplies by
  `policy_scale = [vx_max, vy_max, ω_max]` (default `[0.6, 0.3, 0.6]`)
  to obtain SI velocities.

### 3.2 Observations: Odin1 replaces ZED-X + DLIO

- **Depth**: Odin1 publishes `sensor_msgs/Image` (`32FC1`, meters). The
  callback `@/sru_nav_go2_ros1/src/sru_nav_go2/navigation_policy_node.py:209-225`:
  1. `cv_bridge.imgmsg_to_cv2(passthrough)`,
  2. `nan_to_num(nan=0, posinf=2*max_depth, neginf=0)`,
  3. clamps to `[min_depth, max_depth] = [0.25, 10.0] m`, out-of-range
     pixels set to 0.
- **Odometry frame convention**: Odin1's `twist` is **world frame**. The
  `odom_callback` rotates it back to the body frame using the body
  quaternion (matching training). If your odom is already in body frame,
  set `use_sim: true` and the node passes `twist` straight through
  (`@/sru_nav_go2_ros1/src/sru_nav_go2/navigation_policy_node.py:189-197`).
- **Goal frame contract**: `/goal_pose.header.frame_id` must equal the odom
  `frame_id` (default `odom`). Otherwise the node rejects the message —
  this guards against operators sending a body-frame coordinate as if it
  were odom-frame.

### 3.3 Action egress: straight to `/cmd_vel`

- The original B2W deployment ships its own SDK bridge; on Go2 a tiny
  separate node (subscribes to `/cmd_vel`, talks to `unitree_legged_sdk`
  sport mode) is enough. **This package is robot-agnostic** and does not
  bind to a specific bridge.
- `policy_scale = [0.6, 0.3, 0.6]` is a deliberate halving versus the
  training value `[1.5, 1.0, 1.0]`, leaving margin for real-robot bring-up.
  Training already randomized scale by `Uniform(0.6, 1.2)`, so the network
  is robust to runtime scaling.

### 3.4 Safety: dual-mode joystick deadman

| Mode | Trigger | Behavior |
|---|---|---|
| Default (`require_joystick=true`) | Real-robot deployment | `cmd_vel_ratio=0` until `/joy` is published with `axes[4]` > 0; >15 s without `/joy` forces stop |
| Test (`require_joystick=false`) | `--no-deadman` or yaml=false | `cmd_vel_ratio=1.0` from start, no `/joy` required. **Lifted-wheel / sim / open-field only** |

Source: `@/sru_nav_go2_ros1/src/sru_nav_go2/navigation_policy_node.py:118-121, 272-287`.

### 3.5 Install layer: conda + system ROS coexistence

The Jetson Orin NX ships ROS Noetic (system python 3.8 + cv_bridge.so).
At the same time we want `onnxruntime / opencv / numpy / …` only inside a
conda env. The two worlds must interoperate:

1. **`PYTHONPATH` injection**: prepend
   `/opt/ros/noetic/lib/python3/dist-packages` so the conda python can
   import `rospy / cv_bridge / sensor_msgs / tf2_ros`.
2. **Shebang lock**: `catkin_make` must run with
   `-DPYTHON_EXECUTABLE=$(which python3)` **while the conda env is
   active**, otherwise `devel/lib/.../sru_nav_node`'s `#!` line will point
   at `/usr/bin/python3` and `import onnxruntime` fails at runtime.
3. **ABI conflict fix**: the conda env ships `libffi.so.7`, but the system
   `libp11-kit.so.0` was linked against the system `libffi`'s versioned
   symbols. Without intervention `cv_bridge` crashes with
   `undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0`.
   `launch_sru_nav.sh` automatically `LD_PRELOAD`s the system
   `libffi.so.7` so its symbols take priority.
4. **Hidden rospy deps**: `netifaces` and `defusedxml` are provided to the
   system python via apt packages; the conda env needs them via pip.
   `setup_conda_env.sh` already lists both.
5. **Clock check**: Jetson without an RTC battery often boots with a stale
   clock, breaking pip TLS. The script runs a `date` plausibility check
   and points to `ntpdate / timedatectl`.

Full automation: `@/sru_nav_go2_ros1/scripts/setup_conda_env.sh` +
`@/sru_nav_go2_ros1/scripts/launch_sru_nav.sh`.

### 3.6 Deliberate omissions

| Present in training | Reason for omitting at deployment |
|---|---|
| Heightmap / heightscan critic | Go2 has no LiDAR/heightmap; can be added as an external topic if needed |
| Critic-only observation channels | Critic exists only at training time; only the actor head runs at inference |
| MDPO / mutual-distillation dual actor-critic | Optimizer-side mechanism; irrelevant to the exported actor |
| Reward computation | Inference does not need rewards |
| Action-scale randomization (`Uniform(0.6, 1.2)`) | Deployment uses a fixed scale; the network has been trained to be robust to this range |

---

## 4. File map and responsibilities

```
sru_nav_go2_ros1/
├── CMakeLists.txt                 # catkin def: catkin_python_setup() + install scripts
├── package.xml                    # deps: rospy, sensor_msgs, geometry_msgs, nav_msgs, cv_bridge, tf2_ros
├── setup.py                       # installs src/sru_nav_go2 as a Python package
├── README.md
├── config/
│   ├── sru_nav.yaml               # all runtime ROS params; the only file users normally touch
│   └── waypoints_example.yaml     # multi-goal tour example (odom-frame point list)
├── launch/
│   └── sru_nav_go2.launch         # joy_node + static_tf + sru_nav_node
├── models/
│   ├── vae_encoder.onnx           # exported from training side via export_onnx
│   └── nav_policy.onnx
├── scripts/
│   ├── setup_conda_env.sh         # creates the sru_nav env, installs onnxruntime / cv2 / netifaces …
│   ├── launch_sru_nav.sh          # one-liner launcher (with LD_PRELOAD / PYTHONPATH fixes)
│   ├── sru_nav_node               # ROS node entrypoint (catkin install copies it under devel/lib/…)
│   ├── waypoint_runner.py         # publishes /goal_pose in sequence
│   └── verify_port.sh             # acceptance script (see §7)
├── src/sru_nav_go2/
│   ├── constants.py               # all training-aligned constants
│   ├── model.py                   # ONNX Runtime wrapper: VAE encoder + LSTM-SRU policy
│   ├── navigation_policy_node.py  # main node: odom/depth/joy/goal callbacks + cmd_vel out
│   ├── utils.py                   # quaternion ↔ rotation, projected gravity, etc.
│   ├── visualization.py           # rviz markers
│   └── waypoint_manager.py        # record / replay waypoint list
└── docs/
    ├── DEPLOY_GO2_NX.md           # NX deployment field notes
    ├── PORTING_GUIDE.md           # Chinese version
    └── PORTING_GUIDE_EN.md        # ← this file
```

---

## 5. Reproducing this package with an AI agent

We ship **the same porting prompt in three call forms** (use whichever
fits your tool):

| Entry | Tool | How to invoke |
|---|---|---|
| `.windsurf/workflows/port-sru-to-ros.md` | Cascade (Windsurf IDE) | type `/port-sru-to-ros` in the Cascade panel |
| `.claude/commands/port-sru-to-ros.md` | Claude Code (CLI / IDE) | start `claude`, then `/port-sru-to-ros` |
| `docs/PORTING_GUIDE.md` §5.2 | Any agent that accepts a custom prompt (Cursor, Continue, Aider, …) | paste §5.2 into the system / task prompt |

All three derive from one canonical prompt. Differences are limited to
the launch header and per-tool path conventions.

> ⚠️ **Reproducibility caveat**: a single LLM round generating ~1500 LoC
> of ROS porting code is not 100% reliable. We split the task into a
> 6-step workflow so each step has a grep-able acceptance check; this
> dramatically beats one-shot mega-prompts in success rate.

### 5.1 Workflow at a glance

| Step | Inputs | Outputs | Acceptance |
|---|---|---|---|
| 1. **Recon** | 5 upstream repos + paper | `notes/upstream_recon.md` summarizing arch, obs/action, IO shapes, training hyperparams | file exists and lists net depth, obs dims, policy_scale |
| 2. **Extract inference** | `sru-navigation-learning` export entry | `models/vae_encoder.onnx`, `models/nav_policy.onnx`, `docs/IO_SPEC.md` | onnxruntime can load both; IO names recorded |
| 3. **catkin skeleton** | Step 2 IO spec | `package.xml / CMakeLists.txt / setup.py / launch/ / config/sru_nav.yaml` | `catkin_make` succeeds |
| 4. **Port the node** | Original `sru-robot-deployment` node + IO spec | `src/sru_nav_go2/{constants,model,utils,visualization,waypoint_manager,navigation_policy_node}.py` + `scripts/sru_nav_node` | `roslaunch` brings the node up cleanly |
| 5. **Deployment scripts** | Target platform info | `scripts/setup_conda_env.sh` + `scripts/launch_sru_nav.sh` | clean-env smoke-test in §6 passes |
| 6. **Docs + verify** | All above | `README.md / docs/DEPLOY_*.md / scripts/verify_port.sh` | `bash scripts/verify_port.sh` all green |

### 5.2 Canonical prompt (the body of all three entry files)

```text
You are a senior engineer fluent in ROS1 Noetic, conda, ONNX Runtime, and
PyTorch.

[Task]
From the upstream sources below, generate a catkin package named
`sru_nav_go2_ros1` that deploys the navigation policy from the paper
"Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation
via End-to-End Reinforcement Learning" onto an Odin1 depth camera +
ROS1 Noetic + any robot consuming `geometry_msgs/Twist` on `/cmd_vel`.

[Upstream sources]
- Paper PDF (path provided by user, e.g. ./2506.05997v2.pdf)
- Project page: https://michaelfyang.github.io/sru-project-website/
- Repos under user-provided `${UPSTREAM_DIR}`:
    sru-navigation-learning      — RL training + ONNX export
    sru-navigation-sim           — IsaacLab env + training configs
    sru-robot-deployment         — original B2W+ZED-X deployment (the
                                     direct rewrite target)
    sru-pytorch-spatial-learning — SRU / LSTM-SRU cell
    sru-depth-pretraining        — depth VAE pretraining
- Pre-existing onnx files (if any):
    models/{vae_encoder,nav_policy}.onnx

[Hard constraints]
1. Sensors are fixed to Odin1: depth `/odin1/depth_img_competetion`
   (sensor_msgs/Image, 32FC1, meters), odom
   `/odin1/odometry_highfreq` (nav_msgs/Odometry, world frame).
2. Output is fixed to `/cmd_vel` (geometry_msgs/Twist, body frame). The
   bridge to a specific robot's SDK is out of scope.
3. Only `onnxruntime` for inference. No PyTorch at runtime.
4. Must coexist with system ROS python in a conda env (default
   `sru_nav`, py3.8):
   - inject /opt/ros/<DISTRO>/lib/python3/dist-packages into PYTHONPATH
   - LD_PRELOAD system libffi to fix cv_bridge ABI conflict
   - lock catkin_make's PYTHON_EXECUTABLE to conda python
5. Every path / env / distro / robot model must be overridable via env
   vars or ROS params; never write `/home/<user>` into source files.
6. Preserve all training-aligned numbers: control_frequency=5 Hz,
   rnn_hidden=512, default policy_scale=[0.6,0.3,0.6], joystick axis
   mapping, etc.
7. Safety: joystick deadman + 15 s timeout, with a
   `require_joystick: false` bypass; default must be true.

[Output file list] (all required)
package.xml, CMakeLists.txt, setup.py
launch/sru_nav_go2.launch
config/sru_nav.yaml, config/waypoints_example.yaml
scripts/setup_conda_env.sh, scripts/launch_sru_nav.sh,
scripts/sru_nav_node, scripts/waypoint_runner.py, scripts/verify_port.sh
src/sru_nav_go2/{__init__.py, constants.py, model.py, utils.py,
                 visualization.py, waypoint_manager.py,
                 navigation_policy_node.py}
docs/DEPLOY.md, docs/PORTING_GUIDE.md (Chinese), docs/PORTING_GUIDE_EN.md
models/README.md (how to obtain / export the two ONNX files)

[Workflow] (must be sequential; self-check after each step)
Step 1 — Recon: read repos and paper; emit notes/upstream_recon.md with
        net architecture, obs dims, action space, rsl_rl algorithm config,
        reward list.
Step 2 — IO normalization: locate export_onnx in the training repo, log
        input/output names + shapes, emit docs/IO_SPEC.md.
Step 3 — Generate catkin skeleton; verify with `catkin_make`.
Step 4 — Port the node: use sru-robot-deployment as template, swap
        sensors, drop non-Odin deps, replace PyTorch calls with
        onnxruntime.
Step 5 — Write the two scripts (setup + launcher), covering all of
        constraint #4.
Step 6 — Generate verify_port.sh: package layout, shebang, ONNX load
        test, rostopic list, yaml fields.

[Acceptance] (the user will run verify_port.sh and the commands below)
1. `catkin_make` in a clean workspace: 0 warnings, 0 errors.
2. `bash scripts/setup_conda_env.sh --check` passes.
3. `head -1 devel/lib/sru_nav_go2_ros1/sru_nav_node` points at conda
   python, not `/usr/bin/python3`.
4. `roslaunch sru_nav_go2_ros1 sru_nav_go2.launch require_joystick:=false`
   prints `Navigation policy node is ready.` and emits no error spam.
5. Synthetic odom + a 32FC1 depth frame + a goal_pose results in a
   non-zero `/cmd_vel`.
6. `bash scripts/verify_port.sh` is all green.

[Response discipline]
- Do not emit a single 1000+ line reply; chunk per step, summarize the
  plan first, then write files.
- After each step run the listed self-check, paste the command and
  output, fix on failure before advancing.
- If an upstream file is missing, halt and ask. Never fabricate APIs.
```

The fully executable variants are in
`@/.windsurf/workflows/port-sru-to-ros.md` and
`@/.claude/commands/port-sru-to-ros.md`.

### 5.3 How do I prove that this prompt actually reproduces the package?

Run the workflow on a clean machine, then:

```bash
diff -r --exclude=__pycache__ --exclude=.git \
     ./generated_sru_nav_go2_ros1/ \
     ./sru_nav_go2_ros1/
```

Expected: **identical structure; line-level diffs concentrated in
comments and literal ordering, not in numeric constants, topic names,
or function signatures**. If you see drift on critical items
(`policy_scale` becoming `[1, 1, 1]`, the LD_PRELOAD line missing, etc.),
the agent did not follow the prompt — return to that step and retry.

---

## 6. Personalization and zero-barrier usage

Every "user-specific" knob is exposed as a parameter. The three most
common categories:

### 6.1 Paths

| Parameter | Default | Override |
|---|---|---|
| catkin workspace | `$HOME/code/odin_sru_nav` | `CATKIN_WS=/path bash scripts/launch_sru_nav.sh` |
| conda env name | `sru_nav` | `ENV_NAME=my_env bash scripts/setup_conda_env.sh` |
| ROS distro | `noetic` | `ROS_DISTRO=melodic bash scripts/launch_sru_nav.sh` (py3-compatible only) |
| pip mirror | Tsinghua | `PIP_INDEX_URL=https://pypi.org/simple bash scripts/setup_conda_env.sh` |

### 6.2 Topics and frames

Edit `config/sru_nav.yaml`; **do not edit source code**:

```yaml
depth_topic:    "/your/depth"      # sensor_msgs/Image (32FC1, meters)
odom_topic:     "/your/odometry"   # nav_msgs/Odometry, world frame
joy_topic:      "/joy"
goal_topic:     "/goal_pose"
cmd_vel_topic:  "/cmd_vel"
```

### 6.3 Different robots

You only need a `/cmd_vel` bridge. The package is **robot-agnostic**.
Common substitutions:

- Unitree Go2: ~50 LoC bridge wrapping `unitree_legged_sdk` sport mode
- Unitree B2 / B2W: official ROS1 bridge
- Boston Dynamics Spot: `spot_ros` + `cmd_vel`
- Any ROS sim: subscribes to `/cmd_vel` directly

The only TF you may need to retune is the static
`base_link → odin1_base_link` published by the launch file; the args
`odin1_x / y / z / roll / pitch / yaw` mirror your camera mounting pose.

### 6.4 System-environment check (also embedded in the prompt)

Run once on a fresh host:

```bash
# 1) OS / ROS
lsb_release -a                                # Ubuntu 20.04 recommended (Noetic)
echo $ROS_DISTRO                              # expect: noetic
which roscore                                 # /opt/ros/noetic/bin/roscore

# 2) Conda
conda --version                               # ≥ 4.10
conda env list | grep -E "sru_nav|base"

# 3) Clock (Jetsons without RTC battery often drift)
date                                          # must reflect real wall time

# 4) Network
ping -c 2 8.8.8.8 || echo "WARN: no internet, pip will fail"

# 5) Hardware
ls /dev/input/js* 2>/dev/null \
  || echo "INFO: no joystick — use --no-deadman for headless tests"
rostopic list 2>/dev/null | grep odin1 \
  || echo "WARN: Odin1 driver not running"
```

`scripts/verify_port.sh` codifies the same checks.

---

## 7. Acceptance script — `scripts/verify_port.sh`

Designed for the developer to **self-check immediately after generation**.
The full set lives in the script itself (with `[PASS]/[FAIL]/[WARN]`
markers); the most important checks:

| Check | Command | Expected |
|---|---|---|
| Package layout complete | look for `package.xml CMakeLists.txt setup.py launch config models scripts src/sru_nav_go2/{constants,model,navigation_policy_node}.py` | all present |
| Models load | `python -c "import onnxruntime as ort; ort.InferenceSession('models/vae_encoder.onnx')"` | no exception, prints input shape |
| catkin builds | `catkin_make -DPYTHON_EXECUTABLE=$(which python3)` | exit 0 |
| Shebang correct | `head -1 devel/lib/sru_nav_go2_ros1/sru_nav_node \| grep -q miniconda3.envs.${ENV_NAME}` | match |
| Node comes up | `timeout 8 roslaunch sru_nav_go2_ros1 sru_nav_go2.launch launch_joy:=false require_joystick:=false` | log contains `Navigation policy node is ready` |
| Required yaml params | `grep -E "policy_scale\|require_joystick\|control_frequency" config/sru_nav.yaml` | all three matched |
| LD_PRELOAD fix | `grep -q 'LD_PRELOAD.*libffi' scripts/launch_sru_nav.sh` | match |

The script exits with `ALL CHECKS PASSED` when (and only when) every
check is green. Do not move to real-robot tests before that.

---

## 8. Common pitfalls

In rough order of how often they bite:

1. **Built `catkin_make` outside the conda env** → node shebang baked to
   `/usr/bin/python3`; `import onnxruntime` fails. Fix:
   `conda activate sru_nav && catkin_make clean && catkin_make -DPYTHON_EXECUTABLE=$(which python3)`.
2. **`No module named netifaces / defusedxml`** → rospy hidden deps.
   `pip install netifaces defusedxml` into the conda env. Already in
   `setup_conda_env.sh`.
3. **`libp11-kit.so.0: undefined symbol: ffi_type_pointer`** → conda
   libffi is loaded first. The `LD_PRELOAD` fix in `launch_sru_nav.sh`
   covers it; on platforms with only `libffi.so.8` change the version
   number.
4. **pip `certificate is not yet valid`** → Jetson clock drift. Fix:
   `sudo ntpdate -u ntp.aliyun.com && sudo hwclock --systohc`.
5. **`/cmd_vel` is all zeros** → deadman is on by default. Either plug
   in the joystick and push `axes[4]`, or run with `--no-deadman` (test
   mode only).
6. **Goal silently dropped** → `/goal_pose.frame_id` must equal the
   odom `frame_id`.

---

## 9. License & acknowledgments

- The algorithm and training code remain copyright of the original SRU
  authors under their original LICENSE.
- This deployment port `sru_nav_go2_ros1` is released under an MIT-style
  license; files derived from upstream retain their original headers.
- Odin1 is a third-party depth camera; this repo does not redistribute
  its driver.

If you find a porting bug or want first-class support for more robots /
cameras, please open an issue or PR.

---

## Appendix A: training–deployment numeric cross-check

| Quantity | Training value | Deployment value | Source |
|---|---|---|---|
| `control_frequency` | 10 Hz (env step) | 5 Hz | `constants.DEFAULT_CONTROL_FREQUENCY` |
| `rnn_hidden_size` | 512 | 512 | `b2w/agents/rsl_rl_cfg.py:36` |
| `rnn_num_layers` | 1 | 1 | same file |
| `policy_scaling` | `[1.5, 1.0, 1.0]` × `Uniform(0.8,1.2)/(0.6,1.0)/(0.8,1.2)` | `policy_scale=[0.6,0.3,0.6]` (conservative) | `navigation_env_cfg.py:163` |
| `entropy_coef` | `0.00375` | n/a | `b2w/agents/rsl_rl_cfg.py:50` |
| `value_loss_coef` | `0.02` | n/a | same file |
| Depth resolution into the VAE | `image_input_dims=(64, 5, 8)` (post-encoder) | input depth resized at runtime | `b2w/agents/rsl_rl_cfg.py:41` |
| `min_depth / max_depth` | 0.25 / 10.0 m | 0.25 / 10.0 m | `constants.py` |
| `JOYSTICK_TIMEOUT` | n/a | 15 s | `constants.py:16` |

This appendix is the minimum viable consistency check. Any second-order
port (different robot / different camera) should preserve these numbers
verbatim.

---

## Appendix B: zero-barrier quickstart from a bare machine

```bash
# === System packages (one-time, sudo) ====================================
sudo apt-get update
sudo apt-get install -y curl git build-essential ros-noetic-desktop \
                        ros-noetic-joy ros-noetic-tf2-tools \
                        python3-catkin-tools

# === Install miniconda (skip if you already have one) ====================
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh
bash Miniconda3-latest-Linux-$(uname -m).sh -b -p $HOME/miniconda3
echo 'source $HOME/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc

# === Clone this package ==================================================
mkdir -p ~/code/odin_sru_nav/src
cd      ~/code/odin_sru_nav/src
git clone https://github.com/<YOUR_FORK>/sru_nav_go2_ros1.git

# === Conda deps (onnxruntime / cv2 / netifaces / ...) ====================
cd sru_nav_go2_ros1
bash scripts/setup_conda_env.sh           # 5–10 minutes the first time
bash scripts/setup_conda_env.sh --check   # all PASS before continuing

# === Build (must be inside the conda env) ================================
conda activate sru_nav
cd ~/code/odin_sru_nav
catkin_make -DPYTHON_EXECUTABLE=$(which python3)
source devel/setup.bash

# === Launch (default: safe mode, requires joystick) ======================
cd src/sru_nav_go2_ros1
bash scripts/launch_sru_nav.sh

# Or: lifted-wheel / sim / closed-loop regression (no joystick) ===========
bash scripts/launch_sru_nav.sh --no-deadman

# === Self-check ==========================================================
bash scripts/verify_port.sh
```

When `verify_port.sh` is fully green, you are ready for real-robot
testing.
