# Complete Porting Guide for Reproducing SRU Navigation on Odin1 + ROS1

> Starting from the paper *Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning* and its companion open-source repository [`sru-project-website`](https://michaelfyang.github.io/sru-project-website/), this guide ports the SRU end-to-end navigation policy to the **Odin1 depth camera + ROS1 Noetic + any robot capable of receiving `geometry_msgs/Twist` (`/cmd_vel`)** (with Unitree Go2 as the default hardware reference). The repository `sru_nav_go2_ros1` is the product of this effort.
> **Chinese version**: see [`PORTING_GUIDE_CN.md`](./PORTING_GUIDE_CN.md).

***

## 0. What This Document Is / Is Not

**What it is**: A three-in-one guide containing "Porting Decision Log + AI Replication Prompt + Automated Acceptance Script". After reading, you will:

1. Know why and what we modified;
2. Use a slash command to let an AI agent (Cascade, Claude Code, etc.) **automatically regenerate this repository** starting from the open-source repositories and the paper;
3. Use a single script to verify if the generated results are qualified.

**What it is not**: An explanation of algorithmic principles (please read the paper and original code repositories directly).

**Target Audience**:

- Already possess basic skills with Ubuntu, ROS1, and Python/conda;
- Own a Unitree Go2 (or any mobile robot subscribing to `/cmd_vel`) + Odin1 depth camera;
- Want to run SRU on their own robot, or want to verify "whether an AI agent can replicate a real-robot deployment package from paper + open-source code".

> Zero-barrier commitment: Except for the prerequisite skills mentioned above, all paths, IPs, robot models, and camera models are designed to be **overridden via environment variables or parameters**, without modifying the source code.

***

## 1. Upstream Sources

| Asset | Link / Path | Role in this Port |
| --- | --- | --- |
| Paper | *Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning* | Explains observation/action space, rewards, network architecture (VAE depth encoder + LSTM-SRU + actor head) |
| Project Page | https://michaelfyang.github.io/sru-project-website/ | Indexes repositories, videos, and paper |
| `sru-navigation-learning` | RL training code (rsl_rl style). Provides PPO/MDPO algorithm, reward terms, observation manager | Used for cross-checking **reward design, action squashing, observation hierarchy**, and obtaining the ONNX export script |
| `sru-navigation-sim` | IsaacLab environment, containing B2W / AOW training configurations | Obtains crucial numbers like `policy_scaling`, randomization ranges, observation shape (depth `64×5×8`), etc. |
| `sru-robot-deployment` | Original authors' deployment on Unitree B2W + ZED-X | **The direct reference target for rewriting**: The original uses PyTorch + DLIO odom + custom SDK bridge; this port replaces them with ONNX + Odin1 odom + Go2 bridge |
| `sru-pytorch-spatial-learning` | PyTorch implementation of the SRU/LSTM-SRU cell | Explains RNN hidden state shape (`rnn_hidden_size=512, num_layers=1`), and how the state is flattened when exporting to ONNX |
| `sru-depth-pretraining` | VAE depth encoder pretraining | Obtains `vae_encoder.onnx` input shape (`H×W` normalized depth image, single channel) and latent dimension |

If you do not plan to replicate and just want to run it: drop the two open-source ONNX models (`vae_encoder.onnx` + `nav_policy.onnx`) into `sru_nav_go2_ros1/models/` and skip to §6.

***

## 2. Porting Target Matrix

| Dimension | Authors' Deployment (`sru-robot-deployment`) | This Repository (`sru_nav_go2_ros1`) | Abstracted Interface |
| --- | --- | --- | --- |
| Robot | Unitree **B2W** (wheeled-legged) | Unitree **Go2** (pure legged) | Any robot subscribing to `/cmd_vel` (`geometry_msgs/Twist`, body frame) |
| Depth Sensing | ZED-X | **Odin1** (`32FC1` depth, ~10 Hz) | `~depth_topic` (`sensor_msgs/Image`, 32FC1, meters) |
| Odometry | DLIO (custom LIO) | Odin1 `odometry_highfreq` | `~odom_topic` (`nav_msgs/Odometry`, world frame) |
| Inference Backend | PyTorch JIT | **ONNX Runtime** | `models/{vae_encoder,nav_policy}.onnx` |
| Compute Platform | x86 Industrial PC | **Jetson Orin NX (aarch64) + conda py3.8** | Any Linux + ROS1 Noetic + conda |
| Control Frequency | ~10 Hz | **5 Hz** (training-aligned; Odin depth ~10 Hz is naturally downsampled) | `~control_frequency` |
| Operator Joystick | PS5 PerceptiveNavigationSE2 | Same PS5 mapping, **new `require_joystick: false`** headless mode | `--no-deadman` / yaml `require_joystick` |
| Install & Launch | rosinstall suite | **`setup_conda_env.sh` + `launch_sru_nav.sh`** two scripts | env vars `ENV_NAME / ROS_DISTRO / CATKIN_WS` can all be overridden |

***

## 3. Core Porting Decisions (Motivations & Code References)

### 3.1 Inference: PyTorch → ONNX Runtime

- **Motivation**: On Jetson Orin NX, installing the PyTorch + cv_bridge + onnxruntime dependency stack is highly prone to dependency hell. ONNX Runtime loads a single file, easily switches CPU/GPU EPs, and provides ready-made ARM wheels.
- **Interface**: `LearningModel` in `@/sru_nav_go2_ros1/src/sru_nav_go2/model.py` loads `vae_encoder.onnx` (depth → latent) and `nav_policy.onnx` (latent + state vector + previous action + LSTM hidden → action mean + new hidden).
- **Hidden State Convention**: The training side specifies `rnn_hidden_size=512, num_layers=1`. When exporting to ONNX, `(h, c)` are exposed as input/output tensors, which this node explicitly holds between frames; `reset_hidden_state` corresponds to the `is_first` flag during training.
- **Action Squashing**: The network already contains `tanh`, yielding outputs in `(-1, 1)`. The node then multiplies this by `policy_scale=[vx_max, vy_max, ω_max]` (default `[0.6, 0.3, 0.6]`) to obtain SI velocity commands.

### 3.2 Observation Layer: Odin1 replacing ZED-X + DLIO

- **Depth Image**: Odin1 publishes `sensor_msgs/Image` (`32FC1` in meters). In `@/sru_nav_go2_ros1/src/sru_nav_go2/navigation_policy_node.py:209-225`, we perform:
  1. `cv_bridge.imgmsg_to_cv2(passthrough)`
  2. `nan_to_num(nan=0, posinf=2*max_depth, neginf=0)`
  3. Clip to `[min_depth, max_depth] = [0.25, 10.0] m`, with out-of-bounds pixels set to 0.
- **Odom Frame Convention**: The twist in Odin1 odom is in the **world frame**. In `odom_callback`, we rotate the velocities back to the body frame using the body quaternion (to align with training). If your odom is already in the body frame, simply set `use_sim: true` (implemented in `@/sru_nav_go2_ros1/src/sru_nav_go2/navigation_policy_node.py:189-197`).
- **Goal Frame Convention**: `/goal_pose.header.frame_id` must match the odom's `frame_id` (default `odom`); otherwise, the node rejects it. This prevents the operator from mistakenly publishing body-frame coordinates as odom-frame goals.

### 3.3 Action Output: Direct `/cmd_vel`

- The original authors implemented their own SDK bridge on B2W; for Go2, writing a small node using `unitree_legged_sdk` that subscribes to `/cmd_vel` (separated from this repo) is sufficient. This package is **not** tied to any specific robot SDK bridge.
- The default `policy_scale` is set to `[0.6, 0.3, 0.6]` which is more than twice as conservative as the training value of `[1.5, 1.0, 1.0]`, leaving margin for the physical robot. Since action scales were randomized with `Uniform(0.6, 1.2)` during training, the network is robust to runtime scaling.

### 3.4 Safety: Dual-mode Joystick Deadman

| Mode | Trigger | Behavior |
| --- | --- | --- |
| Default (`require_joystick=true`) | Real deployment | `cmd_vel_ratio=0` initially; `/joy` must be continuously published and `axes[4]` (DEADMAN_RATIO) pushed forward to command velocity. No joy for `>15 s` -> forces velocity to zero. |
| Test (`require_joystick=false`) | `--no-deadman` or yaml set to false | Starts with `cmd_vel_ratio=1.0` immediately. Runs without joystick. **Use only when wheels are lifted / in simulation / in open spaces**. |

Code reference: `@/sru_nav_go2_ros1/src/sru_nav_go2/navigation_policy_node.py:118-121, 272-287`.

> ⚠️ **Do NOT collapse "throttle / forward / deadman" into a single axis.** Upstream
> `sru-robot-deployment` uses **independent** axes for the policy's forward
> command (`AXIS_LINEAR_X = 1`) and the safety deadman/ratio gate
> (`AXIS_DEADMAN_RATIO = 4`). They must stay separate; conflating them disables
> either manual driving or the safety gate.

### 3.4.1 Full PS5 joystick contract (MUST match upstream)

The trained policy and the upstream deployment assume the **exact** axis & button
indices below (source: `sru-robot-deployment/rl_nav_controller/rl_nav_controller/constants.py:19-37`).
**Do not infer these from "common ROS PS5 mappings" — copy them verbatim.**

```text
# axes (sensor_msgs/Joy.axes[i])
AXIS_LINEAR_Y       = 0     # lateral (strafe), used by manual + smart-joy
AXIS_LINEAR_X       = 1     # forward/back, used by manual + smart-joy
AXIS_ANGULAR_Z      = 2     # YAW  (NOT axis 3)
AXIS_LINEAR_Z       = 3     # height delta (smart-joy only)
AXIS_DEADMAN_RATIO  = 4     # deadman gate; cmd_vel_ratio = 1 + axes[4]
AXIS_SMART          = 5     # < -0.5  -> enter smart-joystick goal mode

# buttons (sensor_msgs/Joy.buttons[i])
BUTTON_DOWN              = 0
BUTTON_TRIGGER_WAYPOINTS = 1
BUTTON_RESET_HIDDEN_STATE= 2
BUTTON_UP                = 3
BUTTON_CLEAR_WAYPOINT    = 4
BUTTON_RECORD_WAYPOINT   = 6
BUTTON_ABORT             = 9
BUTTON_SEND_GOAL         = 10
BUTTON_FORWARD           = 11
BUTTON_BACKWARD          = 12
BUTTON_LEFT              = 13
BUTTON_RIGHT             = 14
```

### 3.4.2 Required features to KEEP from `sru-robot-deployment`

Beyond the obvious "subscribe odom/depth/goal/joy, run policy, publish `/cmd_vel`"
loop, the deployment node MUST also implement the following — these are part of
the user-facing safety/usability contract and were validated on real Go2 hardware:

| Feature | Trigger | Notes |
| --- | --- | --- |
| Force reset LSTM hidden state | `BUTTON_RESET_HIDDEN_STATE` (2) | Sets a flag consumed by next `model.predict(..., is_reset=True)`. Operator recovery from stuck recurrent state. |
| Abort current goal | `BUTTON_ABORT` (9) | Clears `target_pos_w`, stops chained waypoint sequence. |
| Record / clear waypoint | `BUTTON_RECORD_WAYPOINT` (6) / `BUTTON_CLEAR_WAYPOINT` (4) | Pushes/pops the current `robot_pos_w` on `WaypointManager`. Cooldown `TRIGGER_BUTTON_COOLDOWN = 1.0 s`. |
| Trigger waypoint sequence | `BUTTON_TRIGGER_WAYPOINTS` (1) | Plays back recorded waypoints, alternating `home` / `inversed` direction. |
| Send moving goal | `BUTTON_SEND_GOAL` (10) | Publishes a body-frame-offset goal accumulated from D-pad nudges (BUTTON_FORWARD/BACKWARD/LEFT/RIGHT/UP/DOWN). |
| Smart-joystick mode | `AXIS_SMART (5) < -0.5` | Continuously generates a rolling body-frame goal from `axes[0,1,3]` scaled by `SMART_JOYSTICK_SCALE = 5.0` / `SMART_JOYSTICK_Z_SCALE = 0.25`, low-pass-filtered with `SMART_JOYSTICK_FILTER_ALPHA = 0.2`. Updated at `SMART_JOYSTICK_UPDATE_FREQUENCY = 5.0 Hz`. |
| Periodic visualization | rospy.Timer at `WAYPOINT_PUBLISH_INTERVAL = TARGET_VECTOR_PUBLISH_INTERVAL = 0.2 s` | Two independent timers publish waypoint markers and the goal-vector arrow (base frame). |
| Near-goal chaining | `_check_near_goal()` returns true within `arrive_goal_threshold * NEAR_GOAL_THRESHOLD_MULTIPLIER (=2.0)` | Used to early-trigger the next waypoint in a sequence, distinct from `_check_goal_reached()`. |
| Own `goal_pose` publisher | Node publishes to `/goal_pose` itself | Required so smart-joystick and waypoint-sequence modes can drive the same callback path as an external goal. |

A correctly ported `navigation_policy_node.py` therefore weighs in around
**550-600 lines** (excluding docstrings). A ~300-line node almost certainly
dropped one of the above blocks.

### 3.5 Installation Layer: conda + System ROS Coexistence (**7 Real-world Gotchas Discovered**)

Jetson Orin NX comes with ROS Noetic (system Python 3.8 + `cv_bridge.so`); onnxruntime / opencv / numpy are easier to install in a conda env. When bridges are built between these two worlds, we encountered several issues (listed in order of discovery):

1. **PYTHONPATH Injection**: `/opt/ros/noetic/lib/python3/dist-packages` must be appended to the conda Python's search path, so that `rospy / cv_bridge / sensor_msgs / tf2_ros` can be imported.
2. **shebang Locking**: `catkin_make` must be executed with `-DPYTHON_EXECUTABLE=$(which python)` **while the conda env is activated**; otherwise, the `#!` in `devel/lib/.../sru_nav_node` will point to `/usr/bin/python3`, failing to import onnxruntime. `launch_sru_nav.sh` implements an **idempotent shebang rewriter** as a fallback.
3. **ABI Conflict (libffi)**: Conda's `libffi.so` is ABI-incompatible with the system's `libp11-kit / cv_bridge` compiled against the system libffi (`undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0`). `launch_sru_nav.sh` searches for the system `libffi.so` using `ldconfig -p` and forces it via `LD_PRELOAD` (typically `.7` on aarch64, `.8` on x86_64).
4. **rospy Implicit Dependencies**: `netifaces` / `defusedxml` are provided via apt in system Python, but must be pip-installed in the conda env. **Symptoms of their absence are highly elusive**: the node starts and transitions to ready normally, but **the moment a subscriber connects to a topic**, the subscription thread crashes with `ModuleNotFoundError`, blocking `/cmd_vel` output. This is now built into `setup_conda_env.sh`.
5. **zsh Variable Leakage -> ROS sourcing setup.zsh by mistake**: When running `bash launch_sru_nav.sh` from a zsh terminal, parent environment variables like `ZSH_VERSION` leak into the bash subprocess, misleading the ROS setup router into sourcing `setup.zsh` instead of `setup.bash`. Since `cd -q` is a zsh-only builtin, bash fails immediately and `set -e` aborts the script. Fixed by unsetting `ZSH_VERSION ZSH_NAME` and exporting `CATKIN_SHELL=bash` before sourcing, wrapped with `set +eu` to prevent uninitialized ROS variables from killing the process.
6. **SIGPIPE Silent Death via `set -e + pipefail`**: In `SYSTEM_LIBFFI=$(ldconfig -p | awk '...{exit}')`, `awk` exiting early causes a SIGPIPE to be sent to `ldconfig`, resulting in exit code 141. Under `pipefail`, the pipeline is evaluated as a failure, silently killing the script under `-e`. Fixed by appending `|| true` to the pipeline.
7. **Clock Sync Issues**: Jetson lack of an RTC battery often reverts system time, causing pip TLS handshakes to fail, and ROS inter-process messaging to drop packets if time offsets exceed 100 ms. **First-time deployments must** execute `sudo ntpdate -u ntp.aliyun.com`.

Complete automation scripts: `@/sru_nav_go2_ros1/scripts/setup_conda_env.sh` and `@/sru_nav_go2_ros1/scripts/launch_sru_nav.sh` address all 7 gotchas.

### 3.6 External Dependencies (not included in this repo)

This repo contains only the SRU policy inference node. The depth/odom driver and `/cmd_vel` actuator bridge are separate:

| Package | Role | conda env (tested) |
| --- | --- | --- |
| `odin_ros_driver` | Odin1 depth camera driver; publishes depth and high-freq odom | `neupan` |
| `unitree_control` (or equivalent) | Subscribes to `/cmd_vel` -> unitree_legged_sdk -> joint cmds | `sru_go2` |

At launch time, the three processes run in three separate terminals; all terminals export `ROS_MASTER_URI=http://localhost:11311`, and the first process to start automatically launches roscore. See §6 and this repo's `docs/DEPLOY.md` §5.

### 3.7 Deliberate Omissions

| Feature in Training | Reason for Omission in Deployment |
| --- | --- |
| Heightmap / heightscan critic | Go2 has no LiDAR; can be added via external topic if needed |
| Critic channels (distance, momentum, etc.) | Critic only used in training; inference only runs actor head |
| MDPO / distillation dual actor-critic | Training-stage optimizer structure; irrelevant to exported actor |
| Training-time reward calculation | Not needed for deployment |
| Action scale training randomization (`Uniform(0.6, 1.2)`) | Fixed scale at deployment; network is already robust |

***

## 4. File List & Roles

```
sru_nav_go2_ros1/
├── CMakeLists.txt                 # catkin package definition; catkin_python_setup() + install scripts
├── package.xml                    # Dependencies: rospy, sensor_msgs, geometry_msgs, nav_msgs, cv_bridge, tf2_ros
├── setup.py                       # Installs src/sru_nav_go2 as a python package
├── README.md
├── config/
│   ├── sru_nav.yaml               # Runtime ROS parameters; the ONLY file users should modify
│   └── waypoints_example.yaml     # Multi-goal navigation example (odom-frame waypoints)
├── launch/
│   └── sru_nav_go2.launch         # joy_node + static_tf + sru_nav_node
├── models/
│   ├── README.md                  # Instructions for exporting ONNX from training
│   ├── vae_encoder.onnx           # Required (User must supply, ~21 MB)
│   └── nav_policy.onnx            # Required (User must supply, ~7 MB)
├── scripts/
│   ├── setup_conda_env.sh         # Creates sru_go2 env; includes PIP mirror fallback, netifaces/defusedxml
│   ├── launch_sru_nav.sh          # One-click launch wrapper (LD_PRELOAD / PYTHONPATH / shebang rewrite
│   │                              #   / CATKIN_SHELL=bash preventing zsh leak / SIGPIPE fallback)
│   ├── sru_nav_node               # ROS node entry point (copied to devel/lib/... during catkin install)
│   ├── waypoint_runner.py         # Publishes sequential /goal_pose
│   └── verify_port.sh             # One-click selfcheck: 6-stage check of package structure, shebang, ONNX, and topics
├── src/sru_nav_go2/
│   ├── __init__.py
│   ├── constants.py               # All training-aligned constants (frequency, joy mapping, scales, etc.)
│   ├── model.py                   # ONNX Runtime wrapper: VAE encoder + LSTM-SRU policy with strict shape validation
│   ├── navigation_policy_node.py  # Main node: odom/depth/joy/goal callbacks + cmd_vel output
│   ├── utils.py                   # Quaternion <-> rotation, reprojection, and math utilities (pure numpy, no torch)
│   ├── visualization.py           # RViz markers (goal vector, moving_goal, waypoints)
│   └── waypoint_manager.py        # Record/play waypoint lists
├── docs/
│   ├── DEPLOY.md                  # Comprehensive deployment manual (10 sections, 11 troubleshooting, 3 terminals)
│   ├── PORTING_GUIDE.md           # Chinese porting guide (in-package copy)
│   ├── PORTING_GUIDE_EN.md        # English version
│   └── IO_SPEC.md                 # ONNX tensor name/shape/dtype contracts
└── notes/
    └── upstream_recon.md          # Upstream reconnaissance summary output by AI agent in Step 1
```

> ⚠️ Note: `models/{vae_encoder,nav_policy}.onnx` **are not bundled in this repository** due to licensing; users must copy them from training outputs (see `models/README.md`).

***

## 5. Replicating this Repo via an AI Agent

This section answers a concrete question: **Once you have this repo, how do you let an AI agent actually generate the SRU porting code for you?**

We provide the same "porting prompt" in 3 formats (choose one):

| Entry File | Tool | Command (run inside the agent chat) |
| --- | --- | --- |
| `.windsurf/workflows/port-sru-to-ros.md` | Cascade (Windsurf IDE) | `/port-sru-to-ros` |
| `.claude/commands/port-sru-to-ros.md` | Claude Code (CLI / IDE extension) | `/port-sru-to-ros` |
| Prompt text in §5.2 of this document | Any agent (Cursor, Continue, Aider, ChatGPT...) | Paste §5.2 into the system/task prompt |

> ⚠️ **Reproducibility Bounds**: Modern LLMs do not achieve a 100% single-shot success rate when generating ~1500 LoC of ROS porting code. We cut the task into a 6-step phased workflow, each step having grep-able validation points. Any step that veers off-course is immediately rolled back and retried, boosting success rate by an order of magnitude compared to single long prompts.

### 5.0 3-Minute Quickstart: What You Actually Need to Do

Yes — **the essence of all slash commands is simply telling the agent to read `.windsurf/workflows/port-sru-to-ros.md` or `.claude/commands/port-sru-to-ros.md` and execute the 6-step process to generate the code**. The slash command is convenient because it handles the step of feeding the ~370-line prompt to the agent, which specifies what to do, what to self-check, and how to recover from failure.

#### Step A: Preparation (Required for all agents)

```text
$WORKSPACE/                         <-- Agent's current working directory (any empty dir)
├── sru_nav_go2_ros1/               <-- Clone/copy this entire repository here
│   ├── .windsurf/workflows/port-sru-to-ros.md     ← For Cascade
│   ├── .claude/commands/port-sru-to-ros.md        ← For Claude Code
│   └── docs/PORTING_GUIDE.md                      ← For generic agents, copy §5.2
└── upstream/                       <-- Place the 5 upstream repositories here
    ├── sru-navigation-learning/
    ├── sru-navigation-sim/
    ├── sru-robot-deployment/
    ├── sru-pytorch-spatial-learning/
    └── sru-depth-pretraining/
```

Minimal Checklist:

1. **This repository** (including `.windsurf/`, `.claude/`, `docs/`) placed in your workspace, since slash commands locate config files via relative paths.
2. **5 upstream repositories** + **paper PDF** (optional but highly recommended): the agent will read them in Step 1 "Reconnaissance". If any are missing, the agent will stop and report what is missing.
3. **Two ONNX models** (`vae_encoder.onnx` + `nav_policy.onnx`): export from training or use ready-made files in the repo's `models/`.

> If you **only want to replicate the ROS package** (without retraining): place the two ONNX files in the agent's working directory, and the prompt will automatically bypass the training-side export step.

#### Step B: Choose invocation method based on your agent

**B-1 · Cascade (Windsurf IDE)**

```text
1. In Windsurf, File -> Open Folder, select the $WORKSPACE directory.
2. In the Cascade chat panel on the right, enter:
       /port-sru-to-ros
3. After hitting Enter, Cascade automatically reads .windsurf/workflows/port-sru-to-ros.md and executes Step 1~6 in order. It will explain what it plans to do at each step and wait for your confirmation.
4. All you need to do is provide file paths when requested, or reply with "retry / skip" on self-check failures.
```

How to confirm Cascade recognizes the workflow: you should see `port-sru-to-ros` under the "Workflows" list in Windsurf settings or the Cascade panel. If it's missing, you may have opened the wrong directory — slash commands are only recognized in the `.windsurf/workflows/` folder of the active workspace root.

**B-2 · Claude Code (CLI or IDE extension)**

```bash
cd $WORKSPACE
claude                                   # Start Claude Code interactive session
# In the Claude prompt:
> /port-sru-to-ros $WORKSPACE upstream sru_go2 noetic
```

All positional arguments are optional; if omitted, they will use default values specified in the frontmatter of `.claude/commands/port-sru-to-ros.md`. Claude will read the `.md` file and execute the 6 steps.

**B-3 · Generic Agents (Cursor / Continue / Aider / Web ChatGPT, etc.)**

```text
1. Copy the entire prompt text in §5.2.
2. Start a new session in your agent, paste the prompt into the system prompt or first user message, specifying:
       - The actual absolute paths of $WORKSPACE and $UPSTREAM_DIR
       - Whether the ONNX models are already present
3. Instruct the agent to follow the 6 steps in the prompt, pasting self-check outputs at each step.
```

#### Step C: How to verify after completion

Regardless of the entry point, run:

```bash
cd $WORKSPACE/sru_nav_go2_ros1
bash scripts/verify_port.sh
```

All green means the agent didn't diverge on key numbers (`policy_scale`, control frequency, LD_PRELOAD, etc.).

#### Step D: Rolling back on failure

The slash command breaks the task into 6 independent phases:

- If a step self-check fails -> tell the agent "Retry Step N, reason: xxx". The agent will roll back to that step **without** rebuilding previously passed parts.
- If the self-check command itself has environment-specific issues -> fix `verify_port.sh` and tell the agent to continue.
- If upstream repos are not found -> the agent will pause and prompt you, resume after supplying them.

---

### 5.1 Phased Workflow Overview (What actually runs under the hood)

| Step | Inputs | Outputs | Validation |
| --- | --- | --- | --- |
| 1. **Reconnaissance** | 5 upstream repos + paper | Summary `notes/upstream_recon.md`: architecture, obs/actions, shapes, hyperparams | File exists and contains layers, obs dims, and `policy_scaling` |
| 2. **Extract Paths** | `export_onnx` of `sru-navigation-learning` | `models/vae_encoder.onnx`, `models/nav_policy.onnx`, `docs/IO_SPEC.md` | ONNX Runtime loads without error, input/output shapes recorded |
| 3. **Catkin Skeleton** | Previous `IO_SPEC.md` | `package.xml / CMakeLists.txt / setup.py / launch/ / config/sru_nav.yaml` | `catkin_make` compiles successfully |
| 4. **Port Node Logic** | Original `sru-robot-deployment` node + `IO_SPEC.md` | `src/sru_nav_go2/{constants,model,utils,visualization,waypoint_manager,navigation_policy_node}.py` + `scripts/sru_nav_node` | Node starts via `roslaunch` without crashing |
| 5. **Deploy Scripts** | Target platform details | `scripts/setup_conda_env.sh` + `scripts/launch_sru_nav.sh` | Clean environment passes §6 validation commands |
| 6. **Docs & Verify** | All artifacts | `README.md / docs/DEPLOY_*.md / scripts/verify_port.sh` | `bash scripts/verify_port.sh` passes fully |

### 5.2 Full Prompt (Copy this for generic agents)

This is the exact prompt body of `.windsurf/workflows/port-sru-to-ros.md` and `.claude/commands/port-sru-to-ros.md` — the "source of truth" is identical across all entry points. If you are not using Cascade or Claude Code, copy the text below:

```
You are an expert engineer skilled in ROS1 Noetic, conda, ONNX Runtime, and PyTorch.
Your goal is to replicate a production-verified SRU deployment package from scratch.
You must strictly adhere to the 7 hard constraints below; no numbers or filenames should be altered.

[Task]
Starting from the upstream sources, generate a catkin package named `sru_nav_go2_ros1`
to deploy the end-to-end navigation policy proposed in the paper "Spatially-Enhanced
Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning"
to an Odin1 depth camera + ROS1 Noetic + any robot consuming `geometry_msgs/Twist (/cmd_vel)`
(reference hardware is Unitree Go2).

[Upstream Sources] (located in $UPSTREAM_DIR/)
- Paper PDF (read using path provided by user, e.g., ./2506.05997v2.pdf)
- Project Homepage: https://michaelfyang.github.io/sru-project-website/
- Repositories:
    sru-navigation-learning      — RL training + ONNX export (extract export_onnx entry point)
    sru-navigation-sim           — IsaacLab env & config (extract policy_scaling, obs dimensions)
    sru-robot-deployment         — Original authors' B2W+ZED-X deployment (direct target for rewriting)
    sru-pytorch-spatial-learning — SRU/LSTM-SRU cell (extract hidden state shape)
    sru-depth-pretraining        — VAE depth pretraining (extract VAE input resolution/latent dimension)
- Pre-existing ONNX models (if available): place at models/{vae_encoder,nav_policy}.onnx

[Hard Constraint ① — Sensor and Control Interfaces]
- Depth topic defaults to `/odin1/depth_img_competetion` (sensor_msgs/Image, 32FC1, in meters).
  The node performs: cv_bridge.imgmsg_to_cv2(passthrough) -> nan_to_num ->
  clip to [min_depth, max_depth]=[0.25, 10.0] m -> resize to (40, 64) for VAE input.
- Odometry topic defaults to `/odin1/odometry_highfreq` (nav_msgs/Odometry, world frame).
  The node rotates the twist back to the body frame using the body quaternion (skip if `use_sim:=true`).
- Control output is fixed to `/cmd_vel` (geometry_msgs/Twist, body frame, m/s + rad/s).
  This package does NOT implement a robot SDK bridge; the bridge (e.g., unitree_control) must be provided separately.
- `/goal_pose.header.frame_id` must match the odom's frame_id; otherwise, the node rejects it.

[Hard Constraint ② — Inference Backend]
- The inference backend must be ONNX Runtime (PyTorch imports/runtime are STRICTLY forbidden).
- ONNX tensor names and shapes must be validated at startup; mismatches must raise immediately:
    vae_encoder.onnx:  input='input' [B,1,40,64]  -> output='mu' [B,64,5,8]
    nav_policy.onnx:   inputs = obs[B,2576] + h[1,B,512] + c[1,B,512]
                       outputs = actions[B,3] + h_new + c_new
  Implementation note: discover names via `session.get_inputs()[i].name` AND assert
  they match the contract above; do NOT hard-code names without the dynamic fallback.
- LSTM hidden states (h, c) must be held explicitly by the node; `reset_hidden_state()`
  must be callable at any time (used by both the warmup and by `BUTTON_RESET_HIDDEN_STATE`).
  Do not auto-reset state when switching goals (aligns with training).
- Logger interface: `LearningModel` must NOT call `.info()` on the ROS module. `rospy`
  has no attribute `info` — its log API is `rospy.loginfo / logwarn / logerr`. Use
  plain `print()` (or `rospy.loginfo` directly) inside `model.py`; do NOT accept a
  generic `logger=rospy` argument that assumes Python `logging.Logger` semantics.
  This crashes the warmup and prevents the node from ever printing
  "Navigation policy node is ready.".

[Hard Constraint ③ — Conda and ROS Coexistence (7 Critical Requirements)]
A) PYTHONPATH Injection: Prepend /opt/ros/$ROS_DISTRO/lib/python3/dist-packages to
   the conda Python search path, allowing rospy and cv_bridge imports.
B) shebang Locking: catkin_make must be executed in the activated conda env with
   -DPYTHON_EXECUTABLE=$(which python); the launch script must rewrite the shebang of
   devel/lib/sru_nav_go2_ros1/sru_nav_node to the conda Python path idempotently.
C) LD_PRELOAD System libffi: Locate the system libffi via `ldconfig -p | awk '/libffi\.so\.[0-9]/{print $NF; exit}'`
   (typically .7 on aarch64, .8 on x86_64) and export LD_PRELOAD to resolve conda cv_bridge ABI conflicts.
D) rospy Implicit Dependencies: setup_conda_env.sh MUST install `netifaces` and `defusedxml` via pip.
   Without them, the node initializes successfully but crashes with ModuleNotFoundError upon the first subscriber.
E) Prevent zsh Variable Leakage: The launch script must export CATKIN_SHELL=bash and unset ZSH_VERSION/ZSH_NAME
   before sourcing ROS setup files, preventing parent shell leakage from executing zsh-only commands like `cd -q`.
F) Prevent SIGPIPE Silence: Ensure ldconfig pipelines wrapped with `|| true` to prevent awk early exit
   from triggering SIGPIPE and killing scripts silently under set -e + pipefail.
G) PIP Mirror Fallback: setup_conda_env.sh must support PIP_INDEX_URL overrides; fallback gracefully
   across pypi.org -> tsinghua -> aliyun -> ustc.

[Hard Constraint ④ — Parameterization and Environment Variables]
- No hard-coded paths (e.g., `/home/user`) are allowed in source code.
- Crucial variables that must be overridable:
    ENV_NAME      default: sru_go2
    PY_VERSION    default: 3.8 (do not change; binds with ROS Noetic ABI)
    ROS_DISTRO    default: noetic
    CATKIN_WS     default: $HOME/catkin_ws
    CONDA_HOME    auto-detected from ~/miniconda3 / ~/anaconda3 / /opt/miniconda3
    PIP_INDEX_URL fallback used if empty; otherwise single source enforced

[Hard Constraint ⑤ — Training-Aligned Constants]
Do not modify the following parameters:
- control_frequency = 5.0 Hz
- rnn_hidden_size = 512, rnn_num_layers = 1
- policy_scale default: [0.6, 0.3, 0.6] (conservative; training original: [1.5, 1.0, 1.0])
- LATERAL_VELOCITY_SCALE = 0.6
- LOW_PASS_FILTER_COEF = [0.9, 0.5, 0.5]
- min_depth = 0.25, max_depth = 10.0
- arrive_goal_threshold = 0.75 m
- NEAR_GOAL_THRESHOLD_MULTIPLIER = 2.0
- joystick_timeout = 15.0 s
- TRIGGER_BUTTON_COOLDOWN = 1.0 s
- SMART_JOYSTICK_SCALE = 5.0, SMART_JOYSTICK_Z_SCALE = 0.25,
  SMART_JOYSTICK_FILTER_ALPHA = 0.2, SMART_JOYSTICK_UPDATE_FREQUENCY = 5.0 Hz
- WAYPOINT_PUBLISH_INTERVAL = TARGET_VECTOR_PUBLISH_INTERVAL = 0.2 s
- PS5 joystick mapping (copy verbatim from
  sru-robot-deployment/rl_nav_controller/rl_nav_controller/constants.py:19-37 —
  do NOT infer from "common ROS PS5 mappings"):
      axes[0] = JOYSTICK_AXIS_LINEAR_Y          (lateral / strafe)
      axes[1] = JOYSTICK_AXIS_LINEAR_X          (forward / back)
      axes[2] = JOYSTICK_AXIS_ANGULAR_Z         (YAW — NOT axis 3)
      axes[3] = JOYSTICK_AXIS_LINEAR_Z          (height delta, smart-joy only)
      axes[4] = JOYSTICK_AXIS_DEADMAN_RATIO     (deadman gate; ratio = 1 + axes[4])
      axes[5] = JOYSTICK_AXIS_SMART             (< -0.5 enters smart-joy mode)
  buttons[0..14] (sensor_msgs/Joy.buttons[i]):
      0=BUTTON_DOWN, 1=BUTTON_TRIGGER_WAYPOINTS, 2=BUTTON_RESET_HIDDEN_STATE,
      3=BUTTON_UP, 4=BUTTON_CLEAR_WAYPOINT, 6=BUTTON_RECORD_WAYPOINT,
      9=BUTTON_ABORT, 10=BUTTON_SEND_GOAL,
      11=BUTTON_FORWARD, 12=BUTTON_BACKWARD, 13=BUTTON_LEFT, 14=BUTTON_RIGHT
  Do NOT collapse "throttle / forward / deadman" into a single axis: AXIS_LINEAR_X (1)
  and AXIS_DEADMAN_RATIO (4) are independent and must both be wired.

[Hard Constraint ⑥ — Safety + Required Operator Features]
- Joystick deadman control and a 15 s timeout.
- Provide a `require_joystick: false` bypass switch for lifted-wheel / simulation
  / closed-loop regression; must default to true. Print a prominent WARNING if
  `require_joystick:=false`.
- The node MUST implement ALL of the following operator features (each is wired
  to a button/axis in Hard Constraint ⑤ and is part of the validated upstream UX).
  Omitting any of these is a regression, not a simplification:
    1. BUTTON_RESET_HIDDEN_STATE -> sets a flag consumed by next
       `model.predict(..., is_reset=True)`.
    2. BUTTON_ABORT -> clears current target_pos_w and any waypoint sequence.
    3. BUTTON_RECORD_WAYPOINT / BUTTON_CLEAR_WAYPOINT -> push/pop on WaypointManager,
       gated by TRIGGER_BUTTON_COOLDOWN.
    4. BUTTON_TRIGGER_WAYPOINTS -> play back waypoints, alternating home/inversed
       direction.
    5. BUTTON_SEND_GOAL + D-pad (FORWARD/BACKWARD/LEFT/RIGHT/UP/DOWN) ->
       accumulate a body-frame delta and publish a /goal_pose. The node must
       own a /goal_pose Publisher (in addition to the Subscriber) so smart-joy
       and waypoint modes can drive the same callback path.
    6. Smart-joystick mode (axes[5] < -0.5) -> at SMART_JOYSTICK_UPDATE_FREQUENCY,
       transform axes[0,1,3] * SMART_JOYSTICK_SCALE (with axes[3] * Z_SCALE) into
       the body frame, low-pass-filter with SMART_JOYSTICK_FILTER_ALPHA, publish
       as a rolling /goal_pose.
    7. Two periodic visualization timers at 0.2 s: one publishes the waypoint
       markers and advances the active waypoint sequence (using
       `_check_near_goal()` with NEAR_GOAL_THRESHOLD_MULTIPLIER); the other
       publishes the goal-vector arrow in the base frame.
  A correctly ported `navigation_policy_node.py` is therefore in the
  **550-600 LoC** range. If your generated node is under ~400 LoC you almost
  certainly dropped one of the blocks above — go back and add it before Step 4
  self-check.

[Hard Constraint ⑦ — Deliberate Omissions (Do not implement)]
- No heightmap/heightscan critic channels (inference only runs the actor).
- No critic head exports.
- No MDPO / distillation details.
- No reward calculations.
- No training-time action scale randomizations.
- (Note: the joystick buttons / smart-joy / waypoints in Hard Constraint ⑥ are
  NOT in this list — they are required deployment features, not training-only.)

[Output File List] (Strictly required)
package.xml, CMakeLists.txt, setup.py
launch/sru_nav_go2.launch
config/sru_nav.yaml, config/waypoints_example.yaml
scripts/setup_conda_env.sh, scripts/launch_sru_nav.sh,
scripts/sru_nav_node, scripts/waypoint_runner.py, scripts/verify_port.sh
src/sru_nav_go2/{__init__.py, constants.py, model.py, utils.py,
                 visualization.py, waypoint_manager.py,
                 navigation_policy_node.py}
docs/DEPLOY.md, docs/PORTING_GUIDE.md (Chinese), docs/PORTING_GUIDE_EN.md,
docs/IO_SPEC.md
notes/upstream_recon.md
models/README.md (explaining ONNX export steps)

[Workflow Steps] (Execute sequentially; self-check each step before proceeding)
Step 1 — Reconnaissance: Before starting, you MUST force-pull the following 5 upstream
        repositories and the paper into $UPSTREAM_DIR/ (i.e., the upstream/ folder); none may be missing:
            git clone sru-pytorch-spatial-learning
            git clone sru-navigation-learning
            git clone sru-navigation-sim
            git clone sru-depth-pretraining
            git clone sru-robot-deployment
        and place the paper PDF (2506.05997v2.pdf) into upstream/.
        If any repository or the paper is missing, report explicitly and pause; do not continue
        without the complete materials.
        Once everything is present, read the repos and paper, extract architectures, obs/actions,
        rsl_rl parameters, and output to notes/upstream_recon.md.
        Self-check: all 5 repository directories and the paper PDF exist under upstream/;
        grep notes/upstream_recon.md for rnn_hidden_size, policy_scaling, obs_dim.
Step 2 — IO Specification: Locate export_onnx in training repositories, extract
        input/output names and shapes, and save to docs/IO_SPEC.md.
        Self-check: Load ONNX models using onnxruntime and assert shapes.
Step 3 — Catkin Skeleton: Generate package configs and run `catkin_make` to verify.
        Self-check: Zero compilation warnings; sru_nav_node entry point exists.
Step 4 — Implement Node: Using original ROS2 node as reference, swap topics,
        remove non-Odin dependencies, replace PyTorch with onnxruntime, and enforce shapes.
        Implement every operator feature listed in Hard Constraint ⑥ (joystick buttons,
        smart-joy, waypoints, two visualization timers, hidden-state reset, /goal_pose
        publisher). Expected node size ~550-600 LoC.
        Self-check (all must pass before moving to Step 5):
          a. `roslaunch` starts without errors AND the log contains BOTH
             "Learning model is ready" (or equivalent warmup-done line) AND
             "Navigation policy node is ready." — if the warmup line is missing,
             the model wrapper most likely tried to call `rospy.info(...)` (which
             does not exist) instead of `print` / `rospy.loginfo` — fix model.py.
          b. `grep -c "BUTTON_" src/sru_nav_go2/navigation_policy_node.py` >= 8
             (must reference at least 8 of the 12 BUTTON_* constants).
          c. `grep -E "AXIS_SMART|SMART_JOYSTICK_SCALE" src/sru_nav_go2/navigation_policy_node.py`
             finds at least one hit (smart-joy mode wired).
          d. `grep -E "rospy\.Timer" src/sru_nav_go2/navigation_policy_node.py | wc -l`
             >= 2 (waypoint + target-vector timers present).
          e. `wc -l src/sru_nav_go2/navigation_policy_node.py` is in [500, 800].
Step 5 — Write setup_conda_env.sh / launch_sru_nav.sh, covering all 7 items (A-G) of Hard
        Constraint ③ and accepting all environment variables of Hard Constraint ④.
        Self-check: setup_conda_env.sh fully OK in a clean env; launch_sru_nav.sh must not be
        silently swallowed by set -e on startup, and key logs ("LD_PRELOAD = ..." /
        "Sourced ROS ..." / "Launching: roslaunch ...") must all be printed.
Step 6 — Generate verify_port.sh + docs (DEPLOY.md / PORTING_GUIDE*.md / IO_SPEC.md /
        models/README.md).
        Self-check: bash scripts/verify_port.sh only SKIPs (not FAILs) when ONNX is missing;
        all PASS when ONNX is present (except phase F real-time ROS topics SKIP when no roscore).

[Acceptance Criteria] (the user will run verify_port.sh and the commands below)
1. `catkin_make` in a clean catkin_ws: 0 warning 0 error.
2. `bash scripts/setup_conda_env.sh --check` all pass (ROS dist-packages may [FAIL] outside
   a container, which is acceptable).
3. `head -1 devel/lib/sru_nav_go2_ros1/sru_nav_node` points to the conda env's python,
   not /usr/bin/python3.
4. `roslaunch sru_nav_go2_ros1 sru_nav_go2.launch require_joystick:=false` starts up and
   shows `Navigation policy node is ready.` without spamming errors.
5. Using `rostopic pub` to send any odom + one 32FC1 depth frame + one goal_pose, `/cmd_vel`
   should output a non-zero Twist (vx ∈ [0, 0.6], vy ∈ [-0.18, 0.18], ωz ∈ [-0.6, 0.6]).
6. `bash scripts/verify_port.sh` all PASS when ONNX is present.
7. In a zsh terminal, `bash scripts/launch_sru_nav.sh require_joystick:=false` starts up
   normally (must not be killed by the setup.zsh `cd -q` error).

[Guidelines]
- Strictly do not generate a single reply longer than 1000+ lines; split by Step, reporting
  the plan before writing files at each step.
- Run the corresponding self-check after each step; paste the commands and outputs back; fix
  immediately on self-check failure.
- If an upstream repository cannot find a file, report explicitly and pause; do not fabricate interfaces.
- Any training constant not in the Hard Constraint ⑤ list must first have its source line number
  recorded in notes/upstream_recon.md before use — do not write from memory.
```

The directly executable versions can be found in `@/.windsurf/workflows/port-sru-to-ros.md` and `@/.claude/commands/port-sru-to-ros.md`.

### 5.3 Verifying "Whether the Prompt Can Replicate This Repository"

This is used for comparing and verifying prompt fidelity on a clean system:

```bash
diff -r --exclude=__pycache__ --exclude=.git \
     ./generated_sru_nav_go2_ros1/ \
     ./sru_nav_go2_ros1/
```

Expected: **Structure is identical. Differences should be limited to comments and literal ordering; critical parameters, topic names, and function signatures must match exactly.**

***

## 6. Customization and Zero-Barrier Use

All variable components are exposed as configurable parameters:

### 6.1 Paths and Environment Variables

| Parameter | Default | Override Command |
| --- | --- | --- |
| Catkin Workspace | `$HOME/catkin_ws` | `CATKIN_WS=/your/workspace_root bash scripts/launch_sru_nav.sh ...` |
| Conda Env Name | `sru_go2` | `ENV_NAME=my_env bash scripts/setup_conda_env.sh` |
| Python Version | `3.8` | `PY_VERSION=3.10 ...` (**strongly discouraged**; tied to ROS Noetic ABI) |
| ROS Distro | `noetic` | `ROS_DISTRO=melodic ...` (Python 3 compatible only) |
| Conda Home | Auto-detected | `CONDA_HOME=/opt/miniconda3 ...` |
| Pip Mirror | Auto fallback | `PIP_INDEX_URL=https://your-mirror/simple ...` |

> ⚠️ `CATKIN_WS` must point to the **workspace root** (the folder containing `src/`), not the package folder. For Go2, this was `/home/unitree/code/odin_sru_nav_from_prompt/`. Persist this command: `echo "export CATKIN_WS=/your/path" >> ~/.zshrc`.

### 6.2 Topics and Frames

Modify the fields in `config/sru_nav.yaml` without changing the source code:

```yaml
depth_topic:    "/your/depth"      # sensor_msgs/Image (32FC1, meters)
odom_topic:     "/your/odometry"   # nav_msgs/Odometry, world frame
joy_topic:      "/joy"
goal_topic:     "/goal_pose"
cmd_vel_topic:  "/cmd_vel"
```

### 6.3 Adapting to Other Robots

Simply supply an actuator bridge that subscribes to `/cmd_vel`. This package is completely independent of specific robot kinematics. Examples:

- Unitree Go2: Write a ~50 line ROS node translating `/cmd_vel` to Sport Mode commands.
- Unitree B2 / B2W: Use the official ROS1 bridge.
- Spot: Use `spot_ros` and subscribe to `/cmd_vel`.
- Simulations: Direct subscription to `/cmd_vel` is supported out-of-the-box.

For TF2, a static transform from `base_link` to `odin1_base_link` is provided in the launch file. Adjust the parameters `odin1_x / y / z / roll / pitch / yaw` according to your mounting offset.

### 6.4 Environment Verification

**Recommended during first-time deployment**:

```bash
# 1) OS / ROS
lsb_release -a                                # Ubuntu 20.04 recommended
echo $ROS_DISTRO                              # should be noetic
which roscore                                 # /opt/ros/noetic/bin/roscore

# 2) Conda
conda --version                               # ≥ 4.10
conda env list | grep -E "sru_go2|base"

# 3) Clock (Jetson RTC batteries are often drained)
date                                          # Verify current time; otherwise pip SSL will fail

# 4) Network
ping -c 2 8.8.8.8 || echo "WARN: no internet, pip will fail"

# 5) Hardware
ls /dev/input/js* 2>/dev/null || echo "INFO: no joystick — use --no-deadman for headless tests"
rostopic list 2>/dev/null | grep odin1 || echo "WARN: Odin1 driver not running"
```

These checks are codified in `verify_port.sh`.

***

## 7. Acceptance Script: `scripts/verify_port.sh`

Designed for **post-generation self-checks**. The script prints detailed `[PASS]/[FAIL]/[WARN]` tags. Key checkpoints:

| Checkpoint | Command | Expected Outcome |
| --- | --- | --- |
| Package Integrity | `find package.xml CMakeLists.txt setup.py launch config models scripts src/sru_nav_go2/{constants,model,navigation_policy_node}.py` | All files exist |
| Model Loading | `python -c "import onnxruntime as ort; ort.InferenceSession('models/vae_encoder.onnx')"` | Loads successfully; prints input shape |
| Catkin Build | `catkin_make -DPYTHON_EXECUTABLE=$(which python3)` | Exit code 0 |
| shebang Match | `head -1 devel/lib/sru_nav_go2_ros1/sru_nav_node | grep -q miniconda3.envs.${ENV_NAME}` | Match |
| Node Launch | `timeout 8 roslaunch sru_nav_go2_ros1 sru_nav_go2.launch launch_joy:=false require_joystick:=false` | Log contains "Navigation policy node is ready" |
| Params Exist | `grep -E "policy_scale|require_joystick|control_frequency" config/sru_nav.yaml` | All three match |
| LD_PRELOAD Present | `grep -q 'LD_PRELOAD.*libffi' scripts/launch_sru_nav.sh` | Match |

Do not deploy on the physical robot unless the terminal prints `ALL CHECKS PASSED`.

***

## 8. Common Pitfalls

Ordered by typical occurrence during deployment (detailed logs and fix details in `sru_nav_go2_ros1/docs/DEPLOY.md` §7):

1. **zsh terminal spawning bash: `setup.zsh: line 7: cd: -q: invalid option`**: The parent shell's `ZSH_VERSION` leaks into the bash subprocess, leading the ROS setup routing to source the zsh branch. Fixed in `launch_sru_nav.sh` by exporting `CATKIN_SHELL=bash` and unsetting `ZSH_VERSION ZSH_NAME` before sourcing.
2. **Scripts exit silently after `PYTHONPATH +=` without logging errors**: Standard Gotcha under `set -e + pipefail + awk '...{exit}'`. Early exit in `awk` triggers SIGPIPE in `ldconfig`, giving exit code 141 and killing the script. Fixed in `launch_sru_nav.sh` by appending `|| true` to the pipeline.
3. **`Missing /home/.../catkin_ws/devel/setup.bash`**: `CATKIN_WS` points to the wrong workspace. Fix: `export CATKIN_WS=/your/real/workspace_root` before running.
4. **`catkin_make` logs "must be invoked in the root of workspace"**: You executed inside `src/sru_nav_go2_ros1/`. Fix: `cd $CATKIN_WS` before invoking `catkin_make`.
5. **`FileNotFoundError: ONNX model not found: .../models/vae_encoder.onnx`**: Model files are missing in `models/`. Ensure both are copied: `vae_encoder.onnx` (~21 MB) and `nav_policy.onnx` (~7 MB).
6. **`No module named 'netifaces'` crashes subscription thread**: The node logs `Navigation policy node is ready` but crashes immediately upon receiving depth/odom data. Fix: `pip install netifaces defusedxml` inside the conda env. This is built into `setup_conda_env.sh`.
7. **`catkin_make` run outside conda environment**: The node shebang gets hardcoded to `/usr/bin/python3`, failing to import `onnxruntime`. Fix: `conda activate sru_go2 && cd $CATKIN_WS && catkin_make clean && catkin_make -DPYTHON_EXECUTABLE=$(which python)`. The `launch_sru_nav.sh` script also rewrites the shebang as a fallback.
8. **pip logs `WARNING: Retrying ... ERROR: Could not find a version`**: Chinese domestic mirrors timed out. `setup_conda_env.sh` automatically falls back to Tsinghua/Alibaba/USTC; force via `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/setup_conda_env.sh`.
9. **`libp11-kit.so.0: undefined symbol: ffi_type_pointer`**: Conda's libffi overrides system symbols, clashing with `cv_bridge`. Fixed in `launch_sru_nav.sh` by dynamically loading system libffi via `LD_PRELOAD` (.7 on aarch64, .8 on x86_64).
10. **pip logs `certificate is not yet valid`**: Jetson time drifted. Fix: `sudo ntpdate -u ntp.aliyun.com && sudo hwclock --systohc`.
11. **`/cmd_vel` is all zero**: Deadman mode is active by default. Push `axes[4]` on your joystick, or launch via `require_joystick:=false` (only on jack stands / in simulation / for closed-loop regression).
12. **Goal pose ignored**: `/goal_pose.header.frame_id` must match the odom's `frame_id` (default `odom`).
13. **No native Noetic on Ubuntu 24.04**: Use the docker image `osrf/ros:noetic-desktop` and mount the workspace. `docs/DEPLOY.md` §7.10 provides complete docker and mirror instructions.

***

## 9. License & Credits

- Algorithm and training code copyright: the original SRU author team, following their original LICENSE.
- This deployment port package `sru_nav_go2_ros1` is open-sourced under an MIT-style license; files containing upstream-derived code retain their respective original headers.
- For Odin1 driver details, see [https://github.com/manifoldsdk/odin_ros_driver](https://github.com/manifoldsdk/odin_ros_driver).

If you find porting bugs or want one-click support for more robots/cameras, issues and PRs are welcome.

***

## Appendix A: Key Constants to Verify with Paper/Training Code

| Name | Training Value | Deployment Value | Source |
| --- | --- | --- | --- |
| `control_frequency` | 10 Hz (env step) | 5 Hz | `constants.DEFAULT_CONTROL_FREQUENCY` |
| `rnn_hidden_size` | 512 | 512 | `b2w/agents/rsl_rl_cfg.py:36` |
| `rnn_num_layers` | 1 | 1 | Same as above |
| `policy_scaling` | `[1.5, 1.0, 1.0]` × `Uniform(0.8,1.2)/(0.6,1.0)/(0.8,1.2)` | `policy_scale=[0.6,0.3,0.6]` (conservative) | `navigation_env_cfg.py:163` |
| `entropy_coef` | `0.00375` | n/a | `b2w/agents/rsl_rl_cfg.py:50` |
| `value_loss_coef` | `0.02` | n/a | Same as above |
| Depth Resolution (VAE input) | Check encoded shape `image_input_dims=(64, 5, 8)` | Any resolution; node downsamples to training specs | `b2w/agents/rsl_rl_cfg.py:41` |
| `min_depth / max_depth` | 0.25 / 10.0 m | 0.25 / 10.0 m | `constants.py` |
| `JOYSTICK_TIMEOUT` | n/a | 15 s | `constants.py:16` |

This table serves as a consistency checklist. Any subsequent ports to other hardware should preserve these values.

***

## Appendix B: Zero-Barrier Quickstart (Real-Robot Path)

This appendix outlines the minimal command sequence to execute. Detailed descriptions are in `sru_nav_go2_ros1/docs/DEPLOY.md`.

### B.1 One-Time Environment Setup

```bash
# === System Level (requires sudo) =========================================
sudo apt-get update
sudo apt-get install -y curl build-essential ros-noetic-desktop \
                        ros-noetic-joy ros-noetic-tf2-tools \
                        python3-catkin-tools ntpdate

# === Install Miniconda (skip if already present) ===========================
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh
bash Miniconda3-latest-Linux-$(uname -m).sh -b -p $HOME/miniconda3
echo 'source $HOME/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc

# === Set Workspace Root (Persisted) =======================================
export CATKIN_WS=$HOME/catkin_ws        # or your actual workspace root
echo "export CATKIN_WS=$CATKIN_WS" >> ~/.zshrc      # or ~/.bashrc
mkdir -p $CATKIN_WS/src && cd $CATKIN_WS/src

# === Clone Repository =====================================================
# Option 1: git clone to src/
git clone https://github.com/<YOUR_FORK>/sru_nav_go2_ros1.git
# Option 2: Extract archive into src/sru_nav_go2_ros1/

# === Place ONNX Models (REQUIRED) ========================================
cp /path/to/vae_encoder.onnx $CATKIN_WS/src/sru_nav_go2_ros1/models/
cp /path/to/nav_policy.onnx  $CATKIN_WS/src/sru_nav_go2_ros1/models/

# === Copy External Dependencies (REQUIRED; not in this repo) ==============
# These packages must be obtained from their respective sources:
cp -r /path/to/odin_ros_driver $CATKIN_WS/src/
cp -r /path/to/unitree_control $CATKIN_WS/src/      # or equivalent cmd_vel→Go2 bridge
# Verify odin_ros_driver/config/control_command.yaml:
#   senddepth: 1     # Enable dense depth
#   sendodom:  1     # Enable odom + odometry_highfreq

# === Build Conda Env (includes onnxruntime / cv2 / netifaces ...) ==========
cd $CATKIN_WS/src/sru_nav_go2_ros1
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    bash scripts/setup_conda_env.sh         # fully automatic; takes 5-10 minutes
bash scripts/setup_conda_env.sh --check     # Should be all [OK] (ROS dist-packages may [FAIL] outside a container, ignorable)

# === Build Package (must be in conda env + workspace root) ================
mamba activate sru_go2
cd $CATKIN_WS                               # Must be workspace root
catkin_make -DPYTHON_EXECUTABLE=$(which python)

# === Self-Check ===========================================================
cd $CATKIN_WS/src/sru_nav_go2_ros1
bash scripts/verify_port.sh                 # All PASS (skips stage F if roscore is absent)
```

### B.2 Three-Terminal Launch (Every Deployment)

Prior to launching, sync system time on **any terminal**:

```bash
sudo ntpdate -u ntp.aliyun.com
```

**Terminal 1 — Odin1 Driver**

```bash
mamba activate neupan                       # Odin driver env
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost ROS_IP=127.0.0.1
cd $HOME/code/OAYN                          # Odin workspace
source ros_ws/devel/setup.zsh
roslaunch odin_ros_driver odin1_ros1.launch
```

**Terminal 2 — Go2 cmd_vel Actuator Bridge**

```bash
mamba activate sru_go2                      # Same env as SRU node
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost ROS_IP=127.0.0.1
cd $CATKIN_WS
source devel/setup.zsh
rosrun unitree_control unitree_vel_controller __name:=vel_to_sdk
```

**Terminal 3 — SRU Navigation Node**

```bash
cd $CATKIN_WS/src/sru_nav_go2_ros1
ENV_NAME=sru_go2 ROS_DISTRO=noetic CATKIN_WS=$CATKIN_WS \
    bash scripts/launch_sru_nav.sh require_joystick:=false
# ↑ require_joystick:=false only for tests; remove parameter for production to enable joystick deadman
```

### B.3 Verifying cmd_vel Communication (Optional Terminal 4)

```bash
mamba activate sru_go2
source /opt/ros/noetic/setup.zsh
source $CATKIN_WS/devel/setup.zsh
export ROS_MASTER_URI=http://localhost:11311

rostopic hz /odin1/depth_img_competetion    # Expected: ~10 Hz
rostopic hz /odin1/odometry_highfreq        # Expected: ~50 Hz
rostopic pub -1 /goal_pose geometry_msgs/PoseStamped \
  '{header: {frame_id: "odom"}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'
rostopic echo /cmd_vel                       # Expected: Non-zero Twist stream
```

Once non-zero `/cmd_vel` starts streaming, deployment is successfully completed. Before launching outdoors, walk through the safety checklist in `docs/DEPLOY.md` §9.
