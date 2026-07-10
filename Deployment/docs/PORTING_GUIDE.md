# 把 SRU 导航策略移植到 Odin1 + ROS1 的完整复现指南

> 从论文 *Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning* 与配套开源仓库
> [`sru-project-website`](https://michaelfyang.github.io/sru-project-website/) 出发，把 SRU 端到端导航策略部署到 **Odin1 深度相机 + ROS1 Noetic + 任意能消费 `geometry_msgs/Twist`（`/cmd_vel`）的机器人**（默认硬件参考为 Unitree Go2）。本仓库 `sru_nav_go2_ros1` 即为此目标的产物。
>
> **English version**: see `PORTING_GUIDE_EN.md`.

---

## 0. 这份文档是什么 / 不是什么

**是**：一份「移植决策记录 + AI 复刻 prompt + 自动验收脚本」的三合一指南。读完即可：

1. 知道我们为什么这样改、改了哪些点；
2. 用一个 slash command 让 AI agent（Cascade / Claude Code 等）从开源仓库与论文出发，**自动重新生成本仓库**；
3. 用一个脚本一键校验生成结果是否合格。

**不是**：算法原理讲解（请直接读论文与原始代码仓）。

**适用读者**：
- 已具备 Ubuntu + ROS1 + Python/conda 基本使用能力；
- 持有 Unitree Go2（或任何能订阅 `/cmd_vel` 的腿足/轮式机器人）+ Odin1 深度相机；
- 想在自己机器人上跑 SRU，或想验证「AI agent 是否能从论文+开源代码复刻一个真机部署包」。

> 0 门槛承诺：除上述前置技能外，所有路径、IP、机器人型号、摄像头型号都设计为**通过环境变量或参数覆盖**，不用改源码。

---

## 1. 上游来源

| 资产 | 链接 / 路径 | 在本移植里的角色 |
|---|---|---|
| 论文 | *Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning* | 解释观测/动作空间、奖励、网络结构（VAE 深度编码器 + LSTM-SRU + actor head） |
| 项目主页 | <https://michaelfyang.github.io/sru-project-website/> | 索引仓库、视频、论文 |
| `sru-navigation-learning` | RL 训练代码（rsl_rl 风格）。给出 PPO/MDPO 算法、reward terms、observation manager | 用于核对 **reward 设计、动作 squash、观测层次**，并取得 ONNX 导出脚本 |
| `sru-navigation-sim` | IsaacLab 环境，含 B2W / AOW 训练配置 | 取得 `policy_scaling`、随机化范围、观测形状（depth `64×5×8`）等关键数字 |
| `sru-robot-deployment` | 原作者在 Unitree B2W + ZED-X 上的真机部署 | **本仓库直接对照重写的对象**：原版用 PyTorch + DLIO odom + 自研 SDK 桥；本仓库换成 ONNX + Odin1 odom + Go2 桥 |
| `sru-pytorch-spatial-learning` | SRU/LSTM-SRU 单元的 PyTorch 实现 | 解释 RNN 隐状态形状（`rnn_hidden_size=512, num_layers=1`），导出 ONNX 时的状态展开方式 |
| `sru-depth-pretraining` | VAE depth encoder 的预训练 | 取得 `vae_encoder.onnx` 的输入形状（`H×W` 深度图归一化后单通道）与潜变量维度 |

如果你不打算复刻、只想跑：把训练完得到的两个 onnx（`vae_encoder.onnx` + `nav_policy.onnx`）丢到 `sru_nav_go2_ros1/models/`，跳到 §6。

---

## 2. 移植目标矩阵

| 维度 | 原作者部署 (`sru-robot-deployment`) | 本仓库 (`sru_nav_go2_ros1`) | 抽象后接口 |
|---|---|---|---|
| 机器人 | Unitree **B2W**（轮足） | Unitree **Go2**（纯腿足） | 任何订阅 `/cmd_vel` (`geometry_msgs/Twist`, body frame) 的机器人 |
| 深度传感 | ZED-X | **Odin1**（`32FC1` depth, ~10 Hz） | `~depth_topic` (`sensor_msgs/Image`, 32FC1, 米) |
| 里程计 | DLIO（自研 LIO） | Odin1 `odometry_highfreq` | `~odom_topic` (`nav_msgs/Odometry`, world frame) |
| 推理后端 | PyTorch JIT | **ONNX Runtime** | `models/{vae_encoder,nav_policy}.onnx` |
| 计算平台 | x86 工控机 | **Jetson Orin NX (aarch64) + conda py3.8** | 任何 Linux + ROS1 Noetic + conda |
| 控制频率 | ~10 Hz | **5 Hz**（与训练一致；Odin depth ~10 Hz 自然降采样） | `~control_frequency` |
| 操作员手柄 | PS5 PerceptiveNavigationSE2 | 同 PS5 mapping，**新增 `require_joystick: false`** 头模式 | `--no-deadman` / yaml `require_joystick` |
| 安装与启动 | rosinstall 全家桶 | **`setup_conda_env.sh` + `launch_sru_nav.sh`** 两脚本 | env vars `ENV_NAME / ROS_DISTRO / CATKIN_WS` 全部可覆盖 |

---

## 3. 核心移植决策（逐项说明动机与对照位置）

### 3.1 推理：PyTorch → ONNX Runtime

- **动机**：Jetson Orin NX 上装 PyTorch + cv_bridge + onnxruntime 的依赖网很容易爆；onnxruntime 单文件加载、CPU/GPU EP 切换简单；ARM wheel 现成。
- **接口**：`@/sru_nav_go2_ros1/src/sru_nav_go2/model.py` 实现 `LearningModel`，分别加载 `vae_encoder.onnx`（深度→latent）与 `nav_policy.onnx`（latent + 状态向量 + 上一动作 + LSTM hidden → action mean + 新 hidden）。
- **隐状态约定**：训练侧 `rnn_hidden_size=512, num_layers=1`；导出 ONNX 时把 `(h, c)` 作为输入输出张量，本端在每帧之间显式持有；`reset_hidden_state` 对应训练里 `is_first` 标志。
- **action squash**：网络已包含 `tanh`，输出在 `(-1, 1)`。本端再乘 `policy_scale=[vx_max, vy_max, ω_max]`（默认 `[0.6, 0.3, 0.6]`）得到 SI 单位速度。

### 3.2 观测层：Odin1 替代 ZED-X + DLIO

- **深度图**：Odin1 发布 `sensor_msgs/Image` (`32FC1`，单位米)。`@/sru_nav_go2_ros1/src/sru_nav_go2/navigation_policy_node.py:209-225` 里做：
  1. `cv_bridge.imgmsg_to_cv2(passthrough)`；
  2. `nan_to_num(nan=0, posinf=2*max_depth, neginf=0)`；
  3. 截断到 `[min_depth, max_depth] = [0.25, 10.0] m`，超界置 0。
- **里程计帧约定**：Odin1 odom 的 twist 是 **world frame**。我们在 `odom_callback` 里用机体四元数把它反旋到 body frame（与训练对齐）。如果你的 odom 已经是 body frame，把 `use_sim: true` 透传即可（`@/sru_nav_go2_ros1/src/sru_nav_go2/navigation_policy_node.py:189-197`）。
- **goal 帧约定**：`/goal_pose.header.frame_id` 必须等于 odom 的 `frame_id`（默认 `odom`），否则节点直接拒绝 — 这是为了防止操作员误把机体系坐标当成 odom 系发出去。

### 3.3 动作下发：直接 `/cmd_vel`

- 原作者在 B2W 上自带 SDK 桥；Go2 直接用 unitree_legged_sdk 写一个订阅 `/cmd_vel` 的小节点（与本仓库分离）即可。本包**不**绑定具体桥。
- `policy_scale` 默认 `[0.6, 0.3, 0.6]` 比训练侧 `[1.5, 1.0, 1.0]` 保守一半多，给真机留余量；训练时已经做过 `Uniform(0.6, 1.2)` 的 scale 随机化，所以网络对运行时缩放鲁棒。

### 3.4 安全：手柄 deadman 双模

| 模式 | 触发 | 行为 |
|---|---|---|
| 默认 (`require_joystick=true`) | 实机部署 | `cmd_vel_ratio=0`，必须 `/joy` 持续发布且 `axes[4]` 推到正才有速度；`>15 s` 无 joy → 强制归零 |
| 测试 (`require_joystick=false`) | `--no-deadman` 或 yaml 改 false | 起步即 `cmd_vel_ratio=1.0`，无 joy 也跑。**只在抬轮 / 仿真 / 空旷场地**用 |

代码位置：`@/sru_nav_go2_ros1/src/sru_nav_go2/navigation_policy_node.py:118-121, 272-287`。

### 3.5 安装层：conda + 系统 ROS 共生

Jetson Orin NX 出厂有 ROS Noetic（系统 python3.8 + cv_bridge.so）；同时我们要 onnxruntime / opencv / numpy 等只在 conda env 里好装。两个世界要打通：

1. **PYTHONPATH 注入**：`/opt/ros/noetic/lib/python3/dist-packages` 加到 conda python 的搜索路径，使 `rospy / cv_bridge / sensor_msgs / tf2_ros` 可 import。
2. **shebang 锁定**：catkin_make 时必须 `-DPYTHON_EXECUTABLE=$(which python3)` **在 conda env 已激活** 状态下跑，否则 `devel/lib/.../sru_nav_node` 的 `#!` 会指向 `/usr/bin/python3`，import onnxruntime 失败。
3. **ABI 冲突修复**：conda env 自带的 `libffi.so.7` 和系统 `libp11-kit.so.0` 版本符号不兼容，导致 `cv_bridge` 调 OpenCV 时崩溃（`undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0`）。`launch_sru_nav.sh` 自动 `LD_PRELOAD=/lib/aarch64-linux-gnu/libffi.so.7` 提前加载系统 libffi。
4. **rospy 隐性依赖**：`netifaces` / `defusedxml` 在系统 python 里靠 apt 提供，conda env 里要 pip 装。`setup_conda_env.sh` 已加进默认列表。
5. **时钟检查**：Jetson 无 RTC 电池常导致系统时间回退，pip TLS 拒接握手。脚本里前置 `date` 检测并提示 `ntpdate / timedatectl`。

完整自动化脚本：`@/sru_nav_go2_ros1/scripts/setup_conda_env.sh` + `@/sru_nav_go2_ros1/scripts/launch_sru_nav.sh`。

### 3.6 故意删减项

| 训练里有的东西 | 部署里去掉的原因 |
|---|---|
| Heightmap / heightscan critic | Go2 不带 LiDAR/heightmap，需要的话外部话题再加 |
| 距离/动量等 critic 通道 | critic 仅训练用，推理只跑 actor head |
| MDPO / distillation 双 actor-critic 互蒸馏 | 训练阶段优化器结构，与导出后的 actor 无关 |
| 训练时 reward terms 计算 | 部署不需要计算 reward |
| Action scale randomization (`Uniform(0.6, 1.2)`) | 部署是固定缩放；网络已经被训练成对该范围鲁棒 |

---

## 4. 文件清单与职能

```
sru_nav_go2_ros1/
├── CMakeLists.txt                 # catkin 包定义；catkin_python_setup() + install scripts
├── package.xml                    # 依赖：rospy, sensor_msgs, geometry_msgs, nav_msgs, cv_bridge, tf2_ros
├── setup.py                       # 把 src/sru_nav_go2 装成 python 包
├── README.md
├── config/
│   ├── sru_nav.yaml               # 运行时全部 ROS 参数；唯一应该被用户改的文件
│   └── waypoints_example.yaml     # 多目标巡航示例（odom 系列点）
├── launch/
│   └── sru_nav_go2.launch         # joy_node + static_tf + sru_nav_node
├── models/
│   ├── vae_encoder.onnx           # 训练侧 export_onnx 产物
│   └── nav_policy.onnx
├── scripts/
│   ├── setup_conda_env.sh         # 创建 sru_nav env，装 onnxruntime/cv2/netifaces…
│   ├── launch_sru_nav.sh          # 一键启动 wrapper（含 LD_PRELOAD / PYTHONPATH 修复）
│   ├── sru_nav_node               # ROS 节点入口（catkin 安装时拷到 devel/lib/...）
│   └── waypoint_runner.py         # 顺序发 /goal_pose
├── src/sru_nav_go2/
│   ├── constants.py               # 所有训练对齐常量（控制频率、轴键映射、scale 等）
│   ├── model.py                   # ONNX Runtime 封装：VAE encoder + LSTM-SRU policy
│   ├── navigation_policy_node.py  # 主节点：odom/depth/joy/goal 回调 + cmd_vel 输出
│   ├── utils.py                   # 四元数<->旋转、reproject 等数学小工具
│   ├── visualization.py           # rviz marker（目标向量、moving_goal、waypoint）
│   └── waypoint_manager.py        # 录制/回放 waypoint 列表
└── docs/
    ├── DEPLOY_GO2_NX.md           # 部署到 NX 的实战记录
    └── PORTING_GUIDE.md           # ← 你正在读的文件
```

---

## 5. 用 AI Agent 自动复刻本仓库

我们提供 **同一份移植 prompt 的三种调用形式**（内容等价，挑顺手的用）：

| 入口 | 适用工具 | 启动方式 |
|---|---|---|
| `.windsurf/workflows/port-sru-to-ros.md` | Cascade（Windsurf IDE） | 在 Cascade 聊天框里输 `/port-sru-to-ros` |
| `.claude/commands/port-sru-to-ros.md` | Claude Code (CLI / IDE 插件) | `claude` 启动后 `/port-sru-to-ros` |
| `docs/PORTING_PROMPT.md` | 任意支持自定义 prompt 的 agent（Cursor、Continue、Aider…） | 把整篇内容粘进系统提示 / 任务描述 |

三个文件由同一份「**真理 prompt**」生成，差异只在启动头与文件路径约定。下面是完整的 prompt 内容（放在 `.windsurf/workflows/port-sru-to-ros.md` 里以 frontmatter 起头）。

> ⚠️ **可重复性边界**：现役 LLM 在生成 ~1500 LoC 的 ROS 移植代码时，单次成功率不是 100%。我们用 **分阶段 workflow** 把任务切成 6 步，每步都有可 grep 的验收点；任意一步偏题立刻回滚重试，比一次性长 prompt 的成功率高一个数量级。

### 5.1 分阶段 workflow 概览

| 阶段 | 输入 | 输出 | 验收 |
|---|---|---|---|
| 1. **勘查** | 上游 5 个仓库 + 论文 | 摘要 `notes/upstream_recon.md`：算法、观测/动作、I/O 形状、训练超参 | 文件存在且包含网络层数、obs dims、policy_scale |
| 2. **抽取推理路径** | `sru-navigation-learning` 的 export_onnx | `models/vae_encoder.onnx`、`models/nav_policy.onnx`、`docs/IO_SPEC.md` | onnxruntime 可加载，输入/输出名/形状记录 |
| 3. **生成 ROS 包骨架** | 上一步 IO_SPEC | `package.xml / CMakeLists.txt / setup.py / launch/ / config/sru_nav.yaml` | `catkin_make` 通过 |
| 4. **移植节点逻辑** | 原 `sru-robot-deployment` 节点 + IO_SPEC | `src/sru_nav_go2/{constants,model,utils,visualization,waypoint_manager,navigation_policy_node}.py` + `scripts/sru_nav_node` | 节点能 `roslaunch` 起来不 crash |
| 5. **写部署脚本** | 目标 platform 信息 | `scripts/setup_conda_env.sh` + `scripts/launch_sru_nav.sh` | 在干净 env 上跑通 §6 验收命令 |
| 6. **文档与验收** | 全部产物 | `README.md / docs/DEPLOY_*.md / scripts/verify_port.sh` | `bash scripts/verify_port.sh` 全绿 |

### 5.2 通用 prompt（与三个入口文件一致的核心内容）

```text
你是一名熟悉 ROS1 Noetic、conda、ONNX Runtime 与 PyTorch 的高级工程师。

【任务】
从以下上游资料出发，生成一个名为 `sru_nav_go2_ros1` 的 catkin 包，
把论文《Spatially-Enhanced Recurrent Memory for Long-Range Mapless
Navigation via End-to-End Reinforcement Learning》提出的端到端导航策略
部署到 Odin1 深度相机 + ROS1 Noetic + 任意能消费
`geometry_msgs/Twist (/cmd_vel)` 的机器人上。

【上游资料】
- 论文 PDF（按用户提供的路径读取，例如 ./2506.05997v2.pdf）
- 项目主页 https://michaelfyang.github.io/sru-project-website/
- 仓库（按用户工作区下子文件夹）：
    sru-navigation-learning      — RL 训练 + ONNX 导出
    sru-navigation-sim           — IsaacLab 环境与训练配置
    sru-robot-deployment         — 原作者 B2W+ZED-X 真机部署（直接对标重写）
    sru-pytorch-spatial-learning — SRU/LSTM-SRU 单元
    sru-depth-pretraining        — VAE depth encoder 预训练
- 已存在的 onnx 模型（如有）：models/{vae_encoder,nav_policy}.onnx

【硬约束】
1. 目标传感器固定为 Odin1。深度话题默认 `/odin1/depth_img_competetion`
   (sensor_msgs/Image, 32FC1, 单位米)；里程计默认
   `/odin1/odometry_highfreq` (nav_msgs/Odometry, world frame)。
2. 控制输出固定为 `/cmd_vel` (geometry_msgs/Twist, body frame)，由调用方
   桥接到具体机器人；本包不实现机器人 SDK 桥。
3. 推理后端只能用 onnxruntime（不引入 PyTorch 运行时依赖）。
4. 必须支持 conda env (默认 `sru_nav`, py3.8) 与系统 ROS 共生：
   - PYTHONPATH 注入 /opt/ros/<DISTRO>/lib/python3/dist-packages
   - 启动脚本 LD_PRELOAD 系统 libffi 修复 cv_bridge 的 ABI 冲突
   - catkin_make 时锁定 PYTHON_EXECUTABLE 为 conda python
5. 所有路径、env、distro、机器人型号必须可被环境变量或 ROS 参数覆盖；
   不得在源码里硬写 `/home/<user>` 一类 path。
6. 保留训练对齐的全部数值（control_frequency=5Hz, rnn_hidden=512,
   policy_scale 默认 [0.6,0.3,0.6]，joystick axis 映射等）。
7. 关键安全机制：手柄 deadman + 15 s 超时；提供 `require_joystick:
   false` 旁路开关，但默认必须为 true。

【输出文件清单】（缺一不可）
package.xml, CMakeLists.txt, setup.py
launch/sru_nav_go2.launch
config/sru_nav.yaml, config/waypoints_example.yaml
scripts/setup_conda_env.sh, scripts/launch_sru_nav.sh,
scripts/sru_nav_node, scripts/waypoint_runner.py, scripts/verify_port.sh
src/sru_nav_go2/{__init__.py, constants.py, model.py, utils.py,
                 visualization.py, waypoint_manager.py,
                 navigation_policy_node.py}
docs/DEPLOY.md, docs/PORTING_GUIDE.md (中文), docs/PORTING_GUIDE_EN.md
models/README.md (说明从训练侧导出 onnx 的步骤)

【工作流程】（必须按顺序，每步完成后 self-check 再进下一步）
Step 1 — 勘查上游：读取仓库与论文，列出网络架构、观测维度、动作空间、
        rsl_rl 算法配置、reward 列表，输出到 notes/upstream_recon.md。
Step 2 — IO 规范化：在训练仓里找到 export_onnx 入口，记录 vae_encoder
        与 nav_policy 的 input/output 名与 shape，写到 docs/IO_SPEC.md。
Step 3 — 生成 catkin 骨架并 `catkin_make` 验证。
Step 4 — 实现节点：以 sru-robot-deployment 的 node 为模板，
        替换传感器话题、剥离非 Odin 依赖、把 PyTorch 调用换成 onnxruntime。
Step 5 — 编写两份脚本（setup_conda_env.sh / launch_sru_nav.sh），
        覆盖 §硬约束 4 的全部点。
Step 6 — 生成 verify_port.sh：检查包结构、shebang 指向、ONNX 推理、
        rostopic 列表、yaml 必填字段。

【验收标准】（用户会跑 verify_port.sh 与下述命令）
1. `catkin_make` 在干净 catkin_ws 下 0 warning 0 error。
2. `bash scripts/setup_conda_env.sh --check` 全部通过。
3. `head -1 devel/lib/sru_nav_go2_ros1/sru_nav_node` 指向
   conda env 的 python，不是 /usr/bin/python3。
4. `roslaunch sru_nav_go2_ros1 sru_nav_go2.launch require_joystick:=false`
   起来后看到 `Navigation policy node is ready.` 且不刷错误。
5. 用 `rostopic pub` 发任意 odom + 一帧 32FC1 depth + 一个 goal_pose，
   `/cmd_vel` 应输出非零 Twist。
6. `bash scripts/verify_port.sh` 全部 PASS。

【回答方式】
- 严禁直接生成 1000+ 行的单条回复；按 Step 拆分，每步先汇报计划再写文件。
- 每步结束后跑相应 self-check，把命令与输出贴回；自检失败立即修复。
- 如果上游仓库找不到某文件，明确报告并暂停，不得编造接口。
```

完整可直接执行的版本见 `@/.windsurf/workflows/port-sru-to-ros.md` 与 `@/.claude/commands/port-sru-to-ros.md`。

### 5.3 怎样验证「这份 prompt 真的能复刻本仓库」？

把上述 workflow 在一台干净机器上跑完，最后做：

```bash
diff -r --exclude=__pycache__ --exclude=.git \
     ./generated_sru_nav_go2_ros1/ \
     ./sru_nav_go2_ros1/
```

预期：**结构完全一致；逐行 diff 应集中在注释 / 字面量顺序上，不应在数值常量、话题名、函数签名上有差异**。如果出现关键差异（例如 `policy_scale` 改成 `[1, 1, 1]`、忘记 LD_PRELOAD），说明 agent 没遵 prompt，回到对应 Step 重启。

---

## 6. 个性化与 0 门槛使用

所有「可能因人而异」的点都开成参数。最常见的三类：

### 6.1 路径

| 参数 | 默认 | 覆盖方式 |
|---|---|---|
| catkin workspace | `$HOME/code/odin_sru_nav` | `CATKIN_WS=/path bash scripts/launch_sru_nav.sh` |
| conda env 名 | `sru_nav` | `ENV_NAME=my_env bash scripts/setup_conda_env.sh` |
| ROS distro | `noetic` | `ROS_DISTRO=melodic bash scripts/launch_sru_nav.sh`（仅 py3 兼容版本） |
| pip 镜像 | 清华源 | `PIP_INDEX_URL=https://pypi.org/simple bash scripts/setup_conda_env.sh` |

### 6.2 话题与帧

改 `config/sru_nav.yaml` 里这些字段，**不用改源码**：

```yaml
depth_topic:    "/your/depth"      # sensor_msgs/Image (32FC1, meters)
odom_topic:     "/your/odometry"   # nav_msgs/Odometry, world frame
joy_topic:      "/joy"
goal_topic:     "/goal_pose"
cmd_vel_topic:  "/cmd_vel"
```

### 6.3 不同机器人

只需提供「订阅 `/cmd_vel` 的桥」。仓库**完全不感知机器人型号**。常见替换：

- Unitree Go2：用 `unitree_legged_sdk` 写 50 行的 cmd_vel→sport_mode 桥；
- Unitree B2 / B2W：用官方 ROS1 桥；
- Spot：`spot_ros` + `cmd_vel`；
- 任何 ROS 仿真：直接订阅 `/cmd_vel`。

唯一需要改的可能是 TF：默认 launch 里发 `base_link → odin1_base_link` 的静态 TF，参数 `odin1_x / y / z / roll / pitch / yaw`，按你的相机安装位置改。

### 6.4 检查环境（你提到要写进文档与 prompt 的部分）

**第一次部署时建议跑一遍**：

```bash
# 1) OS / ROS
lsb_release -a                                # Ubuntu 20.04 推荐 (Noetic 配套)
echo $ROS_DISTRO                              # 应为 noetic
which roscore                                 # /opt/ros/noetic/bin/roscore

# 2) Conda
conda --version                               # ≥ 4.10
conda env list | grep -E "sru_nav|base"

# 3) 时钟（Jetson 无 RTC 电池常坏）
date                                          # 确认是真实当前时间，否则 pip TLS 会拒接

# 4) 网络
ping -c 2 8.8.8.8 || echo "WARN: no internet, pip will fail"

# 5) 硬件
ls /dev/input/js* 2>/dev/null || echo "INFO: no joystick — use --no-deadman for headless tests"
rostopic list 2>/dev/null | grep odin1 || echo "WARN: Odin1 driver not running"
```

`scripts/verify_port.sh` 里把这一段固化为前置检查。

---

## 7. 验收脚本：`scripts/verify_port.sh`

设计成「**生成完代码后，开发者一键自检**」。完整列表见脚本本身（自带带 `[PASS]/[FAIL]/[WARN]` 标记），核心检查项：

| 检查项 | 命令 | 预期 |
|---|---|---|
| 包结构完整 | `find package.xml CMakeLists.txt setup.py launch config models scripts src/sru_nav_go2/{constants,model,navigation_policy_node}.py` | 全部存在 |
| 模型可加载 | `python -c "import onnxruntime as ort; ort.InferenceSession('models/vae_encoder.onnx')"` | 不抛异常，打印 input shape |
| catkin 通过 | `catkin_make -DPYTHON_EXECUTABLE=$(which python3)` | exit 0 |
| shebang 正确 | `head -1 devel/lib/sru_nav_go2_ros1/sru_nav_node \| grep -q miniconda3.envs.${ENV_NAME}` | match |
| 节点能起 | `timeout 8 roslaunch sru_nav_go2_ros1 sru_nav_go2.launch launch_joy:=false require_joystick:=false` | log 含 `Navigation policy node is ready` |
| 关键参数存在 | `grep -E "policy_scale\|require_joystick\|control_frequency" config/sru_nav.yaml` | 三项都命中 |
| LD_PRELOAD 修复 | `grep -q 'LD_PRELOAD.*libffi' scripts/launch_sru_nav.sh` | 命中 |

跑完终端打印 `ALL CHECKS PASSED` 才能放行真机。

---

## 8. 常见陷阱

按踩坑频率排序：

1. **catkin_make 时没在 conda env 里**：节点 shebang 烤进 `/usr/bin/python3`，运行时 `import onnxruntime` 失败。修：`conda activate sru_nav && catkin_make clean && catkin_make -DPYTHON_EXECUTABLE=$(which python3)`。
2. **`No module named netifaces / defusedxml`**：rospy 隐性依赖。`pip install netifaces defusedxml` 进 conda env，`setup_conda_env.sh` 已内置。
3. **`libp11-kit.so.0: undefined symbol: ffi_type_pointer`**：conda libffi 抢加载。`launch_sru_nav.sh` 里的 `LD_PRELOAD` 已修，新平台只需把版本号 7→8 同步。
4. **pip `certificate is not yet valid`**：Jetson 时钟回退。`sudo ntpdate -u ntp.aliyun.com && sudo hwclock --systohc`。
5. **`/cmd_vel` 全 0**：deadman 模式默认开。要么接手柄推 `axes[4]`，要么 `--no-deadman`（仅测试）。
6. **goal 被忽略**：`/goal_pose.frame_id` 必须等于 odom 的 `frame_id`。

---

## 9. 许可与致谢

- 算法与训练代码版权：原 SRU 作者团队，遵循其原 LICENSE。
- 本部署移植包 `sru_nav_go2_ros1` 在 MIT-style 许可下开源；包含上游派生代码的部分文件保留各自原始 header。
- Odin1 是第三方深度相机，本仓库不再分发其驱动。

如发现移植 bug 或想要一键支持更多机器人/相机，欢迎 issue + PR。

---

## 附录 A：与论文/训练代码核对的关键数字

| 名称 | 训练值 | 部署值 | 来源 |
|---|---|---|---|
| `control_frequency` | 10 Hz (env step) | 5 Hz | `constants.DEFAULT_CONTROL_FREQUENCY` |
| `rnn_hidden_size` | 512 | 512 | `b2w/agents/rsl_rl_cfg.py:36` |
| `rnn_num_layers` | 1 | 1 | 同上 |
| `policy_scaling` | `[1.5, 1.0, 1.0]` × `Uniform(0.8,1.2)/(0.6,1.0)/(0.8,1.2)` | `policy_scale=[0.6,0.3,0.6]`（保守） | `navigation_env_cfg.py:163` |
| `entropy_coef` | `0.00375` | n/a | `b2w/agents/rsl_rl_cfg.py:50` |
| `value_loss_coef` | `0.02` | n/a | 同上 |
| 深度图分辨率（VAE 输入） | 见 `image_input_dims=(64, 5, 8)` 编码后形状 | 任意输入分辨率，节点 resize 到训练规格 | `b2w/agents/rsl_rl_cfg.py:41` |
| `min_depth / max_depth` | 0.25 / 10.0 m | 0.25 / 10.0 m | `constants.py` |
| `JOYSTICK_TIMEOUT` | n/a | 15 s | `constants.py:16` |

附录里这张表是「训练-部署一致性」的最小验收清单。任何二次移植（换机器人 / 换相机）都应保留这些数字不变。

---

## 附录 B：从空机器开始的 0 门槛 quickstart

```bash
# === 系统级（一次性，需 sudo） =============================================
sudo apt-get update
sudo apt-get install -y curl git build-essential ros-noetic-desktop \
                        ros-noetic-joy ros-noetic-tf2-tools \
                        python3-catkin-tools

# === 装 miniconda（如已装可跳过） ==========================================
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh
bash Miniconda3-latest-Linux-$(uname -m).sh -b -p $HOME/miniconda3
echo 'source $HOME/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc

# === 拉本仓库（替换成你的 fork / 仓库地址） ================================
mkdir -p ~/code/odin_sru_nav/src
cd      ~/code/odin_sru_nav/src
git clone https://github.com/<YOUR_FORK>/sru_nav_go2_ros1.git

# === 装 conda 依赖（含 onnxruntime / cv2 / netifaces ...） =================
cd sru_nav_go2_ros1
bash scripts/setup_conda_env.sh           # 全自动；首次约 5–10 分钟
bash scripts/setup_conda_env.sh --check   # 自检全 OK 才继续

# === 编译（必须在 conda env 里） ===========================================
conda activate sru_nav
cd ~/code/odin_sru_nav
catkin_make -DPYTHON_EXECUTABLE=$(which python3)
source devel/setup.bash

# === 一键启动（默认安全模式：必须接手柄） ==================================
cd src/sru_nav_go2_ros1
bash scripts/launch_sru_nav.sh

# 或：抬轮 / 仿真 / 闭环回归测试（无手柄） ==================================
bash scripts/launch_sru_nav.sh --no-deadman

# === 一键自检 ==============================================================
bash scripts/verify_port.sh
```

到这里只要 `verify_port.sh` 全绿，就能开始真机测试。
