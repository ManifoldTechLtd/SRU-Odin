  # 部署指南：Go2 NX 上的 `sru_nav_go2_ros1`

环境前提（已确认）：
- Go2 onboard NX 已安装 **ROS 1 Noetic**
- 已安装 **odin_ros_driver**（发布 `/odin1/*` 话题）
- 已安装 **unitree_legged_sdk** + 一个把 `/cmd_vel` 桥接到 Go2 sport_mode 的节点
- Odin1 depth 频率 = 10 Hz，odom_highfreq 频率 = 400 Hz

---

## 1. 把本包放进 catkin 工作空间

```bash
mkdir -p ~/sru_ws/src
ln -s /path/to/SRU_Navigation/sru_nav_go2_ros1 ~/sru_ws/src/sru_nav_go2_ros1
cd ~/sru_ws
catkin_make
source devel/setup.bash
```

> 模型 ONNX 文件位于 `sru_nav_go2_ros1/models/` 下（软链到 `sru-robot-deployment/.../deployment_policies/`）。
> 部署到 Go2 NX 时如果没有原仓库，把 `vae_encoder.onnx` 与 `nav_policy.onnx` 直接拷到 `sru_nav_go2_ros1/models/` 即可。

---

## 2. 创建 Python conda 环境

为什么用 conda：onnxruntime / scipy 等 pip 包不污染系统 ROS python；NX (aarch64) 上 miniforge 是最稳的选择。

```bash
cd ~/sru_ws/src/sru_nav_go2_ros1
bash scripts/setup_conda_env.sh
```

脚本会：
1. 检测 `conda`（没有的话提示安装 miniforge for aarch64）
2. 创建名为 `sru_nav` 的环境（**python=3.8**，与 Noetic 系统 python 匹配，才能复用 `rospy` / `cv_bridge`）
3. 安装 `numpy / scipy / opencv-python / onnxruntime / rospkg / pyyaml / empy`
4. 跑一系列 import sanity check，并尝试加载两个 ONNX 模型，打印输入输出维度

**GPU 加速（可选）**：标准 pip 的 `onnxruntime-gpu` 在 Jetson 上**不能用**。需要去 [NVIDIA Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime) 下载 JetPack 匹配的 `onnxruntime_gpu-*-cp38-cp38-linux_aarch64.whl`，然后：

```bash
conda activate sru_nav
pip install ./onnxruntime_gpu-<ver>-cp38-cp38-linux_aarch64.whl
```

不装 GPU 版本也能跑（CPU EP 上 5 Hz 推理在 NX 上一般够用，因为模型只有 ~28 MB）。

### 验证环境

```bash
bash scripts/setup_conda_env.sh --check
```

期望输出关键行：
```
[OK] numpy ...
[OK] scipy ...
[OK] cv2 ...
[OK] onnxruntime ...
ORT providers   : ['CPUExecutionProvider']     # 或包含 CUDAExecutionProvider
[OK] rospy ...
[OK] cv_bridge ...
```

如果 `rospy` / `cv_bridge` 失败：确认 `/opt/ros/noetic/lib/python3/dist-packages` 存在且 conda env 使用 python 3.8。

---

## 3. 启动流程

打开 4 个终端（或用 tmux）。

### Terminal 1 — Odin1 驱动

确保 `odin_ros_driver/config/control_command.yaml` 中：
```yaml
senddepth: 1     # 启用 dense depth
sendodom:  1     # 启用 odom + odometry_highfreq
```

```bash
cd ../OAYN     
mamba activate neupan
source ros_ws/devel/setup.zsh
export ROS_MASTER_URI=http://localhost:11311  
export ROS_HOSTNAME=localhost
export ROS_IP=127.0.0.1
sudo ntpdate -u ntp.aliyun.com 
roslaunch odin_ros_driver odin1_ros1.launch   # 实际 launch 文件名以驱动包为准
```

### Terminal 2 — Go2 cmd_vel 桥

你现有的 unitree_legged_sdk 桥接节点，它订阅 `/cmd_vel`，下发到 Go2 sport_mode。

```bash
mamba activate sru_nav
export ROS_MASTER_URI=http://localhost:11311  
export ROS_HOSTNAME=localhost
export ROS_IP=127.0.0.1
source devel/setup.zsh
rosrun unitree_control unitree_vel_controller __name:=vel_to_sdk
```

### Terminal 3 — SRU 导航节点（本包）

```bash
mamba activate sru_nav
export ROS_MASTER_URI=http://localhost:11311  
export ROS_HOSTNAME=localhost
export ROS_IP=127.0.0.1
cd src/sru_nav_go2_ros1
bash scripts/launch_sru_nav.sh  
# require_joystick=true，必须接手柄，axes[4] 推油门，15 s 看护

bash scripts/launch_sru_nav.sh --no-deadman
# require_joystick=false，cmd_vel_ratio 直接 1.0
# 发一个 /goal_pose，policy 就会驱动 /cmd_vel
```

`launch_sru_nav.sh` 自动做：source ROS → source catkin_ws → conda activate sru_nav → 把 `/opt/ros/noetic/lib/python3/dist-packages` 注入 `PYTHONPATH` → `roslaunch sru_nav_go2_ros1 sru_nav_go2.launch`。

如果手柄 `/joy` 已经在别处启动：
```bash
bash scripts/launch_sru_nav.sh --no-joy
```

如果不需要静态 TF（你自己发了 `base_link → odin1_base_link`）：
```bash
bash scripts/launch_sru_nav.sh --no-tf
```

### Terminal 4 — 发目标 / 监控

```bash
# 推一个 3 m 前方的目标（odom 帧）
mamba activate sru_nav
export ROS_MASTER_URI=http://localhost:11311  
export ROS_HOSTNAME=localhost
export ROS_IP=127.0.0.1
source devel/setup.zsh
rostopic pub /goal_pose geometry_msgs/PoseStamped \
  "{header: {frame_id: 'odom'},
    pose: {position: {x: 3.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}" --once

# 检查输出
rostopic hz /cmd_vel              # 应稳定 ~5 Hz
rostopic echo /cmd_vel -n 5
```

# 或者发布连续waypoints
```bash
mamba activate sru_nav
export ROS_MASTER_URI=http://localhost:11311  
export ROS_HOSTNAME=localhost
export ROS_IP=127.0.0.1
source devel/setup.zsh
cd src/sru_nav_go2_ros1
rosrun sru_nav_go2_ros1 waypoint_runner.py \
  --file config/waypoints_example.yaml
```

---

## 4. 首跑前的 sanity check 命令

```bash
# 4.1 频率检查
rostopic hz /odin1/depth_img_competetion       # 应 ~10 Hz
rostopic hz /odin1/odometry_highfreq           # 应 ~400 Hz

# 4.2 深度数据合理性
rostopic echo /odin1/depth_img_competetion -n 1 | head -20
# 关注 encoding:"32FC1", height/width, 不全为 0

# 4.3 odom frame
rostopic echo /odin1/odometry_highfreq -n 1 | head -20
# 关注 header.frame_id="odom", child_frame_id="odin1_base_link"

# 4.4 节点状态
rosnode info /sru_nav_node
rostopic list | grep sru_nav

# 4.5 关闭 Go2 cmd_vel 桥，单跑导航节点空载，看推理输出
rostopic echo /cmd_vel
# 拖动 Odin1 / 发个 /goal_pose，看 /cmd_vel 是否产生非零有界数值
```

---

## 5. 如何判断 odom twist 的坐标系？

**这是上机前必须确认的**。理论上 `child_frame_id="odin1_base_link"` 暗示 twist 是 body frame，但很多 SLAM 实现仍在 world frame 输出 twist——必须实测。

### 方法 A：静止判断（最快）

让 Odin1 完全静止：
```bash
rostopic echo -n 20 /odin1/odometry_highfreq | grep -A2 twist
```
所有 `linear.*` 和 `angular.*` 都应该接近 0。这一步只能排除"非零偏置 bug"，分不出 world/body。

### 方法 B：纯前进运动

让 Go2 静止站立，**手动**将 Odin1 沿其镜头方向（body +X）匀速推进 1 米，过程录制：
```bash
rosbag record -O /tmp/odom_test.bag /odin1/odometry_highfreq
```
然后：
```bash
rosbag play /tmp/odom_test.bag
rqt_plot /odin1/odometry_highfreq/twist/twist/linear/x:y
```
- 如果 `linear.x` 是显著正值、`linear.y` 几乎 0 → **body frame**（已经是 base 系）
- 如果 `linear.x` 与 `linear.y` 数值与机体当前 yaw 有关（旋转 Odin1 朝向再做同样测试，结果改变）→ **world frame**

### 方法 C：旋转判断（推荐，更确定）

把 Odin1 朝任意非零方向放置（比如绕 Z 转 90°），再沿 body +X 推进：
- **body frame**：无论朝向如何，`linear.x` 总是正、`linear.y` ≈ 0
- **world frame**：朝向 +Y 推进时 `linear.y` 是正、`linear.x` ≈ 0

### 方法 D：对比 dt 数值导数

```python
# 对位置做数值导数，与 twist 对比
import rosbag, numpy as np
bag = rosbag.Bag('/tmp/odom_test.bag')
ts, pos, twist_lin = [], [], []
for _, msg, _ in bag.read_messages(topics=['/odin1/odometry_highfreq']):
    ts.append(msg.header.stamp.to_sec())
    pos.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
    twist_lin.append([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
ts, pos, twist_lin = map(np.array, (ts, pos, twist_lin))
vel_world = np.diff(pos, axis=0) / np.diff(ts)[:, None]
print('vel_world (sample):', vel_world[100])
print('twist_lin (sample):', twist_lin[100])
# 若两者数值在世界系几乎一致 → twist 是 world frame
# 若 twist_lin 与 vel_world 大小相同但方向旋转 → twist 是 body frame
```

### 根据结果配置本节点

- twist **是 world frame**：保持 `use_sim: false`（节点会按机体 quaternion 反旋转到 base 系，与原 DLIO 部署一致）。
- twist **是 body frame**：在 `config/sru_nav.yaml` 把 `use_sim` 设为 `true`，节点会直接透传 twist。

---

## 6. 常见问题

| 现象 | 排查 |
|---|---|
| `pip` 报 `SSL: CERTIFICATE_VERIFY_FAILED ... certificate is not yet valid` | **系统时钟回退了**（Jetson 无 RTC 电池时常见）。`date` 看一下，然后 `sudo timedatectl set-ntp true && sudo systemctl restart systemd-timesyncd`，或离线 `sudo date -s "2026-05-28 14:55:00" && sudo hwclock --systohc`。`setup_conda_env.sh` 已加前置检测。 |
| conda 报 `Unexpected error writing token file ... aau_token_host` | anaconda.org 遥测 bug，无害。脚本里 `export ANACONDA_ANON_USAGE=false` 已静音。 |
| pip 太慢 / 拉不动 | `setup_conda_env.sh` 默认走清华源；若要换 `PIP_INDEX_URL=... bash scripts/setup_conda_env.sh`。 |
| 节点启动后 `Odometry not ready, skipping depth callback.` | `rostopic hz /odin1/odometry_highfreq` 是否在跑？frame_id 是否正确？ |
| `/cmd_vel` 全是 0 | 手柄 deadman——`axes[4]` 必须推过 0；或 15 s 超时。看 `cmd_vel_ratio` 日志。 |
| Twist 数值剧烈震荡 | 1) `policy_scale` 太大；2) odom twist 坐标系判错（world↔body 弄反）；3) depth NaN 没被过滤 |
| `cv_bridge` import 失败 | conda env 必须是 python 3.8；并且 `PYTHONPATH` 已包含 `/opt/ros/noetic/lib/python3/dist-packages` |
| 推理速度跟不上 5 Hz | `rostopic hz /cmd_vel`；用 `nvtop`/`top` 看 NX 满载情况；考虑装 JetPack onnxruntime-gpu wheel |
| Goal 发出后立即被忽略 | `/goal_pose` 的 `frame_id` 必须等于 `/odin1/odometry_highfreq` 的 `header.frame_id`（默认 `odom`） |

---

## 7. 紧急停止

- 手柄按下 `BUTTON_ABORT`（默认 button index 9） → 立刻发零速、放弃当前目标
- 直接 `Ctrl+C` 杀掉 `sru_nav_node` → `/cmd_vel` 不再有发布者，Go2 桥应自动归零（依你的桥行为）
- 关闭电源开关（终极手段）
