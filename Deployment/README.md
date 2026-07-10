# sru_nav_go2_ros1

ROS 1 (Noetic) port of the SRU autonomous navigation controller, adapted for a
**Unitree Go2 + Odin1** stack.

> 📘 **Want to reproduce or retarget this package?**
> See **[`docs/PORTING_GUIDE.md`](docs/PORTING_GUIDE.md)** (中文)
> or **[`docs/PORTING_GUIDE_EN.md`](docs/PORTING_GUIDE_EN.md)** (English)
> for the full porting decision log + an AI agent slash command
> (`/port-sru-to-ros`) that regenerates this package from the upstream
> SRU repos and the paper, plus an automated `scripts/verify_port.sh`
> acceptance script.

It is a near-1:1 port of
`<upstream-package-path>/sru-robot-deployment/rl_nav_controller/rl_nav_controller/rl_nav_controller.py`
with the following substitutions:

| Original (ROS2, B2W + ZedX + DLIO)                          | This port (ROS1, Go2 + Odin1)             |
| ----------------------------------------------------------- | ----------------------------------------- |
| `/zed/zed_node/depth/depth_registered` (sensor_msgs/Image) | `/odin1/depth_img_competetion`            |
| `/dlio/odom_node/odom` (nav_msgs/Odometry)                  | `/odin1/odometry_highfreq` (~400 Hz)      |
| `/path_manager/path_manager_ros/nav_vel` (Twist)            | `/cmd_vel`                                |
| `rclpy` / Jazzy                                              | `rospy` / Noetic                          |
| `POLICY_SCALE = [1.5, 1.0, 1.0]` (wheeled B2W)              | `POLICY_SCALE = [0.6, 0.3, 0.6]` (Go2)    |

Inference is unchanged: ONNX VAE depth encoder (output 2560-dim latent) +
LSTM SRU policy (hidden 512, h/c carried as explicit ONNX inputs/outputs).

## Layout

```
sru_nav_go2_ros1/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── README.md
├── config/sru_nav.yaml             # all runtime parameters
├── launch/sru_nav_go2.launch       # joy + static TF + main node
├── models/                         # symlinks to deployment_policies/
│   ├── vae_encoder.onnx -> ../../sru-robot-deployment/.../vae_encoder.onnx
│   └── nav_policy.onnx  -> ../../sru-robot-deployment/.../nav_policy.onnx
├── scripts/sru_nav_node            # executable rospy entry point
└── src/sru_nav_go2/                # importable Python package
    ├── constants.py                # tunables (Go2-conservative defaults)
    ├── utils.py                    # quaternion/transform helpers (pure numpy)
    ├── model.py                    # ONNX runtime wrapper (LearningModel)
    ├── visualization.py            # rviz Marker helpers
    ├── waypoint_manager.py
    └── navigation_policy_node.py   # ported NavigationPolicyNode (rospy)
```

## Build

Place this directory inside a catkin workspace and build:

```bash
mkdir -p ~/sru_ws/src
ln -s /home/lfd/project/SRU_Navigation/sru_nav_go2_ros1 ~/sru_ws/src/
cd ~/sru_ws
catkin_make            # or catkin build
source devel/setup.bash
```

## Python dependencies (on Go2's NX, ROS Noetic / Python 3.8+)

```bash
pip install numpy scipy opencv-python onnxruntime    # or onnxruntime-gpu
```

For Jetson with CUDA, install the JetPack-matched `onnxruntime-gpu` wheel from
[NVIDIA's Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime).

## Run

```bash
# 1) Bring up Odin1 (publishes /odin1/depth_img_competetion and /odin1/odometry)
roslaunch odin_ros_driver odin1_ros.launch
#    Make sure config/control_command.yaml has  senddepth: 1  and  sendodom: 1.

# 2) Bring up your Go2 cmd_vel bridge (subscribes to /cmd_vel, drives sport mode)
#    (Your existing setup.)

# 3) Launch the SRU navigation node + joy
roslaunch sru_nav_go2_ros1 sru_nav_go2.launch

# 4) Send a goal (in odom frame)
rostopic pub /goal_pose geometry_msgs/PoseStamped "{
  header: {frame_id: 'odom'},
  pose:   {position: {x: 3.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}
}" --once
```

## Tuning knobs (in `config/sru_nav.yaml`)

- `policy_scale` — start at `[0.6, 0.3, 0.6]`. Increase gradually after
  verifying tracking and safety on Go2.
- `min_depth` / `max_depth` — depth clipping (meters). Default 0.25–10.0.
- `control_frequency` — keep at 5 Hz (matches training).

## Static TF (camera mounting)

Override in the launch line, e.g.:

```bash
roslaunch sru_nav_go2_ros1 sru_nav_go2.launch \
    odin1_x:=0.30 odin1_z:=0.22 odin1_pitch:=0.30
```

This must reflect the **real** Odin1 mounting on Go2 (forward distance,
height, downward pitch) for correct depth-to-body alignment.

## Safety

The joystick acts as a deadman switch. The robot will not move until you
push the throttle axis (axis 4, default scale `1.0 + axis[4]`) above 0.
After `JOYSTICK_TIMEOUT` (15 s) without a joy message the controller
clamps the cmd_vel ratio to zero.

## Known caveats vs. the original deployment

1. Odin1's dense depth (`depth_img_competetion`) is computed host-side and is
   labelled "high computing power required" in the driver README. On Jetson
   Xavier NX this may bottleneck before the policy itself does — verify with
   `rostopic hz /odin1/depth_img_competetion` first.
2. Odom twist frame: original code calls `convert_vel_frame` when
   `use_sim=False`, assuming world-frame twist. If Odin1 publishes twist
   already in `child_frame_id` (`odin1_base_link`) this would double-rotate.
   Verify on real hardware; if so, set `use_sim: true` in the YAML.
3. The pretrained policy was trained on B2W kinematics. Zero-shot transfer
   to Go2 is unverified; treat the first runs as evaluation, not deployment.
