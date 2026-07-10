"""ROS1 (rospy) navigation policy node for Unitree Go2 + Odin1.

Direct port of the original ROS2 NavigationPolicyNode. Behaviour is preserved;
only ROS API calls, time helpers, and topic defaults have been changed.
"""

import time

import cv2  # noqa: F401  (kept for parity; cv2 import inside model.py)
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from sru_nav_go2 import constants
from sru_nav_go2.model import LearningModel
from sru_nav_go2.utils import transform_points, yaw_quat
from sru_nav_go2.visualization import VisualizationManager
from sru_nav_go2.waypoint_manager import RospyLogger, WaypointManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stamp_to_sec(stamp):
    """ROS1 Time -> float seconds."""
    return stamp.secs + stamp.nsecs * 1e-9


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class NavigationPolicyNode(object):
    """RL navigation controller node (ROS1)."""

    def __init__(self,
                 preprocess_model_path,
                 policy_model_path,
                 depth_topic,
                 odom_topic,
                 joy_topic,
                 goal_topic,
                 cmd_vel_topic,
                 min_depth=constants.DEFAULT_MIN_DEPTH,
                 max_depth=constants.DEFAULT_MAX_DEPTH,
                 control_frequency=constants.DEFAULT_CONTROL_FREQUENCY,
                 policy_scale=None,
                 use_sim=False,
                 require_joystick=True):
        # Configuration
        self.use_sim = use_sim
        self.require_joystick = bool(require_joystick)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.control_frequency = float(control_frequency)
        self.odom_ready = False
        self.arrive_goal_threshold = constants.ARRIVE_GOAL_THRESHOLD
        self.last_run_time = 0.0
        self.system_delay = constants.JOYSTICK_TIMEOUT

        # ----- Publishers ------------------------------------------------
        self.cmd_vel_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        self.base_vel_publisher = rospy.Publisher(
            '~base_vel', Twist, queue_size=10)
        self.goal_pose_publisher = rospy.Publisher(goal_topic, PoseStamped, queue_size=1)

        # Visualization
        self.twist_marker_publisher = rospy.Publisher(
            '~vis/twist_cmd_marker', Marker, queue_size=10)
        self.goal_vector_marker_publisher = rospy.Publisher(
            '~vis/goal_vector_marker', Marker, queue_size=10)
        self.moving_goal_marker_publisher = rospy.Publisher(
            '~vis/moving_goal_marker', Marker, queue_size=10)
        self.recorded_waypoints_marker_pub = rospy.Publisher(
            '~vis/recorded_waypoints_marker', Marker, queue_size=10)

        # ----- Utilities --------------------------------------------------
        self.bridge = CvBridge()
        self.model = LearningModel(
            preprocess_model_path=preprocess_model_path,
            policy_model_path=policy_model_path,
            policy_scale=policy_scale,
        )
        self._logger = RospyLogger()
        self.waypoint_manager = WaypointManager(self._logger)
        self.visualization_manager = VisualizationManager()

        # ----- Robot state -----------------------------------------------
        self.map_frame_id = None
        self.robot_frame_id = None
        self.robot_odom_time = 0.0
        self.robot_pos_w = None
        self.robot_orientation_w = None
        self.linear_vel_w = None
        self.angular_vel_w = None
        self.linear_vel = None
        self.angular_vel = None
        self.gravity_vector = None
        self.depth_image = None

        # ----- Navigation state ------------------------------------------
        self.target_pos_w = None
        self.last_target_pos = None
        self.is_reset_hidden_state = False
        self.last_action = self._reset_last_action()
        self.prev_cmd = np.zeros(3)
        self.is_abort_goal = False

        # ----- Joystick state --------------------------------------------
        self.joy_linear_x = 0.0
        self.joy_linear_y = 0.0
        self.joy_angular_z = 0.0
        # When require_joystick is False we want the robot to move at full
        # policy scale even with no /joy publisher (testing / headless runs).
        # When True (default, safe), ratio stays 0 until joy_callback fires.
        self.cmd_vel_ratio = 0.0 if self.require_joystick else 1.0
        self.joy_time = time.time()
        self.last_trigger_time = time.time()

        # ----- Moving goal state -----------------------------------------
        self.moving_goal_delta = [0.0, 0.0, 0.0]

        # ----- Smart joystick state --------------------------------------
        self.smart_joystick_goal = [0.0, 0.0, 0.0]
        self.prev_smart_joystick_goal = [0.0, 0.0, 0.0]
        self.smart_joystick_mode_active = False
        self.smart_joystick_goal_aborted = False
        self.latest_joystick_axes = [0.0, 0.0, 0.0]

        # ----- Subscribers (created LAST so callbacks fire only after init)
        self.odom_subscriber = rospy.Subscriber(
            odom_topic, Odometry, self.odom_callback, queue_size=10)
        self.depth_subscriber = rospy.Subscriber(
            depth_topic, Image, self.depth_callback, queue_size=2)
        self.joy_subscriber = rospy.Subscriber(
            joy_topic, Joy, self.joy_callback, queue_size=10)
        self.target_position_subscriber = rospy.Subscriber(
            goal_topic, PoseStamped, self.target_position_callback, queue_size=1)

        # ----- Timers ----------------------------------------------------
        rospy.Timer(rospy.Duration(constants.WAYPOINT_PUBLISH_INTERVAL),
                    self._timer_publish_recorded_waypoints)
        rospy.Timer(rospy.Duration(constants.TARGET_VECTOR_PUBLISH_INTERVAL),
                    self._timer_publish_target_vector)
        smart_joystick_interval = 1.0 / constants.SMART_JOYSTICK_UPDATE_FREQUENCY
        rospy.Timer(rospy.Duration(smart_joystick_interval),
                    self._timer_update_smart_joystick_goal)

        rospy.loginfo('\033[92mNavigation policy node is ready.\033[0m')

    # =====================================================================
    # Callbacks
    # =====================================================================
    def odom_callback(self, odom_msg):
        # 1) odom timestamp
        self.robot_odom_time = _stamp_to_sec(odom_msg.header.stamp)

        # 2) Frame IDs (once)
        if self.map_frame_id is None:
            self.map_frame_id = odom_msg.header.frame_id
        if self.robot_frame_id is None:
            self.robot_frame_id = odom_msg.child_frame_id

        # 3) Robot pose
        self.robot_pos_w = [
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z,
        ]
        self.robot_orientation_w = [
            odom_msg.pose.pose.orientation.w,
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
        ]

        # 4) Velocities from odom
        self.linear_vel_w = [
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.y,
            odom_msg.twist.twist.linear.z,
        ]
        self.angular_vel_w = [
            odom_msg.twist.twist.angular.x,
            odom_msg.twist.twist.angular.y,
            odom_msg.twist.twist.angular.z,
        ]

        # 5) Convert to base frame (real-hw) or pass-through (sim)
        if self.use_sim:
            self.linear_vel = list(self.linear_vel_w)
            self.angular_vel = list(self.angular_vel_w)
        else:
            self.linear_vel = self.convert_vel_frame(
                self.linear_vel_w, self.robot_orientation_w)
            self.angular_vel = self.convert_vel_frame(
                self.angular_vel_w, self.robot_orientation_w)

        # 6) Publish converted base velocity for debug
        self.publish_base_vel(self.linear_vel, self.angular_vel)

        # 7) Projected gravity in base frame
        self.gravity_vector = self.projected_gravity_vector(self.robot_orientation_w)

        # 8) Mark odom available
        if not self.odom_ready:
            self.odom_ready = True

    def depth_callback(self, depth_msg):
        if not self.odom_ready:
            rospy.logwarn_throttle(
                5.0, '\033[93mOdometry not ready, skipping depth callback.\033[0m')
            return

        try:
            depth_array = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth_array = np.nan_to_num(
                depth_array, nan=0.0, posinf=self.max_depth * 2.0, neginf=0.0)
            depth_array = depth_array.astype(np.float32, copy=False)
            depth_array[depth_array > self.max_depth] = 0.0
            depth_array[depth_array < self.min_depth] = 0.0
            self.depth_image = depth_array
        except Exception as e:
            rospy.logerr('Error converting depth image: {}'.format(e))
            return

        # Respect control frequency
        interval = 1.0 / self.control_frequency
        if (self.robot_odom_time - self.last_run_time) < interval:
            return
        self.last_run_time = self.robot_odom_time

        self.generate_cmd_vel()

    def generate_cmd_vel(self):
        # Visualize moving goal
        self.visualization_manager.publish_moving_goal_marker(
            self.moving_goal_delta, self.robot_pos_w, self.robot_orientation_w,
            self.robot_odom_time, self.map_frame_id, self.moving_goal_marker_publisher,
        )

        if self.smart_joystick_mode_active:
            rospy.loginfo_throttle(
                1.0,
                'Smart joystick mode active - Goal: [{:.2f}, {:.2f}, {:.2f}]'.format(
                    self.smart_joystick_goal[0], self.smart_joystick_goal[1],
                    self.smart_joystick_goal[2],
                ),
            )

        is_arrived = self._check_goal_reached(self.target_pos_w, self.robot_pos_w)
        if is_arrived or self.is_abort_goal:
            twist = Twist()
            twist.linear.x = 0.0 + self.joy_linear_x
            twist.linear.y = 0.0 + self.joy_linear_y
            twist.angular.z = 0.0 + self.joy_angular_z
            self.cmd_vel_publisher.publish(twist)

            if self.is_abort_goal:
                self.waypoint_manager.re_add_aborted_waypoint(self.target_pos_w)

            self.target_pos_w = None
            self.is_abort_goal = False
            self.last_action = self._reset_last_action()
            rospy.loginfo('Target position reset.')

        else:
            # Joystick deadman timeout (only when joystick is required)
            if self.require_joystick and \
                    (self.robot_odom_time - self.joy_time > self.system_delay):
                self.cmd_vel_ratio = 0.0
                diff = self.robot_odom_time - self.joy_time
                rospy.logwarn_throttle(
                    2.0,
                    '\033[93mJoystick timeout, stopping the robot. Time diff: {:.4f}s\033[0m'
                    .format(diff),
                )

            rospy.loginfo_throttle(
                1.0,
                'cmd_vel_ratio: {:.4f} (require_joystick={})'.format(
                    self.cmd_vel_ratio, self.require_joystick))

            cmd, action, _target_vec_b = self.model.predict(
                self.linear_vel, self.angular_vel, self.gravity_vector,
                self.last_action, self.target_pos_w, self.robot_pos_w,
                self.robot_orientation_w, self.depth_image,
                self.is_reset_hidden_state,
            )

            if self.is_reset_hidden_state:
                rospy.logwarn('\033[93mResetting hidden state.\033[0m')
                self.is_reset_hidden_state = False
                self.last_action = self._reset_last_action()

            self.last_action = action.tolist()

            twist = Twist()
            model_cmd = np.array([
                cmd[0].item() * self.cmd_vel_ratio,
                cmd[1].item() * self.cmd_vel_ratio * constants.LATERAL_VELOCITY_SCALE,
                cmd[2].item() * self.cmd_vel_ratio,
            ])
            filter_coef = np.array(constants.LOW_PASS_FILTER_COEF)
            filt_model = filter_coef * model_cmd + (1 - filter_coef) * self.prev_cmd
            twist.linear.x = float(filt_model[0]) + self.joy_linear_x
            twist.linear.y = float(filt_model[1]) + self.joy_linear_y
            twist.angular.z = float(filt_model[2]) + self.joy_angular_z
            self.prev_cmd = filt_model
            self.cmd_vel_publisher.publish(twist)

            self.visualization_manager.publish_twist_marker(
                twist, self.robot_odom_time, self.robot_frame_id,
                self.twist_marker_publisher,
            )

        rospy.loginfo_throttle(
            1.0,
            'Published cmd_vel: linear_x={:.4f}, linear_y={:.4f}, angular_z={:.4f}'.format(
                twist.linear.x, twist.linear.y, twist.angular.z
            ),
        )
        self._reset_joystick()

    # =====================================================================
    # Velocity / orientation helpers
    # =====================================================================
    def publish_base_vel(self, linear_vel, angular_vel):
        twist = Twist()
        twist.linear.x = linear_vel[0]
        twist.linear.y = linear_vel[1]
        twist.angular.z = angular_vel[2]
        self.base_vel_publisher.publish(twist)

    @staticmethod
    def convert_vel_frame(vel_vec, orientation_w):
        # quaternion is (w, x, y, z); scipy expects (x, y, z, w)
        quat_xyzw = [orientation_w[1], orientation_w[2], orientation_w[3], orientation_w[0]]
        rotation = R.from_quat(quat_xyzw)
        inv_rot = rotation.inv()
        return inv_rot.apply(np.array(vel_vec)).tolist()

    @staticmethod
    def projected_gravity_vector(robot_orientation_w):
        quat_xyzw = [robot_orientation_w[1], robot_orientation_w[2],
                     robot_orientation_w[3], robot_orientation_w[0]]
        rotation = R.from_quat(quat_xyzw)
        inv_rot = rotation.inv()
        gravity = np.array([0.0, 0.0, -constants.GRAVITY_MAGNITUDE])
        proj = inv_rot.apply(gravity)
        proj = proj / (np.linalg.norm(proj) + 1e-6)
        return proj.tolist()

    # =====================================================================
    # Joystick handling
    # =====================================================================
    def joy_callback(self, joy_msg):
        if len(joy_msg.axes) <= max(
                constants.JOYSTICK_AXIS_LINEAR_X,
                constants.JOYSTICK_AXIS_LINEAR_Y,
                constants.JOYSTICK_AXIS_LINEAR_Z,
                constants.JOYSTICK_AXIS_ANGULAR_Z,
                constants.JOYSTICK_AXIS_SMART,
                4):
            rospy.logwarn_throttle(5.0, 'Joy message has too few axes; ignoring.')
            return

        self.cmd_vel_ratio = (1.0 + joy_msg.axes[4]) * 1.0

        if (len(joy_msg.buttons) > constants.BUTTON_ABORT and
                joy_msg.buttons[constants.BUTTON_ABORT] == 1):
            self.is_abort_goal = True
            rospy.logwarn('Abort goal')

        dx, dy, dz = self._moving_xyz_with_buttons(joy_msg, scale=constants.MOVING_SCALE)
        self.moving_goal_delta[0] += dx
        self.moving_goal_delta[1] += dy
        self.moving_goal_delta[2] += dz

        current_time = time.time()
        cooldown = constants.TRIGGER_BUTTON_COOLDOWN

        if (not self.is_abort_goal
                and (current_time - self.last_trigger_time) > cooldown
                and len(joy_msg.buttons) > constants.BUTTON_SEND_GOAL
                and joy_msg.buttons[constants.BUTTON_SEND_GOAL] == 1):
            rospy.logwarn('Trigger moving goal')
            self._publish_moving_goal()
            self.last_trigger_time = current_time

        if ((current_time - self.last_trigger_time) > cooldown
                and len(joy_msg.buttons) > constants.BUTTON_RECORD_WAYPOINT
                and joy_msg.buttons[constants.BUTTON_RECORD_WAYPOINT] == 1):
            self.waypoint_manager.record_waypoint(self.robot_pos_w)
            self.last_trigger_time = current_time

        if ((current_time - self.last_trigger_time) > cooldown
                and len(joy_msg.buttons) > constants.BUTTON_CLEAR_WAYPOINT
                and joy_msg.buttons[constants.BUTTON_CLEAR_WAYPOINT] == 1):
            self.waypoint_manager.remove_last_waypoint()
            self.last_trigger_time = current_time

        if (not self.is_abort_goal
                and (current_time - self.last_trigger_time) > cooldown
                and len(joy_msg.buttons) > constants.BUTTON_TRIGGER_WAYPOINTS
                and joy_msg.buttons[constants.BUTTON_TRIGGER_WAYPOINTS] == 1):
            rospy.logwarn('Trigger waypoints')
            if self.waypoint_manager.has_home_waypoints():
                self.waypoint_manager.start_home_waypoint_sequence()
            elif self.waypoint_manager.has_inversed_waypoints():
                self.waypoint_manager.start_inversed_waypoint_sequence()
            self.last_trigger_time = current_time

        if joy_msg.axes[constants.JOYSTICK_AXIS_SMART] < -0.5:
            if not self.smart_joystick_mode_active:
                self.smart_joystick_mode_active = True
                self.smart_joystick_goal_aborted = False

            if not self.smart_joystick_goal_aborted and self.target_pos_w is not None:
                self.is_abort_goal = True
                self.smart_joystick_goal_aborted = True
                rospy.loginfo('Smart joystick mode: aborting current navigation goal')

            self.latest_joystick_axes = [
                joy_msg.axes[constants.JOYSTICK_AXIS_LINEAR_X],
                joy_msg.axes[constants.JOYSTICK_AXIS_LINEAR_Y],
                joy_msg.axes[constants.JOYSTICK_AXIS_LINEAR_Z],
            ]
        else:
            if self.smart_joystick_mode_active:
                self.smart_joystick_mode_active = False
                self.is_abort_goal = True
                rospy.loginfo('Exiting smart joystick mode: aborting current navigation goal')

            self._reset_smart_joystick_goal()
            self.joy_linear_x = (
                joy_msg.axes[constants.JOYSTICK_AXIS_LINEAR_X]
                * constants.LINEAR_SCALE * 1.5)
            self.joy_linear_y = (
                joy_msg.axes[constants.JOYSTICK_AXIS_LINEAR_Y]
                * constants.LINEAR_SCALE)
            self.joy_angular_z = (
                joy_msg.axes[constants.JOYSTICK_AXIS_ANGULAR_Z]
                * constants.ANGULAR_SCALE)

        if (len(joy_msg.buttons) > constants.BUTTON_RESET_HIDDEN_STATE and
                joy_msg.buttons[constants.BUTTON_RESET_HIDDEN_STATE] == 1):
            self.is_reset_hidden_state = True
            rospy.logwarn('Force Reset hidden state')

        self.joy_time = time.time()

    def _generate_waypoint_using_joystick(self, linear_x, linear_y, linear_z):
        if self.robot_pos_w is None or self.robot_orientation_w is None:
            return

        goal_offset_robot = np.array([
            linear_x * constants.SMART_JOYSTICK_SCALE,
            linear_y * constants.SMART_JOYSTICK_SCALE,
            linear_z * constants.SMART_JOYSTICK_SCALE * constants.SMART_JOYSTICK_Z_SCALE,
        ], dtype=np.float32)

        robot_ori = np.array(self.robot_orientation_w, dtype=np.float32)
        robot_yaw_ori = yaw_quat(robot_ori)

        goal_offset_world = transform_points(
            goal_offset_robot[np.newaxis],
            np.zeros(3, dtype=np.float32)[np.newaxis],
            robot_yaw_ori[np.newaxis],
        )[0]

        target_goal_world = [
            self.robot_pos_w[0] + goal_offset_world[0].item(),
            self.robot_pos_w[1] + goal_offset_world[1].item(),
            self.robot_pos_w[2] + goal_offset_world[2].item(),
        ]

        if self.prev_smart_joystick_goal == [0.0, 0.0, 0.0]:
            self.smart_joystick_goal = target_goal_world
        else:
            alpha = constants.SMART_JOYSTICK_FILTER_ALPHA
            self.smart_joystick_goal = [
                alpha * target_goal_world[0] + (1 - alpha) * self.prev_smart_joystick_goal[0],
                alpha * target_goal_world[1] + (1 - alpha) * self.prev_smart_joystick_goal[1],
                alpha * target_goal_world[2] + (1 - alpha) * self.prev_smart_joystick_goal[2],
            ]
        self.prev_smart_joystick_goal = list(self.smart_joystick_goal)

    def _publish_smart_joystick_goal(self):
        if self.robot_pos_w is None or self.robot_orientation_w is None:
            rospy.logwarn_throttle(
                5.0, 'Cannot publish smart joystick goal: robot pose not available')
            return
        self._publish_goal(self.smart_joystick_goal)

    def _reset_smart_joystick_goal(self):
        self.smart_joystick_goal = [0.0, 0.0, 0.0]
        self.prev_smart_joystick_goal = [0.0, 0.0, 0.0]

    @staticmethod
    def _moving_xyz_with_buttons(joy_msg, scale=1.0):
        def b(idx):
            return bool(joy_msg.buttons[idx]) if idx < len(joy_msg.buttons) else False

        dx = scale * (b(constants.BUTTON_FORWARD) - b(constants.BUTTON_BACKWARD))
        dy = scale * (b(constants.BUTTON_LEFT) - b(constants.BUTTON_RIGHT))
        dz = (scale / 5.0) * (b(constants.BUTTON_UP) - b(constants.BUTTON_DOWN))
        return dx, dy, dz

    # =====================================================================
    # Goal publishing
    # =====================================================================
    def _publish_goal(self, goal_pos):
        goal_pose = PoseStamped()
        goal_pose.header = Header()
        goal_pose.header.stamp = rospy.Time.from_sec(self.robot_odom_time)
        goal_pose.header.frame_id = self.map_frame_id if self.map_frame_id else 'odom'
        goal_pose.pose.position.x = float(goal_pos[0])
        goal_pose.pose.position.y = float(goal_pos[1])
        goal_pose.pose.position.z = float(goal_pos[2])
        goal_pose.pose.orientation.w = 1.0
        self.goal_pose_publisher.publish(goal_pose)

    def _publish_moving_goal(self):
        if self.robot_pos_w is None or self.robot_orientation_w is None:
            rospy.logwarn('Cannot publish moving goal: robot pose/orientation not yet set.')
            return

        robot_pos = np.array(self.robot_pos_w, dtype=np.float32)
        robot_ori = np.array(self.robot_orientation_w, dtype=np.float32)
        robot_yaw_ori = yaw_quat(robot_ori)

        moving_goal_pos = np.array(self.moving_goal_delta, dtype=np.float32)
        moving_goal_pos_w = transform_points(
            moving_goal_pos[np.newaxis],
            robot_pos[np.newaxis],
            robot_yaw_ori[np.newaxis],
        )

        self._publish_goal(moving_goal_pos_w[0].tolist())
        self._reset_moving_goal()

    # =====================================================================
    # Timer callbacks
    # =====================================================================
    def _timer_publish_recorded_waypoints(self, _event):
        self.publish_recorded_waypoints()

    def _timer_publish_target_vector(self, _event):
        self.publish_target_vector()

    def _timer_update_smart_joystick_goal(self, _event):
        self.update_smart_joystick_goal()

    def publish_recorded_waypoints(self):
        if self.robot_pos_w is None:
            return

        self.visualization_manager.publish_waypoints_marker(
            self.waypoint_manager.waypoints_visualization,
            self.map_frame_id,
            self.recorded_waypoints_marker_pub,
        )

        if (self.waypoint_manager.is_home_sequence_active()
                and self.waypoint_manager.has_home_waypoints()):
            if (self.target_pos_w is None
                    or self._check_near_goal(self.target_pos_w, self.robot_pos_w)):
                next_wp = self.waypoint_manager.get_next_waypoint_home()
                rospy.loginfo('Publishing next waypoint: {}'.format(next_wp))
                self._publish_goal(next_wp)
            else:
                rospy.loginfo_throttle(1.0, 'Tracking the current waypoint ...')

            if not self.waypoint_manager.has_home_waypoints():
                self.waypoint_manager.stop_home_waypoint_sequence()

        if (self.waypoint_manager.is_inversed_sequence_active()
                and self.waypoint_manager.has_inversed_waypoints()):
            if (self.target_pos_w is None
                    or self._check_near_goal(self.target_pos_w, self.robot_pos_w)):
                next_wp = self.waypoint_manager.get_next_waypoint_inversed()
                rospy.loginfo('Publishing next inversed waypoint: {}'.format(next_wp))
                self._publish_goal(next_wp)
            else:
                rospy.loginfo_throttle(1.0, 'Tracking the current waypoint ...')

            if not self.waypoint_manager.has_inversed_waypoints():
                self.waypoint_manager.stop_inversed_waypoint_sequence()

        self.waypoint_manager.reset_visualization_if_complete()

    def publish_target_vector(self):
        if self.robot_pos_w is None or self.robot_orientation_w is None:
            return
        tgt = self.target_pos_w if self.target_pos_w is not None else self.last_target_pos
        if tgt is None:
            return

        _, target_vec_b = self.model.normalize_target_position(
            tgt, self.robot_pos_w, self.robot_orientation_w
        )

        self.visualization_manager.publish_target_vector_marker(
            target_vec_b, self.robot_odom_time, self.robot_frame_id,
            self.goal_vector_marker_publisher,
        )

    def update_smart_joystick_goal(self):
        if not self.smart_joystick_mode_active:
            return
        self._generate_waypoint_using_joystick(
            self.latest_joystick_axes[0],
            self.latest_joystick_axes[1],
            self.latest_joystick_axes[2],
        )
        self._publish_smart_joystick_goal()

    # =====================================================================
    # Goal callbacks & checks
    # =====================================================================
    def _check_near_goal(self, target_pos_w, robot_pos_w):
        if target_pos_w is None or robot_pos_w is None:
            return True
        dist = np.linalg.norm(np.array(target_pos_w[:2]) - np.array(robot_pos_w[:2]))
        threshold = self.arrive_goal_threshold * constants.NEAR_GOAL_THRESHOLD_MULTIPLIER
        if dist > threshold:
            return False
        rospy.loginfo('Near the current goal position.')
        return True

    def _check_goal_reached(self, target_pos_w, robot_pos_w):
        if target_pos_w is None or robot_pos_w is None:
            return True
        dist = np.linalg.norm(np.array(target_pos_w[:2]) - np.array(robot_pos_w[:2]))
        if dist > self.arrive_goal_threshold:
            return False
        rospy.loginfo('Arrived at the goal position.')
        return True

    def target_position_callback(self, msg):
        # If the goal arrives before odom, skip frame check.
        if self.map_frame_id is not None and msg.header.frame_id and \
                msg.header.frame_id != self.map_frame_id:
            rospy.logerr(
                '\033[91mTarget frame_id "{}" does not match odometry frame_id "{}"\033[0m'
                .format(msg.header.frame_id, self.map_frame_id)
            )
            return

        goal_z = msg.pose.position.z
        if abs(goal_z) < 1e-3:
            if self.robot_pos_w is not None:
                goal_z = self.robot_pos_w[2]
            else:
                rospy.logwarn('Robot position not available, using received z for target.')

        self.target_pos_w = [msg.pose.position.x, msg.pose.position.y, goal_z]
        self.last_target_pos = list(self.target_pos_w)

        rospy.loginfo('Received target position: {}'.format(self.target_pos_w))

    # =====================================================================
    # Misc reset helpers
    # =====================================================================
    def _reset_last_action(self):
        self.prev_cmd = np.zeros(3)
        return [0.0, 0.0, 0.0]

    def _reset_joystick(self):
        self.joy_linear_x = 0.0
        self.joy_linear_y = 0.0
        self.joy_angular_z = 0.0

    def _reset_moving_goal(self):
        self.moving_goal_delta = [0.0, 0.0, 0.0]
        rospy.loginfo('Reset moving goal delta.')
