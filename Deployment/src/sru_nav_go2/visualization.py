"""Visualization utilities for the SRU navigation controller (ROS1 port)."""

import numpy as np
import rospy
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from sru_nav_go2 import constants
from sru_nav_go2.utils import transform_points, yaw_quat


def _stamp_from_sec(sec):
    """Build a rospy.Time stamp from a float seconds value."""
    return rospy.Time.from_sec(float(sec))


class VisualizationManager:
    """Manages visualization markers for the navigation controller."""

    def __init__(self, node=None):
        # `node` retained for API parity with the original class. Not used in ROS1.
        self.node = node

    def publish_twist_marker(self, twist_msg, robot_odom_time, robot_frame_id, publisher):
        if robot_frame_id is None:
            return

        marker = Marker()
        marker.header = Header()
        marker.header.stamp = _stamp_from_sec(robot_odom_time)
        marker.header.frame_id = robot_frame_id

        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = constants.TWIST_MARKER_ID

        start_point = Point(x=0.0, y=0.0, z=0.0)
        end_point = Point(
            x=twist_msg.linear.x * constants.TWIST_MARKER_SCALE,
            y=twist_msg.linear.y * constants.TWIST_MARKER_SCALE,
            z=0.0,
        )
        marker.points = [start_point, end_point]

        marker.scale = Vector3(x=0.2, y=0.4, z=0.4)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        marker.pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        marker.frame_locked = False

        publisher.publish(marker)

    def publish_target_vector_marker(self, target_vec_b, robot_odom_time, robot_frame_id, publisher):
        if robot_frame_id is None:
            return

        marker = Marker()
        marker.header = Header()
        marker.header.stamp = _stamp_from_sec(robot_odom_time)
        marker.header.frame_id = robot_frame_id

        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = constants.TARGET_VECTOR_MARKER_ID

        vec = target_vec_b.flatten()

        start_point = Point(x=0.0, y=0.0, z=0.0)
        end_point = Point(x=float(vec[0]), y=float(vec[1]), z=float(vec[2]))
        marker.points = [start_point, end_point]

        marker.scale = Vector3(x=0.1, y=0.2, z=0.2)
        marker.pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        marker.frame_locked = False

        publisher.publish(marker)

    def publish_moving_goal_marker(self, moving_goal_delta, robot_pos_w, robot_orientation_w,
                                   robot_odom_time, map_frame_id, publisher):
        if robot_pos_w is None or map_frame_id is None:
            return

        robot_pos = np.array(robot_pos_w, dtype=np.float32)
        robot_ori = np.array(robot_orientation_w, dtype=np.float32)
        robot_yaw_ori = yaw_quat(robot_ori)
        moving_goal_pos = np.array(moving_goal_delta, dtype=np.float32)

        moving_goal_pos_w = transform_points(
            moving_goal_pos[np.newaxis],
            robot_pos[np.newaxis],
            robot_yaw_ori[np.newaxis],
        )

        marker = Marker()
        marker.header = Header()
        marker.header.stamp = _stamp_from_sec(robot_odom_time)
        marker.header.frame_id = map_frame_id

        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = constants.MOVING_GOAL_MARKER_ID

        start_point = Point(
            x=float(moving_goal_pos_w[0, 0]),
            y=float(moving_goal_pos_w[0, 1]),
            z=float(moving_goal_pos_w[0, 2]) + 1.0,
        )
        end_point = Point(
            x=float(moving_goal_pos_w[0, 0]),
            y=float(moving_goal_pos_w[0, 1]),
            z=float(moving_goal_pos_w[0, 2]),
        )
        marker.points = [start_point, end_point]

        marker.scale = Vector3(x=0.3, y=0.6, z=0.6)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        marker.pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        marker.frame_locked = False

        publisher.publish(marker)

    def publish_waypoints_marker(self, waypoints_visualization, map_frame_id, publisher):
        """Publish recorded waypoints as a CUBE_LIST marker.

        Note: ROS2 version received a `clock`. In ROS1 we use rospy.Time.now().
        """
        if map_frame_id is None:
            return

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = map_frame_id
        marker.ns = 'recorded_waypoints'
        marker.id = constants.WAYPOINTS_MARKER_ID
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale = Vector3(x=0.3, y=0.3, z=0.3)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.points = []
        for waypoint in waypoints_visualization:
            marker.points.append(Point(x=waypoint[0], y=waypoint[1], z=waypoint[2]))

        publisher.publish(marker)
