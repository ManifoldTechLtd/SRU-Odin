"""Waypoint management for the SRU navigation controller (ROS1 port).

This module is logger-agnostic: pass any object with .info/.warn methods.
For rospy a thin wrapper around rospy.loginfo / rospy.logwarn is provided.
"""

from collections import deque


class RospyLogger:
    """Minimal logger duck-type matching the original rclpy logger interface."""

    def __init__(self):
        import rospy
        self._rospy = rospy

    def info(self, msg):
        self._rospy.loginfo(msg)

    def warn(self, msg):
        self._rospy.logwarn(msg)

    # Keep ROS2-style alias used in original code
    def warning(self, msg):
        self._rospy.logwarn(msg)

    def error(self, msg):
        self._rospy.logerr(msg)


class WaypointManager:
    """Manages waypoint recording, tracking, and playback."""

    def __init__(self, logger):
        self.logger = logger
        self.recorded_waypoints = deque()
        self.inversed_ordered_waypoints = deque()
        self.waypoints_visualization = deque()
        self.is_publish_waypoints_home = False
        self.is_publish_waypoints_inversed = False

    def record_waypoint(self, robot_pos_w):
        if robot_pos_w is None:
            self.logger.warn('Robot position is not available yet.')
            return False

        if len(self.recorded_waypoints) == 0 and len(self.inversed_ordered_waypoints) > 0:
            self.inversed_ordered_waypoints.clear()
            self.waypoints_visualization.clear()
            self.logger.info('Clear the inversed waypoints, and visualization waypoints.')

        self.recorded_waypoints.append(robot_pos_w)
        self.inversed_ordered_waypoints.appendleft(robot_pos_w)
        self.waypoints_visualization.append(robot_pos_w)
        self.logger.info(
            'Waypoint recorded: {}, total waypoints: {}'.format(
                robot_pos_w, len(self.recorded_waypoints)
            )
        )
        return True

    def remove_last_waypoint(self):
        removed_count = 0

        if len(self.recorded_waypoints) > 0:
            self.recorded_waypoints.pop()
            removed_count += 1
            self.logger.info(
                'Removed last waypoint (home), total: {}'.format(len(self.recorded_waypoints))
            )

        if len(self.inversed_ordered_waypoints) > 0:
            self.inversed_ordered_waypoints.popleft()
            removed_count += 1
            self.logger.info(
                'Removed last waypoint (inversed), total: {}'.format(
                    len(self.inversed_ordered_waypoints)
                )
            )

        if len(self.waypoints_visualization) > 0:
            self.waypoints_visualization.pop()
            removed_count += 1
            self.logger.info(
                'Removed last waypoint (visualization), total: {}'.format(
                    len(self.waypoints_visualization)
                )
            )

        if (len(self.waypoints_visualization) < len(self.recorded_waypoints) or
                len(self.waypoints_visualization) < len(self.inversed_ordered_waypoints)):
            raise ValueError(
                'The visualization waypoints is smaller than the recorded waypoints.'
            )

        if removed_count == 0:
            self.logger.warn('No waypoints to remove.')

        return removed_count > 0

    def get_next_waypoint_home(self):
        if self.recorded_waypoints:
            return self.recorded_waypoints.pop()
        return None

    def get_next_waypoint_inversed(self):
        if self.inversed_ordered_waypoints:
            return self.inversed_ordered_waypoints.pop()
        return None

    def start_home_waypoint_sequence(self):
        if len(self.recorded_waypoints) > 0:
            self.is_publish_waypoints_home = True
            return True
        return False

    def start_inversed_waypoint_sequence(self):
        if len(self.inversed_ordered_waypoints) > 0:
            self.is_publish_waypoints_inversed = True
            return True
        return False

    def stop_home_waypoint_sequence(self):
        self.is_publish_waypoints_home = False

    def stop_inversed_waypoint_sequence(self):
        self.is_publish_waypoints_inversed = False

    def reset_visualization_if_complete(self):
        if (not self.recorded_waypoints and not self.inversed_ordered_waypoints
                and self.waypoints_visualization):
            self.logger.info('All waypoints are published, reset visualization.')
            self.waypoints_visualization.clear()
            return True
        return False

    def re_add_aborted_waypoint(self, target_pos_w):
        if self.is_publish_waypoints_home:
            self.recorded_waypoints.append(target_pos_w)
            self.logger.info('Goal aborted - re-adding current goal as a recorded (home) waypoint.')
        if self.is_publish_waypoints_inversed:
            self.inversed_ordered_waypoints.appendleft(target_pos_w)
            self.logger.info(
                'Goal aborted: reinserting the current goal as recorded (inversed) waypoint.'
            )

    def has_home_waypoints(self):
        return len(self.recorded_waypoints) > 0

    def has_inversed_waypoints(self):
        return len(self.inversed_ordered_waypoints) > 0

    def is_home_sequence_active(self):
        return self.is_publish_waypoints_home

    def is_inversed_sequence_active(self):
        return self.is_publish_waypoints_inversed
