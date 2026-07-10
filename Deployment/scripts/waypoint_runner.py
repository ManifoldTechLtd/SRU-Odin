#!/usr/bin/env python3
"""Sequential waypoint runner for sru_nav_go2_ros1.

Publishes a list of goals to /goal_pose one at a time, monitors the robot
odometry to decide when each goal is reached (or has timed out, or the robot
is stuck), then moves on to the next. After the last waypoint, optionally
returns to the pose captured at script start.

Does NOT modify any code in sru_nav_go2_ros1. Works purely as an external
client on top of the existing /goal_pose + /odin1/odometry_highfreq contract.

Usage
-----
    # YAML file
    rosrun sru_nav_go2_ros1 waypoint_runner.py --file waypoints.yaml

    # Command line (x,y[,z] tuples separated by spaces)
    rosrun sru_nav_go2_ros1 waypoint_runner.py \
        --waypoints "3,0 3,2 0,2" --return-home

YAML format
-----------
    frame_id: odom
    arrive_threshold: 0.75      # meters, XY distance to consider 'arrived'
    settle_time: 1.5            # seconds the robot must stay within threshold
    per_goal_timeout: 60.0      # seconds before giving up on a goal
    stuck_window: 10.0          # seconds of trailing window for stuck check
    stuck_displacement: 0.10    # meters; below this in `stuck_window` -> skip
    republish_interval: 5.0     # seconds; re-publish the same goal periodically
    return_home: true
    waypoints:
      - [3.0, 0.0, 0.0]
      - [3.0, 2.0, 0.0]
      - [0.0, 2.0, 0.0]
"""

import argparse
import collections
import math
import os
import sys
import time

import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


# ---------------------------------------------------------------------------
# Defaults (mirror sru_nav_go2_ros1/constants.py where applicable)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    frame_id='odom',
    arrive_threshold=0.75,        # matches constants.ARRIVE_GOAL_THRESHOLD
    settle_time=1.5,
    per_goal_timeout=60.0,
    stuck_window=10.0,
    stuck_displacement=0.10,
    republish_interval=5.0,
    return_home=True,
    odom_topic='/odin1/odometry_highfreq',
    goal_topic='/goal_pose',
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_inline_waypoints(text):
    """Parse "x,y[,z] x,y[,z] ..." into list[list[float]]."""
    out = []
    if not text:
        return out
    for tok in text.split():
        parts = tok.split(',')
        if len(parts) not in (2, 3):
            raise ValueError(
                "Bad waypoint '{}': expected 'x,y' or 'x,y,z'".format(tok))
        xyz = [float(parts[0]), float(parts[1]),
               float(parts[2]) if len(parts) == 3 else 0.0]
        out.append(xyz)
    return out


def _load_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    if 'waypoints' not in data or not data['waypoints']:
        raise ValueError("YAML '{}' has no 'waypoints' list.".format(path))
    norm = []
    for wp in data['waypoints']:
        if len(wp) not in (2, 3):
            raise ValueError("Bad waypoint {} (need 2 or 3 floats).".format(wp))
        x = float(wp[0]); y = float(wp[1])
        z = float(wp[2]) if len(wp) == 3 else 0.0
        norm.append([x, y, z])
    data['waypoints'] = norm
    return data


def build_config(args):
    cfg = dict(DEFAULTS)
    if args.file:
        cfg.update({k: v for k, v in _load_yaml(args.file).items() if v is not None})
    # Command-line overrides win over YAML
    if args.waypoints:
        cfg['waypoints'] = _parse_inline_waypoints(args.waypoints)
    if args.frame_id is not None:
        cfg['frame_id'] = args.frame_id
    if args.timeout is not None:
        cfg['per_goal_timeout'] = float(args.timeout)
    if args.arrive_threshold is not None:
        cfg['arrive_threshold'] = float(args.arrive_threshold)
    if args.return_home is not None:
        cfg['return_home'] = args.return_home
    if 'waypoints' not in cfg or not cfg['waypoints']:
        raise ValueError("No waypoints provided (use --file or --waypoints).")
    return cfg


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class WaypointRunner(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self._pose = None              # latest (x, y, z, t)
        self._pose_history = collections.deque()   # entries: (t, x, y)
        self._home = None              # (x, y, z) captured at first odom

        self.goal_pub = rospy.Publisher(
            cfg['goal_topic'], PoseStamped, queue_size=1, latch=True)
        self.odom_sub = rospy.Subscriber(
            cfg['odom_topic'], Odometry, self._odom_cb, queue_size=20)

        rospy.loginfo("waypoint_runner: waiting for first odom on '%s' ...",
                      cfg['odom_topic'])
        self._wait_for_odom(timeout=30.0)
        self._home = (self._pose[0], self._pose[1], self._pose[2])
        rospy.loginfo("waypoint_runner: home pose captured at "
                      "x=%.3f y=%.3f z=%.3f", *self._home)

        # Give the latched publisher a moment so the first goal isn't lost.
        time.sleep(0.5)

    # --------------------------------------------------------------- callbacks
    def _odom_cb(self, msg):
        t = msg.header.stamp.to_sec() or rospy.get_time()
        p = msg.pose.pose.position
        self._pose = (p.x, p.y, p.z, t)
        self._pose_history.append((t, p.x, p.y))
        # Trim history to a bit more than stuck_window
        cutoff = t - max(self.cfg['stuck_window'] * 2.0, 5.0)
        while self._pose_history and self._pose_history[0][0] < cutoff:
            self._pose_history.popleft()

    def _wait_for_odom(self, timeout):
        t0 = time.time()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and self._pose is None:
            if time.time() - t0 > timeout:
                raise RuntimeError(
                    "No odometry received on '{}' within {:.1f}s.".format(
                        self.cfg['odom_topic'], timeout))
            rate.sleep()

    # ------------------------------------------------------------------ goals
    def _publish_goal(self, xyz):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.cfg['frame_id']
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def _xy_dist(self, ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    def _displacement_in_window(self, window):
        """Max XY distance between any two samples within trailing `window` s."""
        # Snapshot first: _odom_cb runs in the subscriber thread and may
        # append/popleft while we iterate, which raises
        # "deque mutated during iteration".
        snapshot = list(self._pose_history)
        if not snapshot:
            return 0.0
        now = snapshot[-1][0]
        cutoff = now - window
        pts = [(x, y) for (t, x, y) in snapshot if t >= cutoff]
        if len(pts) < 2:
            return 0.0
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        # Bounding box diagonal is a cheap, monotone proxy for "did it move".
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def _run_goal(self, idx, total, xyz, label):
        cfg = self.cfg
        rospy.loginfo("\033[96m[%d/%d %s] -> x=%.3f y=%.3f z=%.3f\033[0m",
                      idx, total, label, xyz[0], xyz[1], xyz[2])
        self._publish_goal(xyz)

        t_start = time.time()
        t_last_pub = t_start
        within_since = None
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            now = time.time()
            elapsed = now - t_start

            # 1) Timeout
            if elapsed > cfg['per_goal_timeout']:
                rospy.logwarn("[%d/%d %s] TIMEOUT after %.1fs; skipping.",
                              idx, total, label, elapsed)
                return 'timeout'

            # 2) Stuck check (only after the window has elapsed)
            if elapsed > cfg['stuck_window']:
                disp = self._displacement_in_window(cfg['stuck_window'])
                if disp < cfg['stuck_displacement']:
                    rospy.logwarn(
                        "[%d/%d %s] STUCK (disp=%.3fm < %.3fm in last %.1fs); "
                        "skipping.", idx, total, label, disp,
                        cfg['stuck_displacement'], cfg['stuck_window'])
                    return 'stuck'

            # 3) Arrived?
            x, y, _z, _t = self._pose
            d = self._xy_dist(x, y, xyz[0], xyz[1])
            if d <= cfg['arrive_threshold']:
                if within_since is None:
                    within_since = now
                elif (now - within_since) >= cfg['settle_time']:
                    rospy.loginfo(
                        "\033[92m[%d/%d %s] ARRIVED (d=%.2fm, settled %.1fs)"
                        "\033[0m", idx, total, label, d, cfg['settle_time'])
                    return 'arrived'
            else:
                within_since = None

            # 4) Periodic re-publish (protects against missed first publish)
            if (cfg['republish_interval'] > 0
                    and (now - t_last_pub) >= cfg['republish_interval']):
                self._publish_goal(xyz)
                t_last_pub = now
                rospy.loginfo_throttle(
                    5.0, "[%d/%d %s] re-published goal (d=%.2fm, t=%.1fs)",
                    idx, total, label, d, elapsed)

            rate.sleep()
        return 'shutdown'

    def run(self):
        cfg = self.cfg
        wps = list(cfg['waypoints'])
        if cfg['return_home']:
            wps.append(list(self._home))
        total = len(wps)

        results = []
        for i, wp in enumerate(wps, start=1):
            label = 'HOME' if (cfg['return_home'] and i == total) else 'WP'
            res = self._run_goal(i, total, wp, label)
            results.append((i, label, wp, res))
            if rospy.is_shutdown():
                break

        # Summary
        rospy.loginfo("\033[95m===== waypoint_runner summary =====\033[0m")
        for i, label, wp, res in results:
            rospy.loginfo("  %2d/%d %s (%.2f, %.2f, %.2f) -> %s",
                          i, total, label, wp[0], wp[1], wp[2], res)
        rospy.loginfo("\033[95m===================================\033[0m")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--file', '-f', type=str, default=None,
                   help='YAML file with waypoints and (optional) parameters.')
    p.add_argument('--waypoints', '-w', type=str, default=None,
                   help='Space-separated "x,y[,z]" tuples (overrides YAML).')
    p.add_argument('--frame-id', type=str, default=None,
                   help="Goal frame_id (default 'odom').")
    p.add_argument('--timeout', type=float, default=None,
                   help='Per-goal timeout in seconds.')
    p.add_argument('--arrive-threshold', type=float, default=None,
                   help='XY arrival threshold in meters.')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--return-home', dest='return_home', action='store_true',
                   default=None, help='Append start pose as final goal.')
    g.add_argument('--no-return-home', dest='return_home', action='store_false',
                   help='Do not return to start pose after last waypoint.')
    return p.parse_args(argv)


def main():
    args = parse_args(rospy.myargv()[1:])
    try:
        cfg = build_config(args)
    except (ValueError, OSError) as exc:
        sys.stderr.write('[waypoint_runner] config error: {}\n'.format(exc))
        sys.exit(2)

    rospy.init_node('waypoint_runner', anonymous=False)

    rospy.loginfo('waypoint_runner config:')
    for k in ('frame_id', 'arrive_threshold', 'settle_time',
              'per_goal_timeout', 'stuck_window', 'stuck_displacement',
              'republish_interval', 'return_home', 'odom_topic', 'goal_topic'):
        rospy.loginfo('  %s = %r', k, cfg[k])
    rospy.loginfo('  waypoints (%d):', len(cfg['waypoints']))
    for i, wp in enumerate(cfg['waypoints'], 1):
        rospy.loginfo('    %2d: (%.3f, %.3f, %.3f)', i, *wp)

    runner = WaypointRunner(cfg)
    try:
        runner.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
