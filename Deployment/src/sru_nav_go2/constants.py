"""Constants and configuration values for the Go2 + Odin1 SRU navigation port.

Defaults are tuned conservatively for Unitree Go2 (no wheels, weaker lateral
ability) versus the original B2W (wheeled). Override at runtime through the
companion YAML / launch parameters.
"""

# ---------------------------------------------------------------------------
# Control parameters
# ---------------------------------------------------------------------------
DEFAULT_CONTROL_FREQUENCY = 5.0       # Hz (matches training)
DEFAULT_MIN_DEPTH = 0.25               # meters
DEFAULT_MAX_DEPTH = 10.0               # meters
ARRIVE_GOAL_THRESHOLD = 0.75           # meters
NEAR_GOAL_THRESHOLD_MULTIPLIER = 2.0
JOYSTICK_TIMEOUT = 15.0                # seconds (safety deadman timeout)

# ---------------------------------------------------------------------------
# Policy output scaling. Original B2W used [1.5, 1.0, 1.0].
# Go2 is conservative until verified safe in real hardware.
# ---------------------------------------------------------------------------
POLICY_SCALE = [0.6, 0.3, 0.6]         # [linear_x, linear_y, angular_z]
LATERAL_VELOCITY_SCALE = 0.6           # extra damping on linear_y

# Low-pass filter coefficients (alpha): output = a*new + (1-a)*prev
LOW_PASS_FILTER_COEF = [0.9, 0.5, 0.5]

# ---------------------------------------------------------------------------
# Joystick axis / button mappings (PS5-style, same as original)
# ---------------------------------------------------------------------------
JOYSTICK_AXIS_LINEAR_X = 1
JOYSTICK_AXIS_LINEAR_Y = 0
JOYSTICK_AXIS_LINEAR_Z = 3
JOYSTICK_AXIS_ANGULAR_Z = 2
JOYSTICK_AXIS_SMART = 5

BUTTON_RESET_HIDDEN_STATE = 2
BUTTON_RECORD_WAYPOINT = 6
BUTTON_CLEAR_WAYPOINT = 4
BUTTON_SEND_GOAL = 10
BUTTON_ABORT = 9
BUTTON_TRIGGER_WAYPOINTS = 1
BUTTON_FORWARD = 11
BUTTON_BACKWARD = 12
BUTTON_LEFT = 13
BUTTON_RIGHT = 14
BUTTON_UP = 3
BUTTON_DOWN = 0

# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------
LINEAR_SCALE = 1.0
ANGULAR_SCALE = 1.0
MOVING_SCALE = 0.2
SMART_JOYSTICK_SCALE = 5.0
SMART_JOYSTICK_UPDATE_FREQUENCY = 5.0  # Hz
SMART_JOYSTICK_Z_SCALE = 0.25
SMART_JOYSTICK_FILTER_ALPHA = 0.2

# ---------------------------------------------------------------------------
# Timer intervals
# ---------------------------------------------------------------------------
WAYPOINT_PUBLISH_INTERVAL = 0.2        # 5 Hz
TARGET_VECTOR_PUBLISH_INTERVAL = 0.2   # 5 Hz
TRIGGER_BUTTON_COOLDOWN = 1.0          # seconds

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
TWIST_MARKER_SCALE = 5.0
TWIST_MARKER_ID = 0
TARGET_VECTOR_MARKER_ID = 1
MOVING_GOAL_MARKER_ID = 2
WAYPOINTS_MARKER_ID = 3

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
GRAVITY_MAGNITUDE = 9.81               # m/s^2
