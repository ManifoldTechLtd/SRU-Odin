# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Camera configuration parameters for different robots and camera types.

This module provides camera-specific parameters for depth noise generation and encoding.
"""

import os
from typing import Optional

from isaaclab.utils import configclass

# Local assets directory for this extension
# Path: depth_utils -> mdp -> navigation -> assets/data
_ASSETS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "data")
)


def _get_encoder_path(model_filename: str) -> str:
    """Helper function to construct encoder model path.

    Args:
        model_filename: Name of the encoder model file (e.g., 'vae_pretrain_new.pth')

    Returns:
        Full path to the encoder model file
    """
    return os.path.join(_ASSETS_DIR, "Policies", "depth_encoder", model_filename)


@configclass
class CameraConfig:
    """Configuration class for camera parameters.

    This class contains all camera-specific parameters needed for depth noise generation
    and depth encoder initialization.
    """

    # Camera intrinsic parameters
    focal_length: float = 25.0
    baseline: float = 0.12

    # Depth range parameters
    min_depth: float = 0.25
    max_depth: float = 10.0

    # Camera resolution (width, height)
    resolution: tuple[int, int] = (53, 30)

    # Depth encoder model path
    depth_encoder_path: str = ""

    def __post_init__(self):
        """Post-initialization to set default encoder path if not provided."""
        if not self.depth_encoder_path:
            self.depth_encoder_path = _get_encoder_path("vae_pretrain_fuse.pth")


# Predefined camera configurations
ZEDX_CAMERA_CONFIG = CameraConfig(
    focal_length=25.0,
    baseline=0.12,
    min_depth=0.25,
    max_depth=10.0,
    resolution=(64, 40),
    depth_encoder_path=_get_encoder_path("vae_pretrain_new.pth"),
)
"""Configuration for ZedX camera (used with b2w and aow_d robots)."""


# -----------------------------------------------------------------------------
# Odin1 (LiDAR-aligned depth on Unitree Go2)
# -----------------------------------------------------------------------------
# Real intrinsics @ native 1600x1296 (from ROS calibration):
#     fx = 737.357 px, fy = 737.292 px, cx = 794.372 px, cy = 666.259 px
#     hFOV = 2*atan(W / 2fx) = 94.67 deg
#     vFOV = 2*atan(H / 2fy) = 82.65 deg
#
# Scaled to VAE-fixed 64x40 input (keep FOV identical):
#     fx_sim = 737.357 * 64/1600   = 29.49 px
#     fy_sim = 737.292 * 40/1296   = 22.76 px
#     cx_sim ~= 31.77 px (close to width/2)
#     cy_sim ~= 20.56 px (close to height/2)
#
# Noise model caveat: the upstream DepthNoise module simulates *stereo*
# disparity-quantization noise (sigma_d proportional to d^2 / (fx * baseline)).
# Odin1 depth comes from LiDAR-to-image alignment, where the real noise is
# closer to a constant ~1-2 cm regardless of range. We keep the stereo-style
# noise as a coarse proxy (set focal_length=fx_sim, baseline=0.05 m) but if
# sim2real depth fidelity becomes a problem, swap in a constant-sigma noise
# model later.
ODIN1_CAMERA_CONFIG = CameraConfig(
    focal_length=29.49,   # pixels (NOT mm; matches sim pinhole fx)
    baseline=0.05,        # fudge: makes per-meter quantization noise sane
    min_depth=0.25,
    max_depth=10.0,
    resolution=(64, 40),  # do not change; VAE encoder input is fixed
    depth_encoder_path=_get_encoder_path("vae_pretrain_new.pth"),
)
"""Configuration for Odin1 LiDAR-aligned depth (used with Go2)."""


# Default camera configuration
DEFAULT_CAMERA_CONFIG = ZEDX_CAMERA_CONFIG
"""Default camera configuration (ZedX camera settings)."""

# Robot-to-camera mapping. Go2 now uses Odin1 (LiDAR-aligned depth). b2w and
# aow_d keep the original ZedX profile so the upstream paper results stay
# reproducible.
ROBOT_CAMERA_CONFIGS = {
    "b2w": ZEDX_CAMERA_CONFIG,
    "aow_d": ZEDX_CAMERA_CONFIG,
    "go2": ODIN1_CAMERA_CONFIG,
}
"""Dictionary mapping robot names to their camera configurations."""


def get_camera_config(robot_name: str, use_default_fallback: bool = False) -> CameraConfig:
    """Get camera configuration for a specific robot.

    Args:
        robot_name: Name of the robot (e.g., 'b2w', 'aow_d')
        use_default_fallback: If True, return DEFAULT_CAMERA_CONFIG when robot not found
                             instead of raising an error (default: False)

    Returns:
        Camera configuration for the specified robot

    Raises:
        KeyError: If robot_name not found and use_default_fallback is False
    """
    if robot_name in ROBOT_CAMERA_CONFIGS:
        return ROBOT_CAMERA_CONFIGS[robot_name]

    if use_default_fallback:
        return DEFAULT_CAMERA_CONFIG

    available_robots = ", ".join(sorted(ROBOT_CAMERA_CONFIGS.keys()))
    raise KeyError(
        f"Robot '{robot_name}' not found in camera configurations. "
        f"Available robots: {available_robots}"
    )
