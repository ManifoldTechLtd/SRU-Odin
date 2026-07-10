"""Verify Go2 nav tasks are registered correctly. Must be launched via isaaclab.sh -p."""

from isaaclab.app import AppLauncher

# Headless minimal launcher
app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_nav_task  # noqa: F401, E402

ids = sorted(s for s in gym.envs.registry.keys() if "Go2" in s or "B2W" in s)
print("=" * 60)
print("Registered Go2 + B2W tasks:")
print("=" * 60)
for i in ids:
    print(f"  {i}")

simulation_app.close()
