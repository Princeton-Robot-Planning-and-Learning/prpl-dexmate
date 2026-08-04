"""Example: capture head camera RGB and depth images and save them for inspection."""

from pathlib import Path

import cv2
import numpy as np
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot

output_dir = Path(__file__).parent / "head_camera_output"
output_dir.mkdir(exist_ok=True)

configs = get_robot_config()
configs.sensors["head_camera"].enabled = True
robot = Robot(configs=configs)

camera_data = robot.sensors.head_camera.get_obs(
    obs_keys=["left_rgb", "right_rgb", "depth"]
)

for key in ("left_rgb", "right_rgb"):
    rgb = camera_data[key]
    print(f"{key} shape: {rgb.shape}")
    cv2.imwrite(str(output_dir / f"{key}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

depth = camera_data["depth"]
print(f"depth shape: {depth.shape}")
np.save(output_dir / "depth.npy", depth)

# Depth values are float distances with possible inf/nan; rescale the finite
# range to 0-255 so the preview PNG is viewable in any image viewer.
finite = np.isfinite(depth)
preview = np.zeros(depth.shape, dtype=np.uint8)
if finite.any():
    d_min = depth[finite].min()
    d_max = depth[finite].max()
    print(f"depth range: {d_min:.3f} to {d_max:.3f}")
    if d_max > d_min:
        preview[finite] = (255 * (depth[finite] - d_min) / (d_max - d_min)).astype(
            np.uint8
        )
cv2.imwrite(str(output_dir / "depth_preview.png"), preview)

print(f"Saved images to {output_dir}")

robot.shutdown()
