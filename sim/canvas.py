"""Canvas geometry for the painting simulation.

The canvas is a flat square lying on the table in front of the robot base,
in the robot's world XY plane. Normalized canvas coordinates (x, y) in
[0, 1]^2 map to world-frame positions; z selects whether the pen/brush is
touching the canvas or lifted for travel between strokes.

This is deliberately independent of which robot is used -- it is the
shared "canvas contract" that any robot adapter (Franka today; Kinova /
xArm later, per the research proposal) maps into its own task frame.
"""

import numpy as np

try:
    from isaacsim.core.utils.rotations import euler_angles_to_quat
except ImportError:  # pragma: no cover - only available inside Isaac Sim's Python
    euler_angles_to_quat = None

CANVAS_CENTER = np.array([0.5, 0.0, 0.0])  # meters, in the robot base frame
CANVAS_SIZE = 0.30  # meters, square canvas side length
PEN_DOWN_HEIGHT = 0.015  # m above CANVAS_CENTER z, pen/brush touching the canvas
PEN_UP_HEIGHT = 0.09  # m above CANVAS_CENTER z, pen/brush lifted while traveling


def canvas_to_world(x_norm: float, y_norm: float, pen_down: bool) -> np.ndarray:
    """Map normalized canvas coords in [0, 1]^2 to a world-frame position."""
    x = CANVAS_CENTER[0] + (x_norm - 0.5) * CANVAS_SIZE
    y = CANVAS_CENTER[1] + (y_norm - 0.5) * CANVAS_SIZE
    z = CANVAS_CENTER[2] + (PEN_DOWN_HEIGHT if pen_down else PEN_UP_HEIGHT)
    return np.array([x, y, z])


def downward_orientation() -> np.ndarray:
    """Quaternion (w, x, y, z) pointing the end effector straight down at the canvas."""
    if euler_angles_to_quat is None:
        raise RuntimeError("downward_orientation() requires Isaac Sim's Python (isaacsim.core.utils.rotations)")
    return euler_angles_to_quat(np.array([np.pi, 0.0, 0.0]))
