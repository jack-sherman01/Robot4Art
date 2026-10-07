"""Isaac Sim standalone script: run the same 20-stroke painting plan as
franka_paint_sim.py, but periodically capture real rendered viewport
frames of the robot's full-body motion, for the "watch it paint" video
on the website.

This succeeds where earlier attempts (a custom Camera sensor bound to
the active viewport's render product, explicit lighting, forcing the
Replicator orchestrator to step -- see git history / sim/README.md)
produced an identical, scene-independent placeholder frame regardless
of settings. The fix was dropping the custom Camera sensor entirely and
using Kit's own viewport-capture utility
(omni.kit.viewport.utility.capture_viewport_to_file) against the
viewport that's already being driven by the main render loop, auto-framed
with frame_viewport_prims -- a different, simpler, standard code path.

Must be run with Isaac Sim's own bundled Python, via Docker (see
sim/README.md for why):

    sim/docker_run.sh franka_paint_video.py --headless

Rendering is expensive, so most physics steps still run with
render=False (fast); only every CAPTURE_EVERY-th step pays the
rendering + capture cost. Frames are written to --frames-dir and then
assembled into an MP4 with ffmpeg on the host (see "Rendering the
painting video" in sim/README.md) -- ffmpeg isn't available inside this
container.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--show", dest="headless", action="store_false")
parser.add_argument("--frames-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "frames"))
parser.add_argument("--position-tolerance", type=float, default=0.012, help="meters")
parser.add_argument("--min-steps-per-waypoint", type=int, default=15)
parser.add_argument("--max-steps-per-waypoint", type=int, default=150)
parser.add_argument("--capture-every", type=int, default=45, help="physics steps between captured frames")
parser.add_argument("--capture-pumps", type=int, default=3, help="simulation_app.update() calls after each capture")
args, _ = parser.parse_known_args()

os.makedirs(args.frames_dir, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402
from isaacsim.robot.manipulators.examples.franka import Franka  # noqa: E402
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import (  # noqa: E402
    RMPFlowController,
)
from omni.kit.viewport.utility import capture_viewport_to_file, frame_viewport_prims, get_active_viewport  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

from canvas import CANVAS_CENTER, CANVAS_SIZE, canvas_to_world, downward_orientation  # noqa: E402
from composition import BRUSHES  # noqa: E402
from example_answers import load_example_painting  # noqa: E402

# How wide a stroke of relative weight 1.0 looks in world meters, with the
# brush's own width_mult layered on top -- 0.05 matches the proportion
# preview_compositions.py uses for the same stroke in normalized [0,1]
# canvas units (0.05 * CANVAS_SIZE), so the simulated paint mark is the
# same relative width as the 2D reference render.
_WIDTH_SCALE_M = 0.05 * CANVAS_SIZE
_MIN_WIDTH_M = 0.003
_PAINT_Z = CANVAS_CENTER[2] + 0.001  # just above the canvas surface, avoids z-fighting


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _tapered_edges(points_xy: np.ndarray, max_width: float, floor: float) -> tuple:
    """Same taper math as preview_compositions.tapered_polygon (narrower
    toward both ends unless floor=1.0), returning the left/right boundary
    of the ribbon separately instead of one flattened outline -- these
    line up index-for-index so consecutive samples can be stitched into
    a quad each, which triangulates correctly even for a curvy (possibly
    non-convex) stroke; a single filled N-gon outline can't guarantee that."""
    n = len(points_xy)
    dirs = np.zeros_like(points_xy)
    dirs[1:-1] = points_xy[2:] - points_xy[:-2]
    dirs[0] = points_xy[1] - points_xy[0]
    dirs[-1] = points_xy[-1] - points_xy[-2]
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dirs = dirs / norms
    perp = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)
    t = np.linspace(0.0, 1.0, n)
    taper = np.sin(np.pi * t) ** 0.7
    widths = max_width * (floor + (1.0 - floor) * taper)
    left = points_xy + perp * (widths / 2)[:, None]
    right = points_xy - perp * (widths / 2)[:, None]
    return left, right


class Painter:
    """Lays down one flat, colored mesh per stroke, so the canvas fills in
    with the painting instead of staying blank -- there's no physical
    ink/pigment model, just a visual stand-in shaped like the stroke the
    end effector is about to trace. One mesh per stroke (not one prim per
    waypoint) is deliberate: an earlier version added a tiny prim for
    every segment (hundreds total) and the live viewport capture silently
    stopped updating partway through the run -- each world.scene.add()
    mid-simulation seems to force an expensive re-sync that a few hundred
    rapid-fire calls couldn't keep up with. 15 prims (one per stroke) is
    cheap and reliable."""

    def __init__(self, world, brush: str):
        self.world = world
        self.brush = BRUSHES[brush]
        self.stage = omni.usd.get_context().get_stage()
        self._next_id = 0

    def paint_stroke(self, points_norm: np.ndarray, color_hex: str, weight: float) -> None:
        points_xy = np.array([canvas_to_world(x, y, pen_down=True)[:2] for x, y in points_norm])
        max_width = max(_WIDTH_SCALE_M * weight * self.brush["width_mult"], _MIN_WIDTH_M)
        left, right = _tapered_edges(points_xy, max_width, self.brush["floor"])

        n = len(points_xy)
        verts = []
        for i in range(n):
            verts.append(Gf.Vec3f(float(left[i, 0]), float(left[i, 1]), _PAINT_Z))
            verts.append(Gf.Vec3f(float(right[i, 0]), float(right[i, 1]), _PAINT_Z))
        counts, indices = [], []
        for i in range(n - 1):
            a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2
            counts += [3, 3]
            indices += [a, b, c, a, c, d]

        prim_path = f"/World/Paint/stroke_{self._next_id:03d}"
        self._next_id += 1
        mesh = UsdGeom.Mesh.Define(self.stage, prim_path)
        mesh.CreatePointsAttr(verts)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateDoubleSidedAttr(True)
        rgb = _hex_to_rgb(color_hex)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))])


class FrameCounter:
    """Tracks physics-step count and captured-frame count across the
    whole run, so goto() can decide per step whether to pay for a
    render+capture or just step physics quickly."""

    def __init__(self, viewport, frames_dir, capture_every, capture_pumps):
        self.viewport = viewport
        self.frames_dir = frames_dir
        self.capture_every = capture_every
        self.capture_pumps = capture_pumps
        self.step_count = 0
        self.frame_count = 0

    def step(self, world):
        self.step_count += 1
        if self.step_count % self.capture_every == 0:
            world.step(render=True)
            path = os.path.join(self.frames_dir, f"frame_{self.frame_count:05d}.png")
            capture_viewport_to_file(self.viewport, path)
            for _ in range(self.capture_pumps):
                simulation_app.update()
            self.frame_count += 1
        else:
            world.step(render=False)


def goto(world, counter, controller, franka, articulation_controller, target_pos, target_orient):
    steps = 0
    while steps < args.max_steps_per_waypoint:
        counter.step(world)
        actions = controller.forward(target_end_effector_position=target_pos, target_end_effector_orientation=target_orient)
        articulation_controller.apply_action(actions)
        ee_pos, _ = franka.end_effector.get_world_pose()
        steps += 1
        error = float(np.linalg.norm(ee_pos - target_pos))
        if steps >= args.min_steps_per_waypoint and error <= args.position_tolerance:
            break


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
    world.scene.add(
        VisualCuboid(
            prim_path="/World/Canvas",
            name="canvas",
            position=CANVAS_CENTER + np.array([0.0, 0.0, -0.0025]),
            scale=np.array([CANVAS_SIZE, CANVAS_SIZE, 0.005]),
            color=np.array([0.96, 0.96, 0.94]),
        )
    )

    stage = omni.usd.get_context().get_stage()
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1200.0)
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(3500.0)
    key.CreateAngleAttr(1.0)

    world.reset()

    viewport = get_active_viewport()
    for _ in range(10):
        world.step(render=True)
    frame_viewport_prims(viewport, ["/World/Franka", "/World/Canvas"])
    for _ in range(10):
        world.step(render=True)

    controller = RMPFlowController(name="painting_controller", robot_articulation=franka)
    articulation_controller = franka.get_articulation_controller()
    orientation = downward_orientation()
    counter = FrameCounter(viewport, args.frames_dir, args.capture_every, args.capture_pumps)

    plan, brush, rationale = load_example_painting()
    print(f"[composition] brush={brush} strokes={len(plan)}", flush=True)
    print(f"[rationale] {rationale}", flush=True)
    painter = Painter(world, brush)
    wall_start = time.time()

    for stroke in plan:
        first_xy = stroke.points[0]
        goto(world, counter, controller, franka, articulation_controller, canvas_to_world(*first_xy, pen_down=False), orientation)
        goto(world, counter, controller, franka, articulation_controller, canvas_to_world(*first_xy, pen_down=True), orientation)
        painter.paint_stroke(stroke.points, stroke.color, stroke.width)
        for xy in stroke.points[1:]:
            goto(world, counter, controller, franka, articulation_controller, canvas_to_world(*xy, pen_down=True), orientation)
        last_xy = stroke.points[-1]
        goto(world, counter, controller, franka, articulation_controller, canvas_to_world(*last_xy, pen_down=False), orientation)
        print(f"[stroke done] {stroke.name}", flush=True)

    # Lift the arm well clear of the canvas and hold, so the video (and
    # its poster frame) end on an unobstructed view of the finished
    # painting instead of the gripper parked over the center of what it
    # just painted. A straight-up lift is used (not a sideways move)
    # because it changes the arm's screen position regardless of the
    # camera's viewing angle -- an earlier sideways-only retreat barely
    # moved on screen from this camera's elevated, oblique framing.
    retreat_target = canvas_to_world(0.5, 0.5, pen_down=False) + np.array([0.0, 0.0, 0.4])
    goto(world, counter, controller, franka, articulation_controller, retreat_target, orientation)
    ee_pos, _ = franka.end_effector.get_world_pose()
    print(f"[retreat] target={retreat_target} reached={ee_pos}", flush=True)
    for _ in range(300):
        counter.step(world)
    print("[retreat] hold done", flush=True)

    # Flush any still-pending async captures before closing.
    for _ in range(60):
        simulation_app.update()

    wall_elapsed = time.time() - wall_start
    print(f"VIDEO_RESULT steps={counter.step_count} frames={counter.frame_count} wall_seconds={wall_elapsed:.1f}", flush=True)
    print(f"Frames written to {args.frames_dir}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
