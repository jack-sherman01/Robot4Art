"""Isaac Sim standalone script: run the same 10-stroke painting plan as
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
from pxr import UsdLux  # noqa: E402

from canvas import CANVAS_CENTER, CANVAS_SIZE, canvas_to_world, downward_orientation  # noqa: E402
from composition import strokes_from_brief  # noqa: E402
from example_answers import load_example_brief  # noqa: E402


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

    brief, rationale = load_example_brief()
    plan = strokes_from_brief(brief)
    print(f"[composition] style={brief.style} pens={brief.pen_names} strokes={len(plan)}", flush=True)
    print(f"[rationale] {rationale}", flush=True)
    wall_start = time.time()

    for stroke in plan:
        first_xy = stroke.points[0]
        goto(world, counter, controller, franka, articulation_controller, canvas_to_world(*first_xy, pen_down=False), orientation)
        goto(world, counter, controller, franka, articulation_controller, canvas_to_world(*first_xy, pen_down=True), orientation)
        for xy in stroke.points[1:]:
            goto(world, counter, controller, franka, articulation_controller, canvas_to_world(*xy, pen_down=True), orientation)
        last_xy = stroke.points[-1]
        goto(world, counter, controller, franka, articulation_controller, canvas_to_world(*last_xy, pen_down=False), orientation)
        print(f"[stroke done] {stroke.name}", flush=True)

    # Flush any still-pending async captures before closing.
    for _ in range(60):
        simulation_app.update()

    wall_elapsed = time.time() - wall_start
    print(f"VIDEO_RESULT steps={counter.step_count} frames={counter.frame_count} wall_seconds={wall_elapsed:.1f}", flush=True)
    print(f"Frames written to {args.frames_dir}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
