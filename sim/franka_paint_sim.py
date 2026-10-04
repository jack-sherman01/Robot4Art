"""Isaac Sim standalone script: drive a Franka arm through a 10-stroke
painting plan on a virtual canvas, headless, and record the result.

This is the Phase-0 simulation validation step described in the research
proposal (private/proposal_en.tex / proposal_zh.tex): before any physical
robot or VLM integration, confirm that a stroke plan in normalized canvas
coordinates can actually be executed by a robot arm's motion controller
and produces a sane, trackable end-effector trajectory.

Must be run with Isaac Sim's own bundled Python. On this machine that
means via Docker (see sim/README.md for why):

    sim/docker_run.sh franka_paint_sim.py --headless

If a native Isaac Sim install is ever available under this account
instead (see find_isaac_sim.py), run it directly:

    $(python3 sim/find_isaac_sim.py)/python.sh sim/franka_paint_sim.py

Known limitation: this validates *motion* (the end effector visits the
right places at the right times), not actual marking of the canvas --
there is no ink/paint deposition model in this first pass. The output
plot overlays the intended stroke paths with the executed end-effector
trace so that can be checked visually.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--show", dest="headless", action="store_false", help="open the Isaac Sim GUI window")
parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
parser.add_argument("--position-tolerance", type=float, default=0.012, help="meters")
parser.add_argument("--min-steps-per-waypoint", type=int, default=15)
parser.add_argument("--max-steps-per-waypoint", type=int, default=150)
args, _ = parser.parse_known_args()

os.makedirs(args.output_dir, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402
from isaacsim.robot.manipulators.examples.franka import Franka  # noqa: E402
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import (  # noqa: E402
    RMPFlowController,
)

from canvas import CANVAS_CENTER, CANVAS_SIZE, canvas_to_world, downward_orientation  # noqa: E402
from composition import strokes_from_brief  # noqa: E402
from example_answers import load_example_brief  # noqa: E402


def goto(world, controller, franka, articulation_controller, target_pos, target_orient, trace, stroke_name, pen_down):
    """Step the sim, driving the end effector toward target_pos via RMPFlow,
    until it is within tolerance or a max step budget is hit. Records every
    simulated end-effector pose along the way."""
    steps = 0
    while steps < args.max_steps_per_waypoint:
        world.step(render=not args.headless)
        actions = controller.forward(target_end_effector_position=target_pos, target_end_effector_orientation=target_orient)
        articulation_controller.apply_action(actions)
        ee_pos, _ = franka.end_effector.get_world_pose()
        trace.append(
            {
                "step": len(trace),
                "stroke": stroke_name,
                "pen_down": pen_down,
                "x": float(ee_pos[0]),
                "y": float(ee_pos[1]),
                "z": float(ee_pos[2]),
                "target_x": float(target_pos[0]),
                "target_y": float(target_pos[1]),
                "target_z": float(target_pos[2]),
            }
        )
        steps += 1
        error = float(np.linalg.norm(ee_pos - target_pos))
        if steps >= args.min_steps_per_waypoint and error <= args.position_tolerance:
            break
    return trace


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

    world.reset()

    controller = RMPFlowController(name="painting_controller", robot_articulation=franka)
    articulation_controller = franka.get_articulation_controller()
    orientation = downward_orientation()

    brief, rationale = load_example_brief()
    plan = strokes_from_brief(brief)
    print(f"[composition] style={brief.style} pens={brief.pen_names} strokes={len(plan)}", flush=True)
    print(f"[rationale] {rationale}", flush=True)
    trace = []
    wall_start = time.time()

    for stroke in plan:
        first_xy = stroke.points[0]
        travel_target = canvas_to_world(first_xy[0], first_xy[1], pen_down=False)
        goto(world, controller, franka, articulation_controller, travel_target, orientation, trace, stroke.name, False)

        down_target = canvas_to_world(first_xy[0], first_xy[1], pen_down=True)
        goto(world, controller, franka, articulation_controller, down_target, orientation, trace, stroke.name, True)

        for xy in stroke.points[1:]:
            target = canvas_to_world(xy[0], xy[1], pen_down=True)
            goto(world, controller, franka, articulation_controller, target, orientation, trace, stroke.name, True)

        last_xy = stroke.points[-1]
        lift_target = canvas_to_world(last_xy[0], last_xy[1], pen_down=False)
        goto(world, controller, franka, articulation_controller, lift_target, orientation, trace, stroke.name, False)
        print(f"[stroke done] {stroke.name}: {len(stroke.points)} waypoints", flush=True)

    wall_elapsed = time.time() - wall_start

    trace_path = os.path.join(args.output_dir, "trace.json")
    with open(trace_path, "w") as f:
        json.dump(
            {
                "num_steps": len(trace),
                "wall_seconds": wall_elapsed,
                "physics_dt": world.get_physics_dt(),
                "sim_seconds": len(trace) * world.get_physics_dt(),
                "strokes": [s.name for s in plan],
                "stroke_colors": {s.name: s.color for s in plan},
                "grammar": brief.grammar,
                "style": brief.style,
                "pen_names": brief.pen_names,
                "rationale": rationale,
                "trace": trace,
            },
            f,
            indent=2,
        )

    max_pen_down_error = max(
        (((p["x"] - p["target_x"]) ** 2 + (p["y"] - p["target_y"]) ** 2) ** 0.5 for p in trace if p["pen_down"]),
        default=float("nan"),
    )

    print(f"SIM_RESULT steps={len(trace)} wall_seconds={wall_elapsed:.1f} sim_seconds={len(trace) * world.get_physics_dt():.1f} "
          f"max_pen_down_xy_error_m={max_pen_down_error:.4f}", flush=True)
    print(f"Trace written to {trace_path}", flush=True)

    _save_plot(plan, trace, os.path.join(args.output_dir, "paint_trace.png"))

    simulation_app.close()


def _save_plot(plan, trace, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for stroke in plan:
        ax.plot(stroke.points[:, 0], stroke.points[:, 1], "--", color=stroke.color, linewidth=1, alpha=0.6, label=f"{stroke.name} (intended)")

    # Convert executed world-frame XY back into normalized canvas coords for
    # an apples-to-apples overlay with the intended plan.
    by_stroke = {}
    for p in trace:
        if not p["pen_down"]:
            continue
        nx = (p["x"] - CANVAS_CENTER[0]) / CANVAS_SIZE + 0.5
        ny = (p["y"] - CANVAS_CENTER[1]) / CANVAS_SIZE + 0.5
        by_stroke.setdefault(p["stroke"], ([], []))[0].append(nx)
        by_stroke[p["stroke"]][1].append(ny)

    for name, (xs, ys) in by_stroke.items():
        ax.plot(xs, ys, "-", linewidth=2, alpha=0.9, label=f"{name} (executed)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Robot4Art - Isaac Sim stroke plan vs. executed end-effector trace")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Plot written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
