"""Render an MP4 of the robot painting from a recorded simulation trace.

Does not need Isaac Sim or Docker -- reads sim/output/trace.json (written
by franka_paint_sim.py) and animates the real recorded end-effector
trajectory with matplotlib. This is a visualization of actual simulated
data (not a 3D render of the robot/scene): a pen marker traces the
canvas, drawing each stroke in its color as the simulated end effector
moves through it, pen lifts shown as travel without a trail.

Why not an Isaac Sim camera render: attempted extensively (several
camera-binding approaches, explicit lighting, Replicator orchestrator
stepping) against this project's Docker-based Isaac Sim 4.5.0 setup; every
attempt returned an identical, scene-independent placeholder frame
regardless of camera position/orientation/lighting, strongly suggesting
the offscreen render product isn't actually connected to the stage in
this headless container configuration. Revisit if that gets resolved;
this trace-based animation is the reliable fallback in the meantime.

Usage:
    python3 sim/render_paint_video.py [--trace sim/output/trace.json] \
        [--out sim/output/paint_video.mp4] [--fps 25] [--duration 24]
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canvas import CANVAS_CENTER, CANVAS_SIZE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

PAPER = "#FAF9F6"
INK = "#161513"
MUTED = "#706B61"


def world_to_norm(x, y):
    return (x - CANVAS_CENTER[0]) / CANVAS_SIZE + 0.5, (y - CANVAS_CENTER[1]) / CANVAS_SIZE + 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=os.path.join(HERE, "output", "trace.json"))
    parser.add_argument("--out", default=os.path.join(HERE, "output", "paint_video.mp4"))
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--duration", type=float, default=22.0, help="target video length in seconds (excluding hold)")
    parser.add_argument("--hold-seconds", type=float, default=2.0, help="freeze on the finished piece at the end")
    args = parser.parse_args()

    with open(args.trace) as f:
        data = json.load(f)
    trace = data["trace"]

    stroke_colors = data.get("stroke_colors")
    if not stroke_colors:
        raise SystemExit(
            "trace.json has no 'stroke_colors' -- it was written by an older franka_paint_sim.py. "
            "Re-run the simulation (sim/docker_run.sh franka_paint_sim.py --headless) to regenerate it."
        )

    n_frames_target = max(1, int(args.fps * args.duration))
    step = max(1, len(trace) // n_frames_target)
    frames = trace[::step]
    if frames[-1] is not trace[-1]:
        frames.append(trace[-1])

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=120)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#DEDAD2")
    title = ax.set_title("", fontsize=13, color=INK, loc="left", pad=12)
    caption = fig.text(0.02, 0.015, "", fontsize=9, color=MUTED)

    trail_lines = {}  # stroke name -> Line2D
    trail_data = {}  # stroke name -> ([xs], [ys])
    pen_marker, = ax.plot([], [], "o", color=INK, markersize=9, zorder=10)
    pen_ring, = ax.plot([], [], "o", markerfacecolor="none", markeredgecolor=INK, markersize=14, zorder=10, alpha=0.5)

    def init():
        pen_marker.set_data([], [])
        pen_ring.set_data([], [])
        return [pen_marker, pen_ring]

    def update(frame):
        nx, ny = world_to_norm(frame["x"], frame["y"])
        ny_plot = 1 - ny
        stroke_name = frame["stroke"]
        color = stroke_colors.get(stroke_name, INK)

        if frame["pen_down"]:
            xs, ys = trail_data.setdefault(stroke_name, ([], []))
            xs.append(nx)
            ys.append(ny_plot)
            if stroke_name not in trail_lines:
                (line,) = ax.plot([], [], "-", color=color, linewidth=3.2, solid_capstyle="round", zorder=5)
                trail_lines[stroke_name] = line
            trail_lines[stroke_name].set_data(xs, ys)
            pen_marker.set_markerfacecolor(color)
            pen_marker.set_markeredgecolor(color)
        else:
            pen_marker.set_markerfacecolor("none")
            pen_marker.set_markeredgecolor(MUTED)

        pen_marker.set_data([nx], [ny_plot])
        pen_ring.set_data([nx], [ny_plot])

        title.set_text(f"Robot4Art — simulated Franka arm  ·  stroke {stroke_name.replace('_', ' ')}")
        caption.set_text(f"NVIDIA Isaac Sim 4.5.0 · recorded end-effector trajectory · step {frame['step']:05d}/{data['num_steps']}")
        return list(trail_lines.values()) + [pen_marker, pen_ring, title, caption]

    hold_frames = int(args.fps * args.hold_seconds)
    all_frames = frames + [frames[-1]] * hold_frames

    anim = animation.FuncAnimation(fig, update, frames=all_frames, init_func=init, blit=False, interval=1000 / args.fps)

    writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    anim.save(args.out, writer=writer)
    print(f"Wrote {args.out} ({len(all_frames)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
