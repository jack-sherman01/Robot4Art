"""Render a few example compositions to PNG for visual inspection.

Does not need Isaac Sim -- this only exercises composition.py, which is
pure Python/numpy/matplotlib. Useful for checking the grammar library
(sim/composition.py) produces sane, on-canvas, <=10-stroke artwork without
waiting on a robot simulator.

Strokes are rendered as tapered filled polygons (thin brush builder
below), not constant-width lines -- a uniform-width centerline reads as
a diagram, not a painting. This is a rendering concern only; the
centerline points stroke_plan/franka_paint_sim actually use for robot
execution are unaffected.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from composition import PromptAnswers, compose

EXAMPLES = [
    PromptAnswers("blue", "Pittsburgh", "to build robots that help people", "curious"),
    PromptAnswers("warm red", "Beijing", "to travel the world", "excited"),
    PromptAnswers("forest green", "Tokyo", "to write a novel", "calm"),
    PromptAnswers("sunset orange", "Paris", "to open a bakery", "cozy"),
    PromptAnswers("deep purple", "Shanghai", "to become a scientist", "determined"),
    PromptAnswers("teal", "Austin", "to make music", "playful"),
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "compositions")

CANVAS_BG = "#F6F1E7"  # warm linen, not stark white


def tapered_polygon(points: np.ndarray, max_width: float) -> np.ndarray:
    """Build a filled brush-stroke outline from a centerline: thin at
    both ends, full width in the middle, like a loaded brush."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 2:
        return points

    dirs = np.zeros_like(points)
    dirs[1:-1] = points[2:] - points[:-2]
    dirs[0] = points[1] - points[0]
    dirs[-1] = points[-1] - points[-2]
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dirs = dirs / norms
    perp = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)

    t = np.linspace(0.0, 1.0, n)
    taper = np.sin(np.pi * t) ** 0.7
    widths = max_width * (0.22 + 0.78 * taper)

    left = points + perp * (widths / 2)[:, None]
    right = points - perp * (widths / 2)[:, None]
    return np.vstack([left, right[::-1]])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    fig.patch.set_facecolor("#FAF9F6")
    for ax, answers in zip(axes.flat, EXAMPLES):
        strokes, brief = compose(answers)
        for s in strokes:
            poly = tapered_polygon(s.points, max_width=0.05 * s.width)
            ax.fill(poly[:, 0], poly[:, 1], color=s.color, linewidth=0, alpha=0.96, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor(CANVAS_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#DEDAD2")
        ax.set_title(
            f"{answers.favorite_color} / {answers.favorite_city}\n{brief.grammar} · {'/'.join(brief.pen_names)} ({len(strokes)} strokes)",
            fontsize=9,
        )
    fig.suptitle("Robot4Art - procedural composition examples (sim/composition.py)", fontsize=12, y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = os.path.join(OUT_DIR, "examples.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
