"""Render a few example compositions to PNG for visual inspection.

Does not need Isaac Sim -- this only exercises composition.py, which is
pure Python/numpy/matplotlib. Useful for checking the grammar/style/brush
library (sim/composition.py) produces sane, on-canvas, <=10-stroke
artwork without waiting on a robot simulator.

Style (composition grammar: arc-over-line / nested curves / radiating
strokes) and brush (rendering tool: fine pen / marker / brush /
watercolor dabs, BRUSHES in composition.py) are independent choices --
the robot's pen/brush holder carries a few distinct tool types, not just
colors, so the examples below deliberately mix styles with
non-default brushes to show they compose freely. A uniform-width
centerline reads as a diagram, not a painting, and a single rendering
treatment can't honestly represent "fine pen" vs. "watercolor brush";
this is a rendering concern only -- the centerline points
stroke_plan/franka_paint_sim actually use for robot execution are
unaffected by any of it.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from composition import BRUSHES, PromptAnswers, compose, rationale_for

EXAMPLES = [
    PromptAnswers("blue", "Pittsburgh", "to build robots that help people", "curious", "modernist"),
    PromptAnswers("warm red", "Beijing", "to travel the world", "excited", "impressionist"),
    PromptAnswers("forest green", "Tokyo", "to write a novel", "calm", "ink_wash"),
    # Deliberately non-default style/brush pairings, to show they're
    # independent choices rather than one bundled "style" pick.
    PromptAnswers("sunset orange", "Paris", "to open a bakery", "cozy", "modernist", "fine_pen"),
    PromptAnswers("deep purple", "Shanghai", "to become a scientist", "determined", "ink_wash", "watercolor"),
    PromptAnswers("teal", "Austin", "to make music", "playful", "impressionist", "marker"),
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "compositions")

CANVAS_BG = "#F6F1E7"  # warm linen, not stark white


def tapered_polygon(points: np.ndarray, max_width: float, floor: float = 0.22) -> np.ndarray:
    """Build a filled brush-stroke outline from a centerline. floor=1.0
    is constant width (a fine pen/marker); lower floors taper more
    toward the ends (a loaded brush)."""
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
    widths = max_width * (floor + (1.0 - floor) * taper)

    left = points + perp * (widths / 2)[:, None]
    right = points - perp * (widths / 2)[:, None]
    return np.vstack([left, right[::-1]])


def render_stroke(ax, stroke, brush_key: str, rng: np.random.Generator):
    brush = BRUSHES[brush_key]
    if brush["render"] == "dabs":
        # A chain of overlapping dabs instead of one continuous shape --
        # impasto texture, closer to how a loaded watercolor brush
        # actually reads up close.
        points = stroke.points
        n = len(points)
        step = max(1, n // 7)
        base_r = 0.028 * stroke.width * brush["width_mult"]
        for i in range(0, n, step):
            x, y = points[i]
            r = base_r * rng.uniform(0.75, 1.15)
            ax.add_patch(plt.Circle((x, y), r, color=stroke.color, alpha=brush["opacity"], linewidth=0, zorder=3))
    else:
        max_width = 0.05 * stroke.width * brush["width_mult"]
        poly = tapered_polygon(stroke.points, max_width=max_width, floor=brush["floor"])
        ax.fill(poly[:, 0], poly[:, 1], color=stroke.color, linewidth=0, alpha=brush["opacity"], zorder=3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 10))
    fig.patch.set_facecolor("#FAF9F6")
    for ax, answers in zip(axes.flat, EXAMPLES):
        strokes, brief = compose(answers)
        rng = np.random.default_rng(brief.seed)
        for s in strokes:
            render_stroke(ax, s, brief.brush, rng)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor(CANVAS_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#DEDAD2")
        style_label = brief.style.replace("_", " ")
        brush_label = BRUSHES[brief.brush]["label"]
        ax.set_title(
            f"{answers.favorite_color} / {answers.favorite_city}\n{style_label} + {brush_label} · {'/'.join(brief.pen_names)} ({len(strokes)} strokes)",
            fontsize=9,
        )
        print(f"--- {style_label} + {brush_label} ({answers.favorite_color}/{answers.favorite_city}) ---")
        print(rationale_for(answers, brief))
        print()
    fig.suptitle("Robot4Art - procedural composition examples (sim/composition.py)", fontsize=12, y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = os.path.join(OUT_DIR, "examples.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
