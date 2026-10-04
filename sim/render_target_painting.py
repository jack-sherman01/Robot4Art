"""Render a single clean image of the composition the robot was told to
paint (example_answers.EXAMPLE_ANSWERS) -- the "what the composer
generated" reference shown next to the Isaac Sim video on the website,
so a visitor can see the two side by side and check the robot actually
painted the same thing. No Isaac Sim needed.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from composition import strokes_from_brief
from example_answers import load_example_brief
from preview_compositions import CANVAS_BG, render_stroke

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "target_painting.png")


def main():
    brief, rationale = load_example_brief()
    strokes = strokes_from_brief(brief)
    rng = np.random.default_rng(brief.seed)
    print(f"[composition] style={brief.style} brush={brief.brush} pens={brief.pen_names}")
    print(f"[rationale] {rationale}")

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(CANVAS_BG)
    for s in strokes:
        render_stroke(ax, s, brief.brush, rng)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_facecolor(CANVAS_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_PATH, dpi=200, facecolor=CANVAS_BG)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
