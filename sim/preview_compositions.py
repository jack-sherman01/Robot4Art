"""Render a few example compositions to PNG for visual inspection.

Does not need Isaac Sim -- this only exercises composition.py, which is
pure Python/numpy/matplotlib. Useful for checking the grammar library
(sim/composition.py) produces sane, on-canvas, <=5-stroke artwork without
waiting on a robot simulator.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, answers in zip(axes.flat, EXAMPLES):
        strokes, brief = compose(answers)
        for s in strokes:
            ax.plot(s.points[:, 0], s.points[:, 1], "-", color=s.color, linewidth=3, solid_capstyle="round")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("#FAFAF8")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{answers.favorite_color} / {answers.favorite_city}\n{brief.grammar} ({len(strokes)} strokes)",
            fontsize=9,
        )
    fig.suptitle("Robot4Art - procedural composition examples (sim/composition.py)", fontsize=12, y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = os.path.join(OUT_DIR, "examples.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
