"""One-off script: ask a real Claude model to directly paint the example
composition (example_answers.EXAMPLE_ANSWERS) -- not pick from a
template -- and cache the result to example_brief.json.

Run this manually when EXAMPLE_ANSWERS changes, not automatically on
every pipeline run -- each call costs real money and tens of seconds,
and is a genuine model decision worth reviewing before committing to it.
Everything downstream (franka_paint_sim.py, franka_paint_video.py,
render_target_painting.py) loads the cached painting via
example_answers.load_example_painting() instead of calling the model
again.

Usage:
    python3 sim/generate_example_brief.py
"""

import json
import os

from example_answers import EXAMPLE_ANSWERS
from llm_composer import derive_painting_llm

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_brief.json")


def main():
    strokes, brush, rationale = derive_painting_llm(EXAMPLE_ANSWERS)
    record = {
        "brush": brush,
        "rationale": rationale,
        "strokes": [
            {"name": s.name, "color": s.color, "points": s.points.tolist(), "width": s.width} for s in strokes
        ],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(f"brush={brush} strokes={len(strokes)} colors={sorted({s.color for s in strokes})}")
    print(f"rationale: {rationale}")


if __name__ == "__main__":
    main()
