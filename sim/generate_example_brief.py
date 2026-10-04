"""One-off script: ask a real Claude model to compose the example
painting (example_answers.EXAMPLE_ANSWERS), and cache the result to
example_brief.json.

Run this manually when EXAMPLE_ANSWERS changes, not automatically on
every pipeline run -- each call costs real money and ~5-25s, and is a
genuine model decision worth reviewing before committing to it, not
something to silently re-roll on every `franka_paint_sim.py` run.
Everything downstream (franka_paint_sim.py, franka_paint_video.py,
render_target_painting.py) loads the cached brief via
example_answers.load_example_brief() instead of calling the model again.

Usage:
    python3 sim/generate_example_brief.py
"""

import dataclasses
import json
import os

from example_answers import EXAMPLE_ANSWERS
from llm_composer import derive_brief_llm

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_brief.json")


def main():
    brief, rationale = derive_brief_llm(EXAMPLE_ANSWERS)
    record = dataclasses.asdict(brief)
    record["rationale"] = rationale
    with open(OUT_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(f"grammar={brief.grammar} style={brief.style} brush={brief.brush} pens={brief.pen_names}")
    print(f"rationale: {rationale}")


if __name__ == "__main__":
    main()
