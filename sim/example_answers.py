"""The one example visitor used everywhere a fixed, representative
composition is needed (Isaac Sim validation, the painting video, the
static "what the composer generated" reference image) -- standing in for
live kiosk input so all three stay in sync with each other."""

from __future__ import annotations

import json
import os

from composition import ArtisticBrief, PromptAnswers

EXAMPLE_ANSWERS = PromptAnswers(
    favorite_color="teal",
    favorite_city="Pittsburgh",
    dream="to build robots that help people",
    mood="curious",
)

_BRIEF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_brief.json")


def load_example_brief() -> tuple[ArtisticBrief, str]:
    """Load the real Claude-generated brief for EXAMPLE_ANSWERS, cached by
    generate_example_brief.py so every consumer (Isaac Sim validation, the
    robot video, the static reference image) uses the exact same model
    decision instead of each re-calling the model (slow, costs money, and
    could disagree with itself between runs)."""
    with open(_BRIEF_PATH) as f:
        record = json.load(f)
    rationale = record.pop("rationale")
    record["center"] = tuple(record["center"])
    return ArtisticBrief(**record), rationale
