"""Real LLM-driven painting: a real Claude model directly composes the
artwork -- which strokes, what shape each one is, what color, and why --
instead of composition.py's hand-written procedural "grammars"
(arc_over_line / nested_curves / radiating_strokes). Those grammars pick
a fixed shape family in code and let the model only choose between a
handful of named options; this module asks the model to author the
actual stroke geometry itself, so the painting is genuinely LLM-made
rather than a human-written template with an LLM-picked label on it.

Invokes the already-authenticated Claude Code CLI headlessly
(`claude -p --restricted ...`), so no separate model API key is needed
in this dev environment. See composition.py's module docstring for why
this isn't wired into the public GitHub Pages web demo (no backend
there, so no real model call is possible from a static page).

The model is only asked for a point in [0,1] canvas space per stroke
vertex (handful of points per stroke, not a dense path) -- turning those
into an actual smooth line is pure rendering math
(stroke_plan.catmull_rom), same reasoning composition.py's
strokes_from_brief uses: no need to burden the model with emitting a
dense, precise curve when a few well-chosen points plus a standard
smoothing spline do the same job more reliably.
"""

from __future__ import annotations

import json
import subprocess

from composition import BRUSHES, MAX_STROKES, STANDARD_PENS, PromptAnswers
from stroke_plan import Stroke, catmull_rom

import numpy as np

_PEN_NAMES = list(STANDARD_PENS.keys())  # includes "black"

_BRUSH_DESCRIPTIONS = {
    "fine_pen": "thin, crisp, no taper",
    "marker": "medium width, slight taper",
    "brush": "thick, fully tapered gesture",
    "watercolor": "loose overlapping dabs",
}

_CANVAS_MARGIN = 0.08


def _build_prompt(answers: PromptAnswers) -> str:
    pens = ", ".join(_PEN_NAMES)
    brushes = "\n".join(f"- {name} ({desc})" for name, desc in _BRUSH_DESCRIPTIONS.items())
    return f"""You are an abstract painter. A robot arm will physically paint exactly what you design here, stroke by stroke, so design a real composition yourself -- don't just pick from a template.

The visitor answered a few personal prompts:
- Favorite color: {answers.favorite_color}
- Favorite city: {answers.favorite_city}
- Dream/aspiration: {answers.dream}
- Mood: {answers.mood or "(not given)"}

Canvas: a normalized square, x and y both in [0, 1], (0, 0) at the bottom-left. Keep all points within [{_CANVAS_MARGIN}, {1 - _CANVAS_MARGIN}] of each axis.

Constraints (the robot physically has these, not a stylistic choice):
- At most {MAX_STROKES} strokes total.
- Each stroke is a smooth path through 2 to 6 points you choose -- pick the points that define the shape you want; they'll be smoothed into a curve automatically, you don't need to output a dense path.
- Each stroke uses exactly one pen color, from: {pens}.
- Pick exactly one brush/tool for the whole piece (it physically can't be swapped mid-painting):
{brushes}

Design an actual abstract composition -- real visual balance, intentional negative space, a clear focal gesture plus supporting strokes, varied stroke lengths and directions and weights -- not a generic diagram, not a symmetric pattern, not one shape repeated with rotations. Let the visitor's answers genuinely shape the composition (their mood, their dream, the feeling of their color and city), not just which colors you pick.

Respond with ONLY a single JSON object, no markdown fences, no other text:
{{
  "brush": "<one of the brush names above>",
  "strokes": [
    {{"color": "<one of the pen names above>", "points": [[x, y], [x, y], ...], "width": <0.3 to 1.0, relative weight of this stroke>}},
    ...
  ],
  "rationale": "2-4 sentences, first person plural 'we', explaining the actual composition you designed and how it connects to the visitor's specific answers"
}}"""


def derive_painting_llm(
    answers: PromptAnswers, model: str = "claude-sonnet-5", timeout: int = 120
) -> tuple[list[Stroke], str, str]:
    """Ask a real Claude model to directly author the painting -- every
    stroke's shape and color, not just a choice between code-written
    templates. Returns (strokes, brush_name, rationale).

    Raises RuntimeError on anything unexpected (CLI failure, malformed
    JSON, a choice outside the offered options) rather than silently
    falling back to a procedural grammar -- this runs as a one-off/batch
    step with a human watching, not live behind a public page, so a loud
    failure beats silently substituting a non-LLM painting.
    """
    prompt = _build_prompt(answers)
    try:
        proc = subprocess.run(
            ["claude", "-p", "--restricted", "--model", model, "--output-format", "json", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            input="",
        )
    except FileNotFoundError as e:
        raise RuntimeError("`claude` CLI not found on PATH -- can't make a real model call here.") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"claude CLI call timed out after {timeout}s") from e

    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI exited {proc.returncode}: {proc.stderr[:500]}")

    try:
        envelope = json.loads(proc.stdout)
        inner = json.loads(envelope["result"])
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Could not parse claude CLI output as the expected JSON: {proc.stdout[:500]}") from e

    return _parse_painting_response(inner)


def _parse_painting_response(inner: dict) -> tuple[list[Stroke], str, str]:
    """Validate and convert one already-parsed model response into
    (strokes, brush_name, rationale). Split out from derive_painting_llm
    so a response obtained some other way (e.g. a retry, or logged from a
    prior call) can be re-validated without paying for another call."""
    brush = inner.get("brush")
    if brush not in BRUSHES:
        raise RuntimeError(f"Model chose an unknown brush: {brush!r}")

    raw_strokes = inner.get("strokes")
    if not isinstance(raw_strokes, list) or not (1 <= len(raw_strokes) <= MAX_STROKES):
        raise RuntimeError(f"Model returned {len(raw_strokes) if isinstance(raw_strokes, list) else raw_strokes!r} strokes, need 1-{MAX_STROKES}")

    strokes = []
    for i, raw in enumerate(raw_strokes):
        color_name = str(raw.get("color", "")).strip().lower()
        if color_name not in STANDARD_PENS:
            raise RuntimeError(f"Stroke {i} has an unknown color: {color_name!r}")

        raw_points = raw.get("points")
        if not isinstance(raw_points, list) or not (2 <= len(raw_points) <= 8):
            raise RuntimeError(f"Stroke {i} has {len(raw_points) if isinstance(raw_points, list) else raw_points!r} points, need 2-8")
        try:
            points = np.array([[float(p[0]), float(p[1])] for p in raw_points])
        except (TypeError, ValueError, IndexError) as e:
            raise RuntimeError(f"Stroke {i} has malformed points: {raw_points!r}") from e
        points = np.clip(points, _CANVAS_MARGIN, 1 - _CANVAS_MARGIN)

        width = float(raw.get("width", 0.7))
        width = min(max(width, 0.3), 1.0)

        strokes.append(
            Stroke(f"stroke_{i}", STANDARD_PENS[color_name], catmull_rom(points, n_per_segment=10), width=width)
        )

    rationale = str(inner.get("rationale", "")).strip()
    if not rationale:
        raise RuntimeError("Model response had no rationale text.")

    return strokes, brush, rationale
