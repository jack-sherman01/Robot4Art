"""Real LLM-driven composition: calls an actual Claude model to make the
creative decisions described in the research proposal's "Semantic-to-
Artistic Composition" section (path 2, "LLM-guided procedural
composition") -- which composition grammar, which two pen colors, which
brush/tool, and why -- instead of composition.py's deterministic
derive_brief() stand-in.

This replaces the stand-in specifically because no model *API key* was
available in earlier development -- but this machine already has the
Claude Code CLI itself authenticated for this session, which can be
invoked headlessly (`claude -p --restricted ...`) to get a real model
response without one. That's what this module does.

This is NOT something the public GitHub Pages web demo can call --
there is no backend, and the CLI is tied to this account's own
authenticated session, not something a visitor's browser can reach. It
is used for the one-off generation of the Isaac Sim example painting
(sim/example_answers.py) and could be used for an offline/batch kiosk
pipeline, but a live per-visitor version on the public site would need
its own hosted backend and API key -- a separate decision (cost,
latency ~5s/call, abuse/rate-limiting on a public page) from just
wiring this in here.

The actual stroke *geometry* (turning "arc_over_line, teal/purple" into
curve control points) stays procedural either way -- composition.py's
strokes_from_brief() -- since that's rendering math, not a creative
decision, and asking a model to emit precise floating-point curve
coordinates reliably is both unnecessary and fragile compared to just
letting it pick from a small, well-described set of options.
"""

from __future__ import annotations

import json
import subprocess

from composition import BRUSHES, STANDARD_PENS, STYLES, ArtisticBrief, PromptAnswers, _text_seed

import numpy as np

_GRAMMAR_DESCRIPTIONS = {
    "arc_over_line": (
        "a single confident rising gesture stroke closed by a sweeping arc, with a "
        "parallel echo stroke and small accent marks. Reads as bold, modernist, architectural."
    ),
    "nested_curves": (
        "several nested sweeping curves of increasing size, like layered brushwork, with "
        "a small flourish. Reads as calm, rippling, contemplative."
    ),
    "radiating_strokes": (
        "strokes radiating outward from a central point like petals, with a small broken "
        "arc at the center. Reads as energetic, bursting, joyful."
    ),
}

_BRUSH_DESCRIPTIONS = {
    "fine_pen": "thin, crisp, no taper",
    "marker": "medium width, slight taper",
    "brush": "thick, fully tapered gesture",
    "watercolor": "loose overlapping dabs",
}

_PEN_NAMES = [n for n in STANDARD_PENS if n != "black"]
STYLES_BY_GRAMMAR = {v["grammar"]: k for k, v in STYLES.items()}


def _build_prompt(answers: PromptAnswers) -> str:
    grammars = "\n".join(f"- {name}: {desc}" for name, desc in _GRAMMAR_DESCRIPTIONS.items())
    brushes = "\n".join(f"- {name} ({desc})" for name, desc in _BRUSH_DESCRIPTIONS.items())
    pens = ", ".join(_PEN_NAMES)
    return f"""You are the creative-composition engine for Robot4Art, an interactive installation where a robot arm paints an original abstract artwork based on a visitor's answers to a few personal questions. The artwork is restricted to at most 10 pen strokes, using a small set of standard pen colors (the robot physically swaps between a limited set of pens, not custom-mixed paint).

Visitor's answers:
- Favorite color: {answers.favorite_color}
- Favorite city: {answers.favorite_city}
- Dream/aspiration: {answers.dream}
- Mood: {answers.mood or "(not given)"}

Available composition grammars (choose exactly one):
{grammars}

Available pen colors (choose exactly two, "primary" and "secondary", must be from this list, spelled exactly as shown): {pens}. (Black is always available separately for fine accent marks -- do not choose it here.)

Available brush/tool types (choose exactly one):
{brushes}

Respond with ONLY a single JSON object, no markdown code fences, no other text, with exactly these keys:
{{"grammar": "...", "primary_pen": "...", "secondary_pen": "...", "brush": "...", "scale": 1.0, "rotation_deg": 0.0, "rationale": "2-4 sentences, first person plural 'we', explaining the creative choices and how they connect to the visitor's specific answers"}}

scale must be between 0.9 and 1.15. rotation_deg must be between -15 and 15."""


def derive_brief_llm(
    answers: PromptAnswers, model: str = "claude-sonnet-5", timeout: int = 90
) -> tuple[ArtisticBrief, str]:
    """Ask a real Claude model to make the creative decisions, instead of
    composition.py's deterministic derive_brief(). Returns (brief, the
    model's own rationale text -- not a template fill-in).

    Raises RuntimeError on anything unexpected (CLI failure, malformed
    JSON, a choice outside the offered options) rather than silently
    falling back -- this runs as a one-off/batch step with a human
    watching, not live behind a public page, so a loud failure beats a
    silent wrong answer.
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

    grammar = inner.get("grammar")
    if grammar not in STYLES_BY_GRAMMAR:
        raise RuntimeError(f"Model chose an unknown grammar: {grammar!r}")

    primary = str(inner.get("primary_pen", "")).strip().lower()
    secondary = str(inner.get("secondary_pen", "")).strip().lower()
    if primary not in _PEN_NAMES or secondary not in _PEN_NAMES or primary == secondary:
        raise RuntimeError(f"Model chose invalid/duplicate pens: {primary!r}, {secondary!r}")

    brush = inner.get("brush")
    if brush not in BRUSHES:
        raise RuntimeError(f"Model chose an unknown brush: {brush!r}")

    scale = float(inner.get("scale", 1.0))
    scale = min(max(scale, 0.85), 1.2)
    rotation_deg = float(inner.get("rotation_deg", 0.0))
    rotation_deg = min(max(rotation_deg, -20.0), 20.0)

    rationale = str(inner.get("rationale", "")).strip()
    if not rationale:
        raise RuntimeError("Model response had no rationale text.")

    seed = _text_seed(answers.favorite_color, answers.favorite_city, answers.dream, answers.mood)
    rng = np.random.default_rng(seed)
    center = (0.5 + float(rng.uniform(-0.09, 0.09)), 0.5 + float(rng.uniform(-0.07, 0.09)))
    palette = [STANDARD_PENS[primary], STANDARD_PENS[secondary], STANDARD_PENS["black"]]
    style = STYLES_BY_GRAMMAR[grammar]

    brief = ArtisticBrief(
        seed=seed,
        grammar=grammar,
        style=style,
        brush=brush,
        pen_names=[primary, secondary],
        palette=palette,
        scale=scale,
        rotation_deg=rotation_deg,
        center=center,
    )
    return brief, rationale
