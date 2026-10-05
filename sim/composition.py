"""Semantic-to-artwork composition: turn a visitor's kiosk answers into a
<=20-stroke artwork plan.

Implements the "LLM-guided procedural composition" path described in the
research proposal (private/proposal_en.tex, Sec. "Semantic-to-Artistic
Composition", path 2): a small library of parametric composition
*grammars* ("arc-over-line, nested curves, radiating strokes") is filled
in with color, scale, and placement derived from the visitor's answers.

No LLM is wired in yet -- there is no model API key available in this
environment. The "which grammar, what parameters" choice that the
proposal assigns to an LLM is made here by a deterministic function of
the visitor's answers instead (PromptAnswers -> ArtisticBrief). This
keeps the rest of the pipeline (brief -> grammar -> Stroke list -> sim/
robot execution) exactly as it will be once a real model is swapped in
for `derive_brief`: everything downstream only depends on the
ArtisticBrief shape, not on how it was produced.

The other path in the proposal (direct CLIPDraw-style stroke
optimization against a vision-language embedding) needs a real model and
is not implemented here.

Colors are drawn from a small, fixed set of common pen/marker colors
(STANDARD_PENS below) -- not freely generated -- since the robot
physically holds a small set of interchangeable pens/markers, not
custom-mixed paint. Picked by nearest hue to the visitor's color answer;
the fine accent work always uses the black pen, which is standard in any
real pen holder.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from stroke_plan import Stroke, _arc, _line, _quadratic_bezier

# Stroke budget: the physical robot holds this many strokes' worth of
# execution time/pen changes (Sec. "Stroke Planning and Vectorization").
MAX_STROKES = 20

# A handful of named colors a visitor might type, mapped to a base hue in
# [0, 1), used to pick the nearest common pen color below. Anything
# unrecognized falls back to hashing the raw text to a hue.
_NAMED_HUES = {
    "red": 0.00, "orange": 0.08, "amber": 0.11, "yellow": 0.15, "gold": 0.13,
    "lime": 0.22, "green": 0.33, "olive": 0.19, "teal": 0.50, "cyan": 0.52,
    "sky": 0.56, "blue": 0.60, "navy": 0.62, "indigo": 0.68, "purple": 0.75,
    "violet": 0.78, "lavender": 0.72, "magenta": 0.83, "pink": 0.90,
    "rose": 0.95, "brown": 0.07, "black": 0.60, "white": 0.60, "gray": 0.60,
    "grey": 0.60,
}

# A realistic, physically-stockable set of common pen/marker colors --
# e.g. a standard whiteboard-marker or Sharpie multipack. Every
# composition's colors come from this list; nothing is custom-mixed, so
# any generated artwork is paintable with pens a robot could plausibly
# have loaded. "black" has no hue entry: it isn't picked by hue-matching,
# it's always available as the accent/fine-detail pen, like in a real set.
STANDARD_PENS = {
    "black": "#232323",
    "red": "#C0392B",
    "orange": "#D2691E",
    "yellow": "#D4A017",
    "green": "#2E7D4F",
    "teal": "#1F7A72",
    "blue": "#2255A4",
    "purple": "#6B3FA0",
    "pink": "#C0527A",
    "brown": "#6F4E2E",
}

_PEN_HUES = {
    "red": 0.00, "orange": 0.07, "yellow": 0.14, "green": 0.36, "teal": 0.49,
    "blue": 0.61, "purple": 0.76, "pink": 0.92, "brown": 0.08,
}

# Visitor-facing style choices (Sec. 4.1, "Interaction Layer"), each tied
# to one grammar. Previously the grammar was picked for the visitor by a
# hash of their answers; letting them choose the style directly is both
# more satisfying to interact with and more reliable for aesthetic
# quality than a 1-in-3 random assignment.
STYLES = {
    "modernist": {"label": "Modernist Gesture", "grammar": "arc_over_line"},
    "impressionist": {"label": "Impressionist Bloom", "grammar": "radiating_strokes"},
    "ink_wash": {"label": "Ink Wash Minimal", "grammar": "nested_curves"},
}
DEFAULT_STYLE = "modernist"

# The robot's pen/brush holder carries a few distinct tool types, not just
# a few colors -- a real design detail from Sec. "Robot-Agnostic
# Execution" ("a multi-slot pen/brush holder ... a small set of common
# pen colors"). Each tool renders differently (width profile, taper,
# opacity), independent of which composition style/grammar is chosen, so
# the visitor can pick style and tool separately. "floor" is the
# minimum width as a fraction of max width (1.0 = constant width, no
# taper; lower = tapers more toward the stroke's ends).
BRUSHES = {
    "fine_pen": {"label": "Fine Pen", "article": "a", "render": "line", "width_mult": 0.45, "floor": 0.85, "opacity": 0.98},
    "marker": {"label": "Marker", "article": "a", "render": "line", "width_mult": 0.75, "floor": 0.55, "opacity": 0.92},
    "brush": {"label": "Brush", "article": "a", "render": "line", "width_mult": 1.0, "floor": 0.22, "opacity": 0.95},
    "watercolor": {"label": "Watercolor Dabs", "article": "", "render": "dabs", "width_mult": 1.0, "floor": 1.0, "opacity": 0.85},
}
# A tool that suits each style if the visitor doesn't override it --
# matches what shipped before the tool became independently selectable.
_DEFAULT_BRUSH_FOR_STYLE = {"modernist": "brush", "impressionist": "watercolor", "ink_wash": "fine_pen"}

# A short, templated explanation of *why* the piece looks the way it does,
# connecting the generated artwork back to the visitor's own answers.
# Stands in for an LLM-written rationale for the same reason derive_brief
# is deterministic rather than a real model call (Sec. "Semantic-to-
# Artistic Composition"): no model API access in this environment yet.
_RATIONALE_TEMPLATES = {
    "arc_over_line": (
        'A single {primary} gesture rises across the canvas and closes with a {secondary} '
        'arc -- a confident line for a dream like "{dream}." The fine black marks scattered '
        "near it carry {mood_article} {mood_phrase} energy, and the two pens were picked to "
        'echo "{color}" and the feel of {city}.'
    ),
    "nested_curves": (
        'Layers of {primary} and {secondary} curves nest inside one another, each a little '
        'larger than the last -- like ripples spreading outward from "{dream}." A small '
        "flourish signs off the outermost curve, and the fine black accents nearby are "
        '{mood_article} {mood_phrase} touch, in colors drawn from "{color}" and {city}.'
    ),
    "radiating_strokes": (
        'Strokes in {primary} and {secondary} radiate outward from a single point, like '
        'petals opening -- a burst of energy for "{dream}." The small black marks at their '
        "tips add {mood_article} {mood_phrase} rhythm, and the palette traces back to "
        '"{color}" and a touch of {city}.'
    ),
}


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def rationale_for(answers: "PromptAnswers", brief: "ArtisticBrief") -> str:
    """A short, human-readable explanation tying the generated artwork
    back to the visitor's own answers -- displayed alongside the piece
    as it's painted, not just the finished result."""
    template = _RATIONALE_TEMPLATES[brief.grammar] + " Laid down with {brush_article}{brush_label}."
    mood_phrase = answers.mood.strip() or "calm"
    brush = BRUSHES[brief.brush]
    return template.format(
        primary=brief.pen_names[0],
        secondary=brief.pen_names[1],
        dream=answers.dream.strip(),
        mood_phrase=mood_phrase,
        mood_article=_article(mood_phrase),
        color=answers.favorite_color.strip(),
        city=answers.favorite_city.strip(),
        brush_article=(brush["article"] + " ") if brush["article"] else "",
        brush_label=brush["label"].lower(),
    )


def _text_seed(*parts: str) -> int:
    """A stable integer seed for a tuple of free-text answers."""
    joined = "␟".join(p.strip().lower() for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _hue_from_text(text: str) -> float:
    text = text.strip().lower()
    for name, hue in _NAMED_HUES.items():
        if name in text:
            return hue
    return (_text_seed(text) % 360) / 360.0


def _hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def _pick_pens(hue: float) -> tuple[str, str]:
    """Nearest pen to the visitor's hue, plus a second pen picked for
    contrast (closest to a third-of-the-wheel away from the first)."""
    names = list(_PEN_HUES.keys())
    primary = min(names, key=lambda n: _hue_distance(_PEN_HUES[n], hue))
    remaining = [n for n in names if n != primary]
    target = (_PEN_HUES[primary] + 1.0 / 3.0) % 1.0
    secondary = min(remaining, key=lambda n: _hue_distance(_PEN_HUES[n], target))
    return primary, secondary


@dataclass
class PromptAnswers:
    """One visitor's kiosk answers (Sec. 4.1, "Interaction Layer")."""

    favorite_color: str
    favorite_city: str
    dream: str
    mood: str = ""
    style: str = DEFAULT_STYLE  # one of STYLES' keys
    brush: str = ""  # one of BRUSHES' keys, or "" to use the style's default tool


@dataclass
class ArtisticBrief:
    """The output of the semantic-composition step: enough to pick and
    parameterize a stroke grammar, but not yet a stroke plan itself."""

    seed: int
    grammar: str  # which grammar function to use
    style: str = DEFAULT_STYLE  # the visitor-facing style name (STYLES key)
    brush: str = ""  # the resolved tool name (BRUSHES key), independent of style
    pen_names: list[str] = field(default_factory=list)  # [primary, secondary] from STANDARD_PENS
    palette: list[str] = field(default_factory=list)  # [primary, secondary, "black"] hex
    scale: float = 1.0  # 0.9-1.15, overall composition size
    rotation_deg: float = 0.0  # overall composition rotation
    center: tuple = (0.5, 0.5)  # off-center composition anchor


def derive_brief(answers: PromptAnswers) -> ArtisticBrief:
    """Deterministic stand-in for the LLM call described in the proposal.
    Same answers always produce the same brief; different answers almost
    always produce a different one (different grammar and/or pens)."""
    seed = _text_seed(answers.favorite_color, answers.favorite_city, answers.dream, answers.mood)
    rng = np.random.default_rng(seed)

    hue = _hue_from_text(answers.favorite_color)
    primary_pen, secondary_pen = _pick_pens(hue)
    palette = [STANDARD_PENS[primary_pen], STANDARD_PENS[secondary_pen], STANDARD_PENS["black"]]

    style = answers.style if answers.style in STYLES else DEFAULT_STYLE
    grammar = STYLES[style]["grammar"]
    brush = answers.brush if answers.brush in BRUSHES else _DEFAULT_BRUSH_FOR_STYLE[style]
    scale = float(rng.uniform(0.9, 1.15))
    rotation_deg = float(rng.uniform(-15.0, 15.0))
    # A gentle off-center anchor (rule-of-thirds-ish) instead of always
    # dead-centering the composition on the canvas.
    center = (0.5 + float(rng.uniform(-0.09, 0.09)), 0.5 + float(rng.uniform(-0.07, 0.09)))

    return ArtisticBrief(
        seed=seed,
        grammar=grammar,
        style=style,
        brush=brush,
        pen_names=[primary_pen, secondary_pen],
        palette=palette,
        scale=scale,
        rotation_deg=rotation_deg,
        center=center,
    )


def _rotate(points: np.ndarray, center: np.ndarray, deg: float) -> np.ndarray:
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return (points - center) @ rot.T + center


def _clip01(points: np.ndarray, margin: float = 0.08) -> np.ndarray:
    return np.clip(points, margin, 1.0 - margin)


def _accent_dabs(
    rng: np.random.Generator,
    anchors: list[np.ndarray],
    color: str,
    n: int,
    scale: float,
    name_prefix: str = "accent",
) -> list[Stroke]:
    """Small comma-shaped accent marks anchored close to existing
    structural points, not scattered randomly -- like the fine black-pen
    flicks a painter adds right at their own brushwork, not confetti."""
    strokes = []
    for i in range(n):
        anchor = anchors[rng.integers(0, len(anchors))]
        jitter = np.array([rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02)]) * scale
        origin = anchor + jitter
        angle_deg = rng.uniform(0, 360)
        radius = scale * rng.uniform(0.018, 0.032)
        sweep = rng.uniform(55, 95) * (1 if rng.random() > 0.5 else -1)
        strokes.append(
            Stroke(
                f"{name_prefix}_{i}",
                color,
                _clip01(_arc(origin, radius, angle_deg, angle_deg + sweep, n=6)),
                width=rng.uniform(0.2, 0.3),
            )
        )
    return strokes


def _grammar_arc_over_line(rng: np.random.Generator, palette: list[str], scale: float, c: np.ndarray) -> list[Stroke]:
    """A confident rising gesture closed by a sweeping arc, layered echo
    strokes for painterly depth, a smaller counter-gesture for balance,
    and a denser scatter of fine accent dabs."""
    half = 0.28 * scale
    p0 = c + np.array([-half, -half * rng.uniform(0.75, 1.0)])
    p1 = c + np.array([half * rng.uniform(0.9, 1.05), half * rng.uniform(0.85, 1.05)])
    mid = (p0 + p1) / 2 + np.array([rng.uniform(-0.03, 0.03), rng.uniform(0.02, 0.07)]) * scale
    arc_center = p1 - np.array([0.01, 0.11 * scale])

    strokes = [
        Stroke("rising_gesture", palette[0], _clip01(_quadratic_bezier(p0, mid, p1, n=18)), width=1.0),
        Stroke("closing_arc", palette[1], _clip01(_arc(arc_center, 0.11 * scale, -15, 150, n=20)), width=0.72),
    ]

    # Several thinner echoes roughly parallel to the main gesture, each
    # offset and shrinking a bit more -- like repeated brushstrokes built
    # up in a real gesture painting, not one lone duplicate.
    anchors = [p0, p1, mid, arc_center]
    n_echoes = 3
    prev_p0, prev_p1 = p0, p1
    for i in range(n_echoes):
        offset = np.array([rng.uniform(-0.03, 0.03), rng.uniform(0.07, 0.11)]) * scale
        echo_p0, echo_p1 = prev_p0 + offset, prev_p1 + offset * 0.6
        echo_mid = (echo_p0 + echo_p1) / 2 + np.array([0.0, rng.uniform(0.02, 0.05)]) * scale
        color = palette[0] if i % 2 == 0 else palette[1]
        strokes.append(
            Stroke(
                f"echo_gesture_{i}",
                color,
                _clip01(_quadratic_bezier(echo_p0, echo_mid, echo_p1, n=16)),
                width=0.45 - 0.1 * i,
            )
        )
        anchors += [echo_p0, echo_p1]
        prev_p0, prev_p1 = echo_p0, echo_p1

    # A smaller counter-gesture angled well away from the main gesture
    # (roughly perpendicular, not parallel) and anchored off to one side,
    # for real compositional contrast instead of a second near-duplicate
    # line buried in the same bundle.
    main_dir = (p1 - p0)
    main_dir = main_dir / np.linalg.norm(main_dir)
    perp_dir = np.array([-main_dir[1], main_dir[0]])
    counter_sign = 1.0 if rng.random() > 0.5 else -1.0
    theta = np.radians(rng.uniform(75, 105) * counter_sign)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    counter_dir = rot @ main_dir
    counter_len = half * rng.uniform(0.7, 0.95)
    anchor = c + perp_dir * (0.14 * scale) * counter_sign
    cq0 = anchor - counter_dir * counter_len * 0.35
    cq1 = anchor + counter_dir * counter_len * 0.65
    cq_mid = (cq0 + cq1) / 2 + perp_dir * rng.uniform(-0.02, 0.02) * scale
    strokes.append(Stroke("counter_gesture", palette[1], _clip01(_quadratic_bezier(cq0, cq_mid, cq1, n=14)), width=0.55))
    anchors += [cq0, cq1]

    accent_origin = c + np.array([rng.uniform(-0.28, -0.12), rng.uniform(-0.14, -0.02)]) * scale
    accent_end = accent_origin + np.array([rng.uniform(0.12, 0.2), rng.uniform(-0.03, 0.03)]) * scale
    strokes.append(Stroke("horizon_accent", palette[2], _clip01(_line(accent_origin, accent_end, n=8)), width=0.4))

    strokes += _accent_dabs(rng, anchors, palette[2], MAX_STROKES - len(strokes), scale)
    return strokes


def _grammar_nested_curves(rng: np.random.Generator, palette: list[str], scale: float, c: np.ndarray) -> list[Stroke]:
    """Nested sweeping curves of increasing size, like layered brushwork,
    short connecting tendrils between a few of them, flourishes at both
    the innermost and outermost tips, and fine accent dabs."""
    strokes = []
    n_curves = 6
    colors = [palette[0], palette[1]] * (n_curves // 2 + 1)
    anchors = []
    curve_tips = []  # (p0, p2) per curve, for tendrils and flourish placement
    # One shared "lean" and control-depth ratio for every curve in the
    # family, so they actually read as concentric/nested -- only
    # growing in size -- instead of each wobbling independently.
    lean = rng.uniform(-0.15, 0.15)
    depth_ratio = rng.uniform(0.55, 0.85)
    for i in range(n_curves):
        r = (0.08 + 0.05 * i) * scale
        p0 = c + np.array([-r, r * (0.25 + lean)])
        p1 = c + np.array([lean * r * 0.4, -r * depth_ratio])
        p2 = c + np.array([r, r * (0.3 - lean)])
        curve_tips.append((p0, p2))
        anchors += [p0, p2]
        width = 1.0 - 0.75 * (i / (n_curves - 1))
        strokes.append(
            Stroke(f"nested_curve_{i}", colors[i], _clip01(_quadratic_bezier(p0, p1, p2, n=16)), width=width)
        )

    # Short tendrils linking alternating curves -- like a painter
    # connecting layered brushwork with quick fine-pen touches, not just
    # isolated nested arcs.
    for i in range(1, n_curves, 2):
        p0 = curve_tips[i - 1][1]
        p1 = curve_tips[i][1]
        strokes.append(Stroke(f"tendril_{i}", palette[2], _clip01(_line(p0, p1, n=6)), width=0.3))

    for tag, (p0, p2) in (("inner", curve_tips[0]), ("outer", curve_tips[-1])):
        flourish_dir = np.array([rng.uniform(0.6, 1.0), rng.uniform(-0.1, 0.25)]) * (1 if tag == "outer" else -1)
        flourish_dir /= np.linalg.norm(flourish_dir)
        flourish_len = 0.09 * scale
        anchor = p2 if tag == "outer" else p0
        dash_origin = anchor - flourish_dir * flourish_len * 0.2
        dash_end = anchor + flourish_dir * flourish_len
        strokes.append(Stroke(f"flourish_{tag}", palette[2], _clip01(_line(dash_origin, dash_end, n=8)), width=0.42))

    strokes += _accent_dabs(rng, anchors, palette[2], MAX_STROKES - len(strokes), scale)
    return strokes


def _grammar_radiating_strokes(rng: np.random.Generator, palette: list[str], scale: float, c: np.ndarray) -> list[Stroke]:
    """Strokes radiating outward with organic (not perfectly even)
    spacing and length -- most straight rays, a few curved into petal
    shapes for variety -- a broken double arc near the center, and fine
    accent dabs."""
    strokes = []
    anchors = []
    n_rays = 10
    colors = [palette[0], palette[1]] * (n_rays // 2 + 1)
    base_angle = rng.uniform(0, 360)
    spread = 360.0 / n_rays
    for i in range(n_rays):
        angle = np.radians(base_angle + i * spread + rng.uniform(-spread * 0.22, spread * 0.22))
        length = (0.11 + 0.11 * rng.uniform(0.4, 1.0)) * scale
        inner = (0.03 + 0.02 * rng.uniform(0, 1)) * scale
        direction = np.array([np.cos(angle), np.sin(angle)])
        p0 = c + inner * direction
        p1 = c + length * direction
        width = 0.85 if i % 2 == 0 else 0.55
        anchors.append(p1)
        if i % 3 == 2:
            # Bend every third ray into a gentle petal curve, so the
            # burst isn't made of perfectly identical straight spokes.
            perp = np.array([-direction[1], direction[0]])
            bend = perp * length * rng.uniform(0.12, 0.22) * (1 if rng.random() > 0.5 else -1)
            mid = (p0 + p1) / 2 + bend
            points = _quadratic_bezier(p0, mid, p1, n=12)
        else:
            points = _line(p0, p1, n=10)
        strokes.append(Stroke(f"ray_{i}", colors[i], _clip01(points), width=width))

    strokes.append(
        Stroke(
            "center_arc",
            palette[2],
            _clip01(_arc(c, 0.065 * scale, rng.uniform(0, 60), rng.uniform(220, 300), n=14)),
            width=0.45,
        )
    )
    strokes.append(
        Stroke(
            "center_arc_inner",
            palette[2],
            _clip01(_arc(c, 0.035 * scale, rng.uniform(120, 200), rng.uniform(260, 340), n=10)),
            width=0.32,
        )
    )
    strokes += _accent_dabs(rng, anchors, palette[2], MAX_STROKES - len(strokes), scale)
    return strokes


_GRAMMAR_FNS = {
    "arc_over_line": _grammar_arc_over_line,
    "nested_curves": _grammar_nested_curves,
    "radiating_strokes": _grammar_radiating_strokes,
}


def strokes_from_brief(brief: ArtisticBrief) -> list[Stroke]:
    """Render an already-decided ArtisticBrief into actual stroke
    geometry. Split out from compose() so a brief produced some other
    way (e.g. llm_composer.derive_brief_llm, a real model call) can
    reuse the same procedural rendering -- this part is just math
    (turning "arc_over_line, teal/purple, scale 1.0" into curve control
    points), not a creative decision, so it stays deterministic either
    way."""
    rng = np.random.default_rng(brief.seed)
    center = np.array(brief.center)
    strokes = _GRAMMAR_FNS[brief.grammar](rng, brief.palette, brief.scale, center)

    if abs(brief.rotation_deg) > 1e-6:
        strokes = [
            Stroke(s.name, s.color, _clip01(_rotate(s.points, center, brief.rotation_deg)), width=s.width)
            for s in strokes
        ]

    assert len(strokes) <= MAX_STROKES, f"grammar {brief.grammar!r} produced {len(strokes)} strokes, budget is {MAX_STROKES}"
    return strokes


def compose(answers: PromptAnswers) -> tuple[list[Stroke], ArtisticBrief]:
    """Visitor answers -> (<=MAX_STROKES-stroke artwork plan, the brief that produced it).

    Uses the deterministic stand-in (derive_brief) for the creative
    decision. See llm_composer.py for a version that asks a real model
    instead."""
    brief = derive_brief(answers)
    return strokes_from_brief(brief), brief
