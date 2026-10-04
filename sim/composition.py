"""Semantic-to-artwork composition: turn a visitor's kiosk answers into a
5-stroke artwork plan.

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
"""

from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass, field

import numpy as np

from stroke_plan import Stroke, _arc, _line, _quadratic_bezier

# A handful of named colors a visitor might type, mapped to a base hue in
# [0, 1). Anything not recognized falls back to hashing the raw text to a
# hue, so every answer still produces a distinct, stable palette.
_NAMED_HUES = {
    "red": 0.00, "orange": 0.08, "amber": 0.11, "yellow": 0.15, "gold": 0.13,
    "lime": 0.22, "green": 0.33, "teal": 0.50, "cyan": 0.52, "sky": 0.56,
    "blue": 0.60, "indigo": 0.68, "purple": 0.75, "violet": 0.78, "magenta": 0.83,
    "pink": 0.90, "rose": 0.95, "brown": 0.07, "black": 0.0, "white": 0.0,
    "gray": 0.0, "grey": 0.0,
}


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


def _hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return "#{:02X}{:02X}{:02X}".format(round(r * 255), round(g * 255), round(b * 255))


@dataclass
class PromptAnswers:
    """One visitor's kiosk answers (Sec. 4.1, "Interaction Layer")."""

    favorite_color: str
    favorite_city: str
    dream: str
    mood: str = ""


@dataclass
class ArtisticBrief:
    """The output of the semantic-composition step: enough to pick and
    parameterize a stroke grammar, but not yet a stroke plan itself."""

    seed: int
    grammar: str  # which grammar function to use
    palette: list[str] = field(default_factory=list)  # 2-3 hex colors
    scale: float = 1.0  # 0.7-1.3, overall composition size
    rotation_deg: float = 0.0  # overall composition rotation


_GRAMMARS = ("arc_over_line", "nested_curves", "radiating_strokes")


def derive_brief(answers: PromptAnswers) -> ArtisticBrief:
    """Deterministic stand-in for the LLM call described in the proposal.
    Same answers always produce the same brief; different answers almost
    always produce a different one (different grammar and/or palette)."""
    seed = _text_seed(answers.favorite_color, answers.favorite_city, answers.dream, answers.mood)
    rng = np.random.default_rng(seed)

    base_hue = _hue_from_text(answers.favorite_color)
    city_hue_shift = (_text_seed(answers.favorite_city) % 1000) / 1000.0 * 0.12
    palette = [
        _hex(base_hue, 0.65, 0.78),
        _hex(base_hue + 0.08 + city_hue_shift, 0.55, 0.60),
        "#1B1B1B",
    ]

    grammar = _GRAMMARS[seed % len(_GRAMMARS)]
    scale = float(rng.uniform(0.85, 1.15))
    rotation_deg = float(rng.uniform(-20.0, 20.0))

    return ArtisticBrief(seed=seed, grammar=grammar, palette=palette, scale=scale, rotation_deg=rotation_deg)


def _rotate(points: np.ndarray, center: np.ndarray, deg: float) -> np.ndarray:
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return (points - center) @ rot.T + center


def _clip01(points: np.ndarray, margin: float = 0.08) -> np.ndarray:
    return np.clip(points, margin, 1.0 - margin)


def _grammar_arc_over_line(rng: np.random.Generator, palette: list[str], scale: float) -> list[Stroke]:
    """A rising diagonal closed by a small arc, plus two short accents."""
    c = np.array([0.5, 0.5])
    half = 0.32 * scale
    p0 = c + np.array([-half, -half * rng.uniform(0.8, 1.1)])
    p1 = c + np.array([half, half * rng.uniform(0.8, 1.1)])
    strokes = [
        Stroke("rising_diagonal", palette[0], _clip01(_line(p0, p1))),
        Stroke("closing_arc", palette[0], _clip01(_arc(p1 - np.array([0.0, 0.12 * scale]), 0.14 * scale, -40, 220))),
    ]
    accent_origin = c + np.array([rng.uniform(-0.3, -0.1), rng.uniform(-0.1, 0.1)]) * scale
    accent_end = accent_origin + np.array([rng.uniform(0.1, 0.22), rng.uniform(-0.05, 0.05)]) * scale
    strokes.append(Stroke("horizon_accent", palette[2], _clip01(_line(accent_origin, accent_end))))
    dash_origin = c + np.array([rng.uniform(0.05, 0.25), rng.uniform(-0.25, -0.1)]) * scale
    dash_end = dash_origin + np.array([rng.uniform(0.06, 0.12), rng.uniform(0.02, 0.06)]) * scale
    strokes.append(Stroke("accent_dash", palette[2], _clip01(_line(dash_origin, dash_end))))
    return strokes


def _grammar_nested_curves(rng: np.random.Generator, palette: list[str], scale: float) -> list[Stroke]:
    """Two or three nested S-curves of increasing size around a center."""
    c = np.array([0.5, 0.5])
    strokes = []
    n_curves = 3
    for i in range(n_curves):
        r = (0.14 + 0.09 * i) * scale
        p0 = c + np.array([-r, r * 0.3])
        p1 = c + np.array([0.0, -r * rng.uniform(0.5, 0.9)])
        p2 = c + np.array([r, r * 0.3])
        color = palette[i % 2]
        strokes.append(Stroke(f"nested_curve_{i}", color, _clip01(_quadratic_bezier(p0, p1, p2))))
    dash_origin = c + np.array([rng.uniform(-0.3, 0.3), rng.uniform(0.2, 0.3)]) * scale
    dash_end = dash_origin + np.array([rng.uniform(0.06, 0.14), rng.uniform(-0.03, 0.03)]) * scale
    strokes.append(Stroke("accent_dash", palette[2], _clip01(_line(dash_origin, dash_end))))
    return strokes


def _grammar_radiating_strokes(rng: np.random.Generator, palette: list[str], scale: float) -> list[Stroke]:
    """Short strokes radiating outward from a center point, plus one arc."""
    c = np.array([0.5, 0.5])
    n_rays = 4
    strokes = []
    base_angle = rng.uniform(0, 360)
    for i in range(n_rays):
        angle = np.radians(base_angle + i * (360.0 / n_rays) + rng.uniform(-10, 10))
        length = (0.18 + 0.05 * (i % 2)) * scale
        p0 = c + 0.05 * scale * np.array([np.cos(angle), np.sin(angle)])
        p1 = c + length * np.array([np.cos(angle), np.sin(angle)])
        color = palette[i % 2]
        strokes.append(Stroke(f"ray_{i}", color, _clip01(_line(p0, p1))))
    strokes.append(Stroke("center_arc", palette[2], _clip01(_arc(c, 0.07 * scale, 0, 300))))
    return strokes


_GRAMMAR_FNS = {
    "arc_over_line": _grammar_arc_over_line,
    "nested_curves": _grammar_nested_curves,
    "radiating_strokes": _grammar_radiating_strokes,
}


def compose(answers: PromptAnswers) -> tuple[list[Stroke], ArtisticBrief]:
    """Visitor answers -> (<=5-stroke artwork plan, the brief that produced it)."""
    brief = derive_brief(answers)
    rng = np.random.default_rng(brief.seed)
    strokes = _GRAMMAR_FNS[brief.grammar](rng, brief.palette, brief.scale)

    if abs(brief.rotation_deg) > 1e-6:
        center = np.array([0.5, 0.5])
        strokes = [
            Stroke(s.name, s.color, _clip01(_rotate(s.points, center, brief.rotation_deg))) for s in strokes
        ]

    assert len(strokes) <= 5, f"grammar {brief.grammar!r} produced {len(strokes)} strokes, budget is 5"
    return strokes, brief
