"""A placeholder 5-stroke minimalist artwork.

Stands in for the VLM-generated artistic brief described in the research
proposal (private/proposal_en.tex, Sec. "Semantic-to-Artistic Composition"):
a short personal prompt from a visitor would normally be composed into this
kind of stroke plan. This fixed plan exists so the simulation / robot
execution pipeline can be built and validated before the generative stage
is wired in -- swap `default_stroke_plan()` for a real VLM call later
without touching anything downstream of it.

Each Stroke is an ordered list of points in normalized canvas coordinates,
[0, 1]^2, with (0, 0) at one corner of the canvas. Points are sampled
densely enough along each curve that following them in order traces out a
smooth line/arc/bezier on the canvas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stroke:
    name: str
    color: str  # hex color, for the future paint/marker selection
    points: np.ndarray  # shape (N, 2), normalized canvas coords in [0, 1]


def _line(p0: np.ndarray, p1: np.ndarray, n: int = 10) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.outer(1 - t, p0) + np.outer(t, p1)


def _arc(center: np.ndarray, radius: float, start_deg: float, end_deg: float, n: int = 16) -> np.ndarray:
    t = np.radians(np.linspace(start_deg, end_deg, n))
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1)


def _quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 14) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def default_stroke_plan() -> list[Stroke]:
    """A fixed 5-stroke composition: a rising diagonal closed by a small
    arc, a short horizontal accent, a gentle S-curve, and a small accent
    dash -- matching the "rising diagonal with a small enclosed arc, warm
    palette, open negative space" example composition in the proposal."""
    return [
        Stroke("rising_diagonal", "#C1440E", _line(np.array([0.15, 0.18]), np.array([0.78, 0.74]))),
        Stroke("closing_arc", "#C1440E", _arc(np.array([0.78, 0.62]), 0.14, -40, 220)),
        Stroke("horizon_accent", "#1B1B1B", _line(np.array([0.18, 0.30]), np.array([0.42, 0.33]))),
        Stroke(
            "s_curve",
            "#2E6F72",
            _quadratic_bezier(np.array([0.20, 0.55]), np.array([0.42, 0.40]), np.array([0.60, 0.52])),
        ),
        Stroke("accent_dash", "#1B1B1B", _line(np.array([0.68, 0.22]), np.array([0.76, 0.25]))),
    ]
