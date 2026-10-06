"""Stroke geometry primitives shared across the composition and
simulation pipeline.

`Stroke` and the curve-sampling helpers below are the building blocks
`composition.py` uses to generate artwork plans (see that module for the
actual visitor-answers -> stroke-plan logic). Each Stroke is an ordered
list of points in normalized canvas coordinates, [0, 1]^2, with (0, 0) at
one corner of the canvas. Points are sampled densely enough along each
curve that following them in order traces out a smooth line/arc/bezier
on the canvas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stroke:
    name: str
    color: str  # hex color, for the future paint/marker selection
    points: np.ndarray  # shape (N, 2), normalized canvas coords in [0, 1]
    width: float = 1.0  # relative brush width, 0-1, for rendering (not physical)


def _line(p0: np.ndarray, p1: np.ndarray, n: int = 10) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.outer(1 - t, p0) + np.outer(t, p1)


def _arc(center: np.ndarray, radius: float, start_deg: float, end_deg: float, n: int = 16) -> np.ndarray:
    t = np.radians(np.linspace(start_deg, end_deg, n))
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1)


def _quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 14) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def catmull_rom(points: np.ndarray, n_per_segment: int = 10) -> np.ndarray:
    """Smooth a handful of control points (e.g. ones an LLM picked
    directly) into a dense path, by fitting a Catmull-Rom spline through
    them -- unlike a quadratic bezier this passes exactly through every
    given point, which matters when the points themselves (not a
    separate control polygon) are the author's actual intent."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    if len(pts) == 2:
        return _line(pts[0], pts[1], n=n_per_segment)

    # Pad with reflected endpoints so the first/last real segment has a
    # sensible tangent.
    pad = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    n_segments = len(pad) - 3
    for i in range(n_segments):
        p0, p1, p2, p3 = pad[i], pad[i + 1], pad[i + 2], pad[i + 3]
        last = i == n_segments - 1
        t = np.linspace(0.0, 1.0, n_per_segment, endpoint=last)[:, None]
        t2, t3 = t * t, t * t * t
        seg = 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
        out.append(seg)
    return np.vstack(out)
