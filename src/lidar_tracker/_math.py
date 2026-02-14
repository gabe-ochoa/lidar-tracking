from __future__ import annotations

import math

import numpy as np

from .types import CartesianPoint, PolarPoint


def polar_to_cartesian(angle_deg: float, distance_mm: float) -> tuple[float, float]:
    """Convert a single polar point to (x, y) in mm."""
    rad = math.radians(angle_deg)
    return distance_mm * math.cos(rad), distance_mm * math.sin(rad)


def polar_array_to_cartesian(
    angles_deg: np.ndarray, distances_mm: np.ndarray
) -> np.ndarray:
    """Vectorized polar to cartesian. Returns shape (N, 2) array of [x, y]."""
    rads = np.radians(angles_deg)
    return np.column_stack([distances_mm * np.cos(rads), distances_mm * np.sin(rads)])


def polar_points_to_cartesian(points: list[PolarPoint]) -> list[CartesianPoint]:
    """Convert a list of PolarPoints to CartesianPoints."""
    if not points:
        return []
    angles = np.array([p.angle_deg for p in points])
    dists = np.array([p.distance_mm for p in points])
    xy = polar_array_to_cartesian(angles, dists)
    return [CartesianPoint(x=float(row[0]), y=float(row[1])) for row in xy]


def distance_between(a: CartesianPoint, b: CartesianPoint) -> float:
    """Euclidean distance between two cartesian points."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
