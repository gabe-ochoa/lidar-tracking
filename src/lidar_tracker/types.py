from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PolarPoint:
    """A single lidar measurement in polar coordinates."""

    angle_deg: float  # 0-360
    distance_mm: float  # millimeters, 0 = invalid


@dataclass(slots=True)
class CartesianPoint:
    """A point in 2D cartesian space (millimeters from sensor origin)."""

    x: float
    y: float


@dataclass(slots=True)
class Cluster:
    """A group of related points detected as a single object."""

    centroid: CartesianPoint
    points: list[CartesianPoint]
    bounding_radius_mm: float


@dataclass(slots=True)
class TrackedObject:
    """A tracked object in a single frame."""

    object_id: int
    centroid: CartesianPoint
    velocity: CartesianPoint  # mm/frame (dx, dy since last frame)
    bounding_radius_mm: float
    age: int  # number of frames this ID has been tracked
    points: list[CartesianPoint]


@dataclass(slots=True)
class TrajectoryPoint:
    """A single point in an object's trajectory."""

    x: float
    y: float
    frame_number: int
    timestamp: float | None = None


@dataclass(slots=True)
class TrackingFrame:
    """Result of processing a single scan."""

    frame_number: int
    objects: list[TrackedObject]
    timestamp: float | None = None
