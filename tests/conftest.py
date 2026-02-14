"""Fixtures for generating synthetic lidar scans."""

from __future__ import annotations

import math

import pytest

from lidar_tracker import PolarPoint


def make_room_scan(
    wall_distance_mm: float = 5000.0,
    num_points: int = 720,
) -> list[PolarPoint]:
    """Generate a circular room scan (all walls at uniform distance).

    Default 720 points matches the 720 angular bins in BackgroundModel
    to ensure every bin has background data.
    """
    step = 360.0 / num_points
    return [
        PolarPoint(angle_deg=i * step, distance_mm=wall_distance_mm)
        for i in range(num_points)
    ]


def make_person_points(
    angle_center_deg: float,
    distance_mm: float,
    width_deg: float = 8.0,
    num_points: int = 10,
) -> list[PolarPoint]:
    """Generate points simulating a person at a given angle and distance."""
    start = angle_center_deg - width_deg / 2
    step = width_deg / (num_points - 1) if num_points > 1 else 0
    return [
        PolarPoint(
            angle_deg=(start + i * step) % 360,
            distance_mm=distance_mm + (i % 3) * 10,  # slight depth variation
        )
        for i in range(num_points)
    ]


def make_scan_with_people(
    wall_distance_mm: float = 5000.0,
    num_wall_points: int = 720,
    people: list[tuple[float, float]] | None = None,
) -> list[PolarPoint]:
    """Generate a room scan with optional people.

    Args:
        people: List of (angle_center_deg, distance_mm) for each person.
    """
    scan = make_room_scan(wall_distance_mm, num_wall_points)
    for angle, dist in (people or []):
        scan.extend(make_person_points(angle, dist))
    return scan


@pytest.fixture
def room_scan():
    """A simple circular room scan with no people."""
    return make_room_scan()


@pytest.fixture
def scan_with_one_person():
    """Room scan with one person at 90 degrees, 2000mm away."""
    return make_scan_with_people(people=[(90.0, 2000.0)])


@pytest.fixture
def scan_with_two_people():
    """Room scan with two people at different positions."""
    return make_scan_with_people(people=[(90.0, 2000.0), (270.0, 3000.0)])
