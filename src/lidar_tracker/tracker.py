from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import CartesianPoint, Cluster, TrackedObject


@dataclass
class _Track:
    """Internal track state."""

    track_id: int
    centroid: CartesianPoint
    velocity: CartesianPoint
    bounding_radius_mm: float
    points: list[CartesianPoint]
    age: int = 0
    missing_frames: int = 0
    confirmed: bool = False


class ObjectTracker:
    """Assigns persistent IDs to clusters across frames."""

    def __init__(
        self,
        max_match_distance_mm: float = 800.0,
        max_missing_frames: int = 10,
        min_confirm_frames: int = 2,
    ):
        self._max_match_dist = max_match_distance_mm
        self._max_missing = max_missing_frames
        self._min_confirm = min_confirm_frames
        self._next_id = 1
        self._tracks: list[_Track] = []

    def update(self, clusters: list[Cluster]) -> list[TrackedObject]:
        """Match clusters to tracks, manage lifecycle, return confirmed objects."""
        # Predict positions using velocity
        predicted = []
        for t in self._tracks:
            predicted.append(
                CartesianPoint(
                    x=t.centroid.x + t.velocity.x,
                    y=t.centroid.y + t.velocity.y,
                )
            )

        # Build cost matrix
        matches, unmatched_tracks, unmatched_clusters = self._assign(
            predicted, clusters
        )

        # Update matched tracks
        for track_idx, cluster_idx in matches:
            t = self._tracks[track_idx]
            c = clusters[cluster_idx]
            t.velocity = CartesianPoint(
                x=c.centroid.x - t.centroid.x,
                y=c.centroid.y - t.centroid.y,
            )
            t.centroid = c.centroid
            t.bounding_radius_mm = c.bounding_radius_mm
            t.points = c.points
            t.age += 1
            t.missing_frames = 0
            if t.age >= self._min_confirm:
                t.confirmed = True

        # Increment missing counter for unmatched tracks
        for track_idx in unmatched_tracks:
            self._tracks[track_idx].missing_frames += 1
            self._tracks[track_idx].age += 1

        # Create new tracks for unmatched clusters
        for cluster_idx in unmatched_clusters:
            c = clusters[cluster_idx]
            self._tracks.append(
                _Track(
                    track_id=self._next_id,
                    centroid=c.centroid,
                    velocity=CartesianPoint(x=0.0, y=0.0),
                    bounding_radius_mm=c.bounding_radius_mm,
                    points=c.points,
                    age=1,
                    confirmed=self._min_confirm <= 1,
                )
            )
            self._next_id += 1

        # Remove dead tracks
        self._tracks = [
            t for t in self._tracks if t.missing_frames <= self._max_missing
        ]

        # Return confirmed, visible tracks
        return [
            TrackedObject(
                object_id=t.track_id,
                centroid=t.centroid,
                velocity=t.velocity,
                bounding_radius_mm=t.bounding_radius_mm,
                age=t.age,
                points=t.points,
            )
            for t in self._tracks
            if t.confirmed and t.missing_frames == 0
        ]

    def _assign(
        self,
        predicted: list[CartesianPoint],
        clusters: list[Cluster],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Greedy nearest-neighbor assignment with gating."""
        num_tracks = len(predicted)
        num_clusters = len(clusters)

        if num_tracks == 0:
            return [], [], list(range(num_clusters))
        if num_clusters == 0:
            return [], list(range(num_tracks)), []

        # Cost matrix: distances between predicted track positions and cluster centroids
        cost = np.full((num_tracks, num_clusters), np.inf)
        for t in range(num_tracks):
            for c in range(num_clusters):
                dx = predicted[t].x - clusters[c].centroid.x
                dy = predicted[t].y - clusters[c].centroid.y
                d = np.sqrt(dx * dx + dy * dy)
                if d <= self._max_match_dist:
                    cost[t, c] = d

        # Greedy assignment: sort valid pairs by cost, assign greedily
        pairs = []
        for t in range(num_tracks):
            for c in range(num_clusters):
                if np.isfinite(cost[t, c]):
                    pairs.append((cost[t, c], t, c))
        pairs.sort()

        matches = []
        used_tracks: set[int] = set()
        used_clusters: set[int] = set()

        for _, t, c in pairs:
            if t not in used_tracks and c not in used_clusters:
                matches.append((t, c))
                used_tracks.add(t)
                used_clusters.add(c)

        unmatched_tracks = [t for t in range(num_tracks) if t not in used_tracks]
        unmatched_clusters = [c for c in range(num_clusters) if c not in used_clusters]
        return matches, unmatched_tracks, unmatched_clusters
