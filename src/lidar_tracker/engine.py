from __future__ import annotations

from .background import BackgroundModel
from .clustering import cluster_points
from ._math import polar_points_to_cartesian
from .tracker import ObjectTracker
from .trajectory import TrajectoryStore
from .types import PolarPoint, TrackingFrame, TrajectoryPoint


class TrackingEngine:
    """Main entry point: feed lidar scans, get tracked objects with trajectories.

    Usage::

        engine = TrackingEngine()
        for scan in lidar_scans:
            frame = engine.process_scan([(angle, dist), ...])
            for obj in frame.objects:
                print(f"Object {obj.object_id} at ({obj.centroid.x}, {obj.centroid.y})")
    """

    def __init__(
        self,
        # Background model
        background_learning_rate: float = 0.02,
        foreground_threshold_mm: float = 150.0,
        min_learning_frames: int = 30,
        angle_bins: int = 720,
        # Clustering
        cluster_eps_mm: float = 200.0,
        cluster_min_samples: int = 3,
        max_cluster_radius_mm: float = 500.0,
        # Tracking
        max_match_distance_mm: float = 800.0,
        max_missing_frames: int = 10,
        min_confirm_frames: int = 2,
        # Trajectories
        max_trajectory_length: int = 0,
    ):
        self._background = BackgroundModel(
            angle_bins=angle_bins,
            learning_rate=background_learning_rate,
            foreground_threshold_mm=foreground_threshold_mm,
            min_learning_frames=min_learning_frames,
        )
        self._cluster_eps = cluster_eps_mm
        self._cluster_min_samples = cluster_min_samples
        self._max_cluster_radius = max_cluster_radius_mm
        self._tracker = ObjectTracker(
            max_match_distance_mm=max_match_distance_mm,
            max_missing_frames=max_missing_frames,
            min_confirm_frames=min_confirm_frames,
        )
        self._trajectories = TrajectoryStore(
            max_trajectory_length=max_trajectory_length
        )
        self._frame_count = 0

    def process_scan(
        self,
        points: list[PolarPoint] | list[tuple[float, float]],
        timestamp: float | None = None,
    ) -> TrackingFrame:
        """Process a single lidar scan and return tracking results.

        Args:
            points: List of PolarPoint or (angle_deg, distance_mm) tuples.
                    Points with distance <= 0 are automatically filtered.
            timestamp: Optional timestamp for trajectory recording.

        Returns:
            TrackingFrame with all currently tracked objects.
        """
        # Normalize input
        polar = [
            p if isinstance(p, PolarPoint) else PolarPoint(angle_deg=p[0], distance_mm=p[1])
            for p in points
        ]
        polar = [p for p in polar if p.distance_mm > 0]

        # Update background model
        self._background.update(polar)

        # Classify foreground
        foreground = self._background.classify(polar)

        # Convert to cartesian and cluster
        cartesian = polar_points_to_cartesian(foreground)
        clusters = cluster_points(
            cartesian,
            eps_mm=self._cluster_eps,
            min_samples=self._cluster_min_samples,
            max_cluster_radius_mm=self._max_cluster_radius,
        )

        # Track objects
        tracked = self._tracker.update(clusters)

        # Record trajectories
        for obj in tracked:
            self._trajectories.record(
                object_id=obj.object_id,
                x=obj.centroid.x,
                y=obj.centroid.y,
                frame_number=self._frame_count,
                timestamp=timestamp,
            )

        frame = TrackingFrame(
            frame_number=self._frame_count,
            objects=tracked,
            timestamp=timestamp,
        )
        self._frame_count += 1
        return frame

    @property
    def background_ready(self) -> bool:
        """True once the background model has learned the static scene."""
        return self._background.is_ready()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def get_trajectory(self, object_id: int) -> list[TrajectoryPoint]:
        """Get the position history for a specific tracked object."""
        return self._trajectories.get(object_id)

    def get_all_trajectories(self) -> dict[int, list[TrajectoryPoint]]:
        """Get all trajectory histories."""
        return self._trajectories.get_all()

    def reset_background(self) -> None:
        """Clear the background model (call if room layout changes)."""
        self._background.reset()

    def reset(self) -> None:
        """Full reset: clear background, tracking state, and trajectories."""
        self._background.reset()
        self._tracker = ObjectTracker(
            max_match_distance_mm=self._tracker._max_match_dist,
            max_missing_frames=self._tracker._max_missing,
            min_confirm_frames=self._tracker._min_confirm,
        )
        self._trajectories = TrajectoryStore(
            max_trajectory_length=self._trajectories._max_length
        )
        self._frame_count = 0
