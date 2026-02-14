from __future__ import annotations

from collections import deque

from .types import TrajectoryPoint


class TrajectoryStore:
    """Records position history for all tracked objects."""

    def __init__(self, max_trajectory_length: int = 0):
        """
        Args:
            max_trajectory_length: Max points per trajectory. 0 = unlimited.
        """
        self._max_length = max_trajectory_length
        self._trajectories: dict[int, deque[TrajectoryPoint]] = {}

    def record(
        self,
        object_id: int,
        x: float,
        y: float,
        frame_number: int,
        timestamp: float | None = None,
    ) -> None:
        """Append a position to this object's trajectory."""
        if object_id not in self._trajectories:
            self._trajectories[object_id] = deque(
                maxlen=self._max_length if self._max_length > 0 else None
            )
        self._trajectories[object_id].append(
            TrajectoryPoint(x=x, y=y, frame_number=frame_number, timestamp=timestamp)
        )

    def get(self, object_id: int) -> list[TrajectoryPoint]:
        """Get the full trajectory for an object. Returns [] if unknown."""
        if object_id not in self._trajectories:
            return []
        return list(self._trajectories[object_id])

    def get_all(self) -> dict[int, list[TrajectoryPoint]]:
        """Get all trajectories."""
        return {k: list(v) for k, v in self._trajectories.items()}

    def prune_inactive(
        self, active_ids: set[int]
    ) -> dict[int, list[TrajectoryPoint]]:
        """Remove and return trajectories for objects no longer tracked."""
        pruned = {}
        to_remove = [k for k in self._trajectories if k not in active_ids]
        for k in to_remove:
            pruned[k] = list(self._trajectories.pop(k))
        return pruned
