from __future__ import annotations

import numpy as np

from .types import PolarPoint


class BackgroundModel:
    """Learns the static scene and classifies points as foreground/background.

    Uses an exponential moving average (EMA) per angular bin to learn the
    typical distance at each angle. Points significantly closer than the
    learned background are classified as foreground (moving objects).
    """

    def __init__(
        self,
        angle_bins: int = 720,
        learning_rate: float = 0.02,
        foreground_threshold_mm: float = 150.0,
        min_learning_frames: int = 30,
    ):
        self._num_bins = angle_bins
        self._learning_rate = learning_rate
        self._threshold = foreground_threshold_mm
        self._min_frames = min_learning_frames

        self._bin_width = 360.0 / angle_bins
        self._background = np.full(angle_bins, np.inf)
        self._bin_counts = np.zeros(angle_bins, dtype=int)
        self._frame_count = 0

    def _angle_to_bin(self, angle_deg: float) -> int:
        return int(angle_deg / self._bin_width) % self._num_bins

    def update(self, points: list[PolarPoint]) -> None:
        """Feed a scan to update the background model."""
        for p in points:
            b = self._angle_to_bin(p.angle_deg)
            if self._bin_counts[b] == 0:
                self._background[b] = p.distance_mm
            else:
                # Only update background with points at or beyond current background.
                # Foreground objects (closer) should not pull the background inward.
                if p.distance_mm >= self._background[b] - self._threshold:
                    self._background[b] += self._learning_rate * (
                        p.distance_mm - self._background[b]
                    )
            self._bin_counts[b] += 1
        self._frame_count += 1

    def classify(self, points: list[PolarPoint]) -> list[PolarPoint]:
        """Return only foreground points (objects closer than background)."""
        if not self.is_ready():
            return []
        foreground = []
        for p in points:
            b = self._angle_to_bin(p.angle_deg)
            bg_dist = self._background[b]
            if np.isinf(bg_dist):
                continue
            if bg_dist - p.distance_mm > self._threshold:
                foreground.append(p)
        return foreground

    def is_ready(self) -> bool:
        """True once enough frames have been seen to trust the model."""
        return self._frame_count >= self._min_frames

    def reset(self) -> None:
        """Clear the background model."""
        self._background[:] = np.inf
        self._bin_counts[:] = 0
        self._frame_count = 0
