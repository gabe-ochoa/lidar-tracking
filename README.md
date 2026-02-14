# lidar-tracker

A standalone Python library that tracks moving objects (people) from 2D lidar scan data. It accepts polar point clouds, subtracts the static background, clusters foreground points into objects, assigns persistent IDs across frames, and records trajectories.

No hardware dependencies, no I/O — pure computation. The only runtime dependency is NumPy.

## Installation

```bash
pip install lidar-tracker
```

Or for development:

```bash
uv sync
```

## Quick start

```python
from lidar_tracker import TrackingEngine

engine = TrackingEngine()

# Feed scans as (angle_deg, distance_mm) tuples
frame = engine.process_scan([(angle, dist), ...])

for obj in frame.objects:
    print(f"Object #{obj.object_id} at ({obj.centroid.x:.0f}, {obj.centroid.y:.0f})mm")

# Get trajectory for a specific object
trail = engine.get_trajectory(obj.object_id)
```

The engine also accepts `list[PolarPoint]` instead of tuples. Points with `distance <= 0` are filtered automatically.

## How it works

```
Input polar points
  → BackgroundModel.update() + classify()    # learn static scene, extract foreground
  → polar_points_to_cartesian()              # convert to x,y mm
  → cluster_points()                         # DBSCAN → list[Cluster]
  → ObjectTracker.update()                   # match to existing tracks → list[TrackedObject]
  → TrajectoryStore.record()                 # append positions to history
  → TrackingFrame                            # output
```

1. **Background subtraction** — An exponential moving average per angular bin learns the static scene (walls, furniture). Points significantly closer than the learned background are classified as foreground.
2. **Clustering** — Foreground points are converted to Cartesian coordinates and grouped using grid-accelerated DBSCAN (no scipy/sklearn needed). Oversized clusters are rejected.
3. **Tracking** — Clusters are matched to existing tracks via greedy nearest-neighbor with velocity-based prediction. New tracks require a confirmation period before receiving a persistent ID. Lost tracks survive briefly to handle occlusion.
4. **Trajectory recording** — Each tracked object's position history is stored in a bounded deque for downstream analysis.

## Configuration

All parameters are set via the `TrackingEngine` constructor:

| Parameter | Default | Description |
|---|---|---|
| `background_learning_rate` | 0.02 | EMA alpha for background model |
| `foreground_threshold_mm` | 150.0 | Min distance closer than background to be foreground |
| `min_learning_frames` | 30 | Frames before tracking activates (~4s at 8 Hz) |
| `angle_bins` | 720 | Angular resolution of background model (0.5 deg) |
| `cluster_eps_mm` | 200.0 | DBSCAN neighborhood radius |
| `cluster_min_samples` | 3 | Min points to form a cluster |
| `max_cluster_radius_mm` | 500.0 | Reject clusters larger than this |
| `max_match_distance_mm` | 800.0 | Max distance to match a track to a cluster |
| `max_missing_frames` | 10 | Frames before a lost track is retired |
| `min_confirm_frames` | 2 | Frames before a new track gets a confirmed ID |
| `max_trajectory_length` | 0 | Max points per trajectory (0 = unlimited) |

## Conventions

- All distances are in **millimeters**
- All angles are in **degrees** (0-360)
- Cartesian coordinates: x = right, y = up, origin = sensor position
- Velocity is in **mm/frame** (multiply by scan rate to convert to mm/s)
- `TrackingEngine` is not thread-safe — call `process_scan` from a single thread

## Tests

```bash
uv run pytest
```

## License

MIT
