# CLAUDE.md — lidar-tracker

## What is this?

A standalone Python library that tracks moving objects (people) from 2D lidar scan data. It accepts polar point clouds, subtracts the static background, clusters foreground points into objects, assigns persistent IDs across frames, and records trajectories. No hardware dependencies, no I/O — pure computation.

## Project layout

```
src/lidar_tracker/
├── __init__.py           # Public API re-exports
├── types.py              # PolarPoint, CartesianPoint, Cluster, TrackedObject, TrackingFrame, TrajectoryPoint
├── _math.py              # Polar/cartesian conversion (numpy vectorized)
├── background.py         # BackgroundModel — EMA per angular bin
├── clustering.py         # cluster_points() — grid-accelerated DBSCAN
├── tracker.py            # ObjectTracker — persistent ID assignment via greedy matching
├── trajectory.py         # TrajectoryStore — per-object position history
└── engine.py             # TrackingEngine — top-level orchestrator (main user API)

tests/
├── conftest.py           # Synthetic scan fixtures (room walls, simulated people)
├── test_background.py
├── test_clustering.py
├── test_tracker.py
└── test_engine.py
```

## Setup and tests

```bash
uv sync
uv run pytest        # 27 tests
uv run pytest -v     # verbose
```

## Dependencies

- **Runtime:** `numpy>=1.24` (only dependency)
- **Dev:** `pytest>=8.0`

## Usage

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

## Processing pipeline

```
Input polar points
  → BackgroundModel.update() + classify()    # learn static scene, extract foreground
  → polar_points_to_cartesian()              # convert to x,y mm
  → cluster_points()                         # DBSCAN → list[Cluster]
  → ObjectTracker.update()                   # match to existing tracks → list[TrackedObject]
  → TrajectoryStore.record()                 # append positions to history
  → TrackingFrame                            # output
```

## Key classes

### TrackingEngine (engine.py)
The main entry point. Orchestrates all subsystems. Constructor accepts all tunable parameters:

| Parameter | Default | What it controls |
|---|---|---|
| `background_learning_rate` | 0.02 | EMA alpha for background model |
| `foreground_threshold_mm` | 150.0 | Min distance closer than background to be foreground |
| `min_learning_frames` | 30 | Frames before tracking activates (~4s at 8Hz) |
| `angle_bins` | 720 | Angular resolution of background model (0.5°) |
| `cluster_eps_mm` | 200.0 | DBSCAN neighborhood radius |
| `cluster_min_samples` | 3 | Min points to form a cluster |
| `max_cluster_radius_mm` | 500.0 | Reject clusters larger than this |
| `max_match_distance_mm` | 800.0 | Max distance to match a track to a cluster |
| `max_missing_frames` | 10 | Frames before a lost track is retired |
| `min_confirm_frames` | 2 | Frames before a new track gets a confirmed ID |
| `max_trajectory_length` | 0 | Max points per trajectory (0 = unlimited) |

### BackgroundModel (background.py)
Learns the static scene (walls, furniture) using an EMA per angular bin. Classifies points as foreground if they are significantly closer than the learned background distance.

### ObjectTracker (tracker.py)
Assigns persistent IDs using greedy nearest-neighbor matching with velocity-based prediction. Handles: new objects entering, objects leaving, brief occlusion (tracks survive `max_missing_frames`), and requires `min_confirm_frames` before confirming a new track.

### cluster_points (clustering.py)
Grid-accelerated DBSCAN. No scipy/sklearn dependency. Rejects clusters exceeding `max_cluster_radius_mm`.

### TrajectoryStore (trajectory.py)
Records position history per object ID using bounded deques. `prune_inactive()` removes trajectories for retired tracks.

## Conventions

- All distances are in **millimeters**
- All angles are in **degrees** (0-360)
- Cartesian coordinates: x = right, y = up, origin = sensor position
- Velocity is in **mm/frame** (not mm/second — multiply by scan rate to get real units)
- `TrackingEngine` is not thread-safe — call `process_scan` from a single thread
- The library has no I/O — it does not read from serial ports, files, or network

## Writing tests

Test fixtures in `conftest.py` generate synthetic scans:
- `make_room_scan(wall_distance_mm, num_points)` — uniform circular room
- `make_person_points(angle, distance)` — cluster of points simulating a person
- `make_scan_with_people(wall_distance_mm, people=[(angle, dist), ...])` — combined

Use 720+ wall points in synthetic scans to match the default 720 angular bins, otherwise some bins will have no background data and foreground detection will be unreliable.
