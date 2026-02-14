# Design — lidar-tracker

## Goal

A reusable library for tracking multiple people walking through a space using a single 2D lidar mounted at the center of the room. Built as a standalone package so it can be used in any project — not coupled to specific hardware or visualization.

## Design decisions

### Standalone package, no hardware dependency

The tracker accepts generic polar points `(angle_deg, distance_mm)`. It does not know about RPLIDAR, serial ports, or any specific hardware. The conversion from hardware-specific types to polar tuples is the caller's responsibility (a one-liner). This means the tracker works with any 2D lidar.

### numpy is the only dependency

DBSCAN is implemented from scratch with a grid spatial index rather than pulling in scipy or scikit-learn. For the expected scale (20-100 foreground points per frame, 2-10 tracked objects), the hand-rolled implementation is fast enough and avoids a heavy transitive dependency tree.

### EMA background model (not median, not GMM)

The background model uses an exponential moving average per angular bin. Alternatives considered:

- **Running median**: Requires storing all historical samples per bin. EMA is O(1) memory per bin.
- **Gaussian Mixture Model**: Overkill for a static scene with one expected distance per angle. GMMs are useful when the background itself changes (e.g., waving trees) which doesn't apply indoors.

The EMA converges in ~30 frames (~4 seconds at 8Hz). The `learning_rate=0.02` parameter controls how quickly it adapts. Only points at or beyond the current background distance update the model — foreground objects (closer) don't corrupt it.

The `foreground_threshold_mm=150` means a point must be at least 150mm closer than the background to be classified as foreground. This catches human bodies (200-400mm depth in lidar cross-section) while ignoring measurement noise (typically <50mm jitter).

### Grid-accelerated DBSCAN (not k-means, not simple distance)

DBSCAN was chosen over alternatives because:

- It finds arbitrarily shaped clusters (a person is not a circle in lidar space)
- It handles noise naturally (stray points become noise, not their own cluster)
- It doesn't require knowing the number of people in advance
- The two parameters (`eps_mm`, `min_samples`) map directly to physical quantities

The grid spatial index hashes points into cells of size `eps_mm`, making neighbor queries O(1) per point. For 20-100 foreground points, DBSCAN is effectively O(n).

Parameters: `eps_mm=200` bridges gaps between lidar returns on a person's body. `min_samples=3` filters single-point noise while catching even small partial detections. `max_cluster_radius_mm=500` rejects anything too large to be a single person.

### Greedy assignment (not Hungarian)

Track-to-cluster matching uses greedy nearest-neighbor: sort all valid (track, cluster) pairs by distance, assign greedily. With 2-10 objects, this produces the same result as the Hungarian algorithm in virtually all cases. The implementation is trivial and has no dependencies.

The matcher uses velocity-based prediction: each track's position is extrapolated by its last velocity before computing distances. This helps when two people cross paths — their predicted positions diverge, preventing ID swaps.

### Track lifecycle

- **New tracks** start as tentative and are not reported until they've been seen for `min_confirm_frames` consecutive frames. This prevents single-frame noise from generating phantom IDs.
- **Lost tracks** (no matching cluster) survive for `max_missing_frames` (~1.25 seconds at 8Hz). During this time they're kept alive with their last velocity, so when a person reappears after brief occlusion, the predicted position is close to their actual position and the ID reconnects.
- **Retired tracks** are removed after exceeding `max_missing_frames`. Their trajectories remain in the `TrajectoryStore` until explicitly pruned.

### Known limitations

- **Two people merging**: When two people stand closer than `eps_mm` (200mm), DBSCAN merges them into one cluster. One track loses its match and goes into the "missing" state. When they separate, it reconnects. This is acceptable for the walking-in-a-room use case.
- **Far-range sparsity**: At >8 meters, a person may generate fewer than `min_samples` points per scan and not form a cluster. Adaptive `eps_mm` based on distance from sensor is a potential fix but not implemented.
- **Not thread-safe**: `TrackingEngine.process_scan()` must be called from a single thread. The caller is responsible for synchronization if needed.
- **Velocity jitter**: Raw centroid positions jitter frame-to-frame due to lidar noise. Velocity is computed as a simple delta without smoothing. A future improvement would add an EMA or Kalman filter on centroid positions.

## Intended use cases

1. **People tracking in rooms** — the primary use case. Single lidar at center, 2-10 people, room size up to ~12m diameter.
2. **Occupancy counting** — use track creation/retirement events to count entries and exits.
3. **Traffic analysis** — use trajectory histories to compute paths, dwell time, and flow patterns.
4. **Hardware evaluation** — feed different lidars through the same tracker to compare detection quality, scan rate effects, and tracking reliability.

## Future directions

- **Adaptive clustering** — scale `eps_mm` with distance from sensor to handle near and far targets equally
- **Centroid smoothing** — EMA or Kalman filter on positions before velocity computation
- **Zone events** — callback system for "object entered zone X", "object left zone Y"
- **Serialization** — save/load tracker state and trajectories for offline analysis
- **Multi-lidar fusion** — accept scans from multiple sensors and merge into a single tracking frame
