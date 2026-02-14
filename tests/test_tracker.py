from lidar_tracker import CartesianPoint, Cluster, ObjectTracker


def _make_cluster(x: float, y: float, n_points: int = 5) -> Cluster:
    points = [CartesianPoint(x=x + i, y=y + i) for i in range(n_points)]
    return Cluster(centroid=CartesianPoint(x=x, y=y), points=points, bounding_radius_mm=100.0)


def test_new_track_not_confirmed_immediately():
    tracker = ObjectTracker(min_confirm_frames=2)
    clusters = [_make_cluster(1000.0, 1000.0)]
    objects = tracker.update(clusters)
    # First frame: not yet confirmed
    assert len(objects) == 0


def test_track_confirmed_after_min_frames():
    tracker = ObjectTracker(min_confirm_frames=2)
    clusters = [_make_cluster(1000.0, 1000.0)]
    tracker.update(clusters)
    objects = tracker.update(clusters)
    assert len(objects) == 1
    assert objects[0].object_id == 1


def test_persistent_id_across_frames():
    tracker = ObjectTracker(min_confirm_frames=1)
    c1 = [_make_cluster(1000.0, 1000.0)]
    objs1 = tracker.update(c1)
    assert len(objs1) == 1
    id1 = objs1[0].object_id

    # Move slightly
    c2 = [_make_cluster(1050.0, 1050.0)]
    objs2 = tracker.update(c2)
    assert len(objs2) == 1
    assert objs2[0].object_id == id1


def test_two_objects_get_different_ids():
    tracker = ObjectTracker(min_confirm_frames=1)
    clusters = [_make_cluster(1000.0, 1000.0), _make_cluster(3000.0, 3000.0)]
    objects = tracker.update(clusters)
    assert len(objects) == 2
    ids = {o.object_id for o in objects}
    assert len(ids) == 2


def test_lost_track_disappears_after_max_missing():
    tracker = ObjectTracker(min_confirm_frames=1, max_missing_frames=3)
    clusters = [_make_cluster(1000.0, 1000.0)]
    tracker.update(clusters)

    # Object disappears
    for _ in range(3):
        objects = tracker.update([])
    # Should still be tracked (missing_frames = 3, at the limit)
    # But not visible (missing_frames > 0)
    assert len(objects) == 0

    # One more frame — track gets retired
    tracker.update([])
    # New cluster at same position should get new ID
    objects = tracker.update(clusters)
    # Not confirmed yet with min_confirm_frames=1... update again
    objects = tracker.update(clusters)
    assert len(objects) == 1
    assert objects[0].object_id != 1  # new ID


def test_velocity_is_computed():
    tracker = ObjectTracker(min_confirm_frames=1)
    tracker.update([_make_cluster(1000.0, 1000.0)])
    objects = tracker.update([_make_cluster(1100.0, 1000.0)])
    assert len(objects) == 1
    assert abs(objects[0].velocity.x - 100.0) < 0.1
    assert abs(objects[0].velocity.y) < 0.1


def test_no_clusters_returns_empty():
    tracker = ObjectTracker()
    assert tracker.update([]) == []
