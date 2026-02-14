from lidar_tracker import TrackingEngine, PolarPoint
from tests.conftest import make_room_scan, make_scan_with_people


def test_no_objects_during_learning():
    engine = TrackingEngine(min_learning_frames=10, min_confirm_frames=1)
    scan = make_scan_with_people(people=[(90.0, 2000.0)])
    for _ in range(9):
        frame = engine.process_scan(scan)
        assert len(frame.objects) == 0
    assert not engine.background_ready


def test_detects_person_after_learning():
    engine = TrackingEngine(
        min_learning_frames=5,
        min_confirm_frames=1,
        foreground_threshold_mm=150.0,
    )
    room = make_room_scan(wall_distance_mm=5000.0)
    # Learn background
    for _ in range(10):
        engine.process_scan(room)

    assert engine.background_ready

    # Now add a person
    scan = make_scan_with_people(
        wall_distance_mm=5000.0, people=[(90.0, 2000.0)]
    )
    # Need a couple frames for track to confirm
    engine.process_scan(scan)
    frame = engine.process_scan(scan)
    assert len(frame.objects) >= 1


def test_tracks_two_people():
    engine = TrackingEngine(
        min_learning_frames=5,
        min_confirm_frames=1,
    )
    room = make_room_scan(wall_distance_mm=5000.0)
    for _ in range(10):
        engine.process_scan(room)

    scan = make_scan_with_people(
        wall_distance_mm=5000.0,
        people=[(90.0, 2000.0), (270.0, 3000.0)],
    )
    engine.process_scan(scan)
    frame = engine.process_scan(scan)
    assert len(frame.objects) == 2
    ids = {o.object_id for o in frame.objects}
    assert len(ids) == 2


def test_trajectory_recorded():
    engine = TrackingEngine(
        min_learning_frames=5,
        min_confirm_frames=1,
    )
    room = make_room_scan(wall_distance_mm=5000.0)
    for _ in range(10):
        engine.process_scan(room)

    scan = make_scan_with_people(
        wall_distance_mm=5000.0, people=[(90.0, 2000.0)]
    )
    engine.process_scan(scan)
    frame = engine.process_scan(scan)
    assert len(frame.objects) == 1

    obj_id = frame.objects[0].object_id
    traj = engine.get_trajectory(obj_id)
    assert len(traj) >= 1


def test_accepts_tuples():
    engine = TrackingEngine(min_learning_frames=2, min_confirm_frames=1)
    scan = [(i * 0.9, 5000.0) for i in range(400)]
    engine.process_scan(scan)
    frame = engine.process_scan(scan)
    assert frame.frame_number == 1


def test_filters_zero_distance():
    engine = TrackingEngine(min_learning_frames=2)
    scan = [PolarPoint(angle_deg=0.0, distance_mm=0.0)]
    frame = engine.process_scan(scan)
    assert len(frame.objects) == 0


def test_reset():
    engine = TrackingEngine(min_learning_frames=2)
    room = make_room_scan()
    for _ in range(5):
        engine.process_scan(room)
    assert engine.background_ready
    assert engine.frame_count == 5

    engine.reset()
    assert not engine.background_ready
    assert engine.frame_count == 0


def test_frame_numbers_increment():
    engine = TrackingEngine(min_learning_frames=1)
    room = make_room_scan()
    for i in range(5):
        frame = engine.process_scan(room)
        assert frame.frame_number == i
