from lidar_tracker import BackgroundModel, PolarPoint
from tests.conftest import make_room_scan, make_scan_with_people


def test_not_ready_before_min_frames():
    model = BackgroundModel(min_learning_frames=10)
    scan = make_room_scan()
    for _ in range(9):
        model.update(scan)
    assert not model.is_ready()
    model.update(scan)
    assert model.is_ready()


def test_empty_foreground_for_static_room():
    model = BackgroundModel(min_learning_frames=5)
    scan = make_room_scan(wall_distance_mm=4000.0)
    for _ in range(10):
        model.update(scan)
    fg = model.classify(scan)
    assert len(fg) == 0


def test_detects_person_as_foreground():
    model = BackgroundModel(min_learning_frames=5, foreground_threshold_mm=150.0)
    room = make_room_scan(wall_distance_mm=5000.0)
    # Learn background
    for _ in range(10):
        model.update(room)

    # Now add a person at 2000mm (well within 5000mm wall)
    scan_with_person = make_scan_with_people(
        wall_distance_mm=5000.0, people=[(90.0, 2000.0)]
    )
    model.update(scan_with_person)
    fg = model.classify(scan_with_person)
    # The person points should be foreground
    assert len(fg) > 0
    for p in fg:
        assert p.distance_mm < 5000.0 - 150.0


def test_reset_clears_model():
    model = BackgroundModel(min_learning_frames=5)
    scan = make_room_scan()
    for _ in range(10):
        model.update(scan)
    assert model.is_ready()
    model.reset()
    assert not model.is_ready()


def test_returns_empty_when_not_ready():
    model = BackgroundModel(min_learning_frames=10)
    scan = make_scan_with_people(people=[(90.0, 2000.0)])
    for _ in range(5):
        model.update(scan)
    fg = model.classify(scan)
    assert fg == []
