from lidar_tracker import CartesianPoint, cluster_points


def test_single_tight_cluster():
    points = [CartesianPoint(x=i * 10.0, y=0.0) for i in range(10)]
    clusters = cluster_points(points, eps_mm=50.0, min_samples=3)
    assert len(clusters) == 1
    assert len(clusters[0].points) == 10


def test_two_separated_clusters():
    group_a = [CartesianPoint(x=i * 10.0, y=0.0) for i in range(10)]
    group_b = [CartesianPoint(x=2000.0 + i * 10.0, y=0.0) for i in range(10)]
    clusters = cluster_points(group_a + group_b, eps_mm=50.0, min_samples=3)
    assert len(clusters) == 2


def test_noise_points_discarded():
    cluster_pts = [CartesianPoint(x=i * 10.0, y=0.0) for i in range(10)]
    noise = [CartesianPoint(x=5000.0, y=5000.0)]
    clusters = cluster_points(cluster_pts + noise, eps_mm=50.0, min_samples=3)
    assert len(clusters) == 1
    assert len(clusters[0].points) == 10


def test_too_few_points_returns_empty():
    points = [CartesianPoint(x=0.0, y=0.0), CartesianPoint(x=10.0, y=0.0)]
    clusters = cluster_points(points, eps_mm=50.0, min_samples=3)
    assert len(clusters) == 0


def test_oversized_cluster_rejected():
    # Spread points over 2000mm — way bigger than max_cluster_radius
    points = [CartesianPoint(x=i * 100.0, y=0.0) for i in range(30)]
    clusters = cluster_points(
        points, eps_mm=150.0, min_samples=3, max_cluster_radius_mm=500.0
    )
    assert len(clusters) == 0


def test_empty_input():
    assert cluster_points([], eps_mm=200.0, min_samples=3) == []


def test_centroid_is_mean_of_points():
    points = [
        CartesianPoint(x=0.0, y=0.0),
        CartesianPoint(x=100.0, y=0.0),
        CartesianPoint(x=50.0, y=50.0),
    ]
    clusters = cluster_points(points, eps_mm=200.0, min_samples=3)
    assert len(clusters) == 1
    c = clusters[0].centroid
    assert abs(c.x - 50.0) < 0.1
    assert abs(c.y - 50.0 / 3) < 0.1
