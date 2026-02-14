from __future__ import annotations

from collections import defaultdict

import numpy as np

from .types import CartesianPoint, Cluster


def cluster_points(
    points: list[CartesianPoint],
    eps_mm: float = 200.0,
    min_samples: int = 3,
    max_cluster_radius_mm: float = 500.0,
) -> list[Cluster]:
    """Group nearby cartesian points into clusters using DBSCAN.

    Points classified as noise are discarded. Clusters larger than
    max_cluster_radius_mm are also discarded (not person-sized).
    """
    if len(points) < min_samples:
        return []

    xy = np.array([[p.x, p.y] for p in points])
    labels = _grid_dbscan(xy, eps_mm, min_samples)

    clusters = []
    for label in range(labels.max() + 1):
        mask = labels == label
        cluster_xy = xy[mask]
        centroid_xy = cluster_xy.mean(axis=0)
        centroid = CartesianPoint(x=float(centroid_xy[0]), y=float(centroid_xy[1]))

        dists_from_centroid = np.sqrt(
            ((cluster_xy - centroid_xy) ** 2).sum(axis=1)
        )
        bounding_radius = float(dists_from_centroid.max())

        if bounding_radius > max_cluster_radius_mm:
            continue

        cluster_points_list = [
            CartesianPoint(x=float(row[0]), y=float(row[1])) for row in cluster_xy
        ]
        clusters.append(
            Cluster(
                centroid=centroid,
                points=cluster_points_list,
                bounding_radius_mm=bounding_radius,
            )
        )

    return clusters


def _grid_dbscan(
    xy: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """DBSCAN with a grid spatial index. Returns labels (-1 = noise)."""
    n = len(xy)
    cell_size = eps

    # Build grid index
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(n):
        cx = int(xy[i, 0] // cell_size)
        cy = int(xy[i, 1] // cell_size)
        cells[(cx, cy)].append(i)

    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    eps_sq = eps * eps

    for i in range(n):
        if labels[i] != -1:
            continue

        neighbors = _range_query(xy, i, cells, cell_size, eps_sq)
        if len(neighbors) < min_samples:
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            j += 1

            if labels[q] == -2:
                # Was explicitly marked noise — skip (shouldn't happen with -1 init)
                labels[q] = cluster_id
                continue

            if labels[q] != -1:
                continue

            labels[q] = cluster_id
            q_neighbors = _range_query(xy, q, cells, cell_size, eps_sq)
            if len(q_neighbors) >= min_samples:
                seed_set.extend(q_neighbors)

        cluster_id += 1

    return labels


def _range_query(
    xy: np.ndarray,
    idx: int,
    cells: dict[tuple[int, int], list[int]],
    cell_size: float,
    eps_sq: float,
) -> list[int]:
    """Find all points within eps of xy[idx] using the grid index."""
    px, py = xy[idx]
    cx = int(px // cell_size)
    cy = int(py // cell_size)

    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for j in cells.get((cx + dx, cy + dy), ()):
                dist_sq = (xy[j, 0] - px) ** 2 + (xy[j, 1] - py) ** 2
                if dist_sq <= eps_sq:
                    neighbors.append(j)
    return neighbors
