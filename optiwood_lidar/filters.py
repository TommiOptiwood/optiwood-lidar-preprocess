"""
Filters for point cloud cleaning and clipping.

remove_noise — Statistical Outlier Removal (SOR) via scipy KDTree.
clip_bbox    — clip to a 2-D bounding box.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from .loaders import MMLPointCloud


def remove_noise(
    pc: MMLPointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> MMLPointCloud:
    """
    Remove statistical outliers (SOR algorithm).

    For each point, the mean distance to its *nb_neighbors* nearest neighbours
    is computed. Points whose mean distance exceeds
    ``global_mean + std_ratio * global_std`` are removed.

    This is the same algorithm used by Open3D and PDAL filters.outlier.

    Parameters
    ----------
    pc : MMLPointCloud
    nb_neighbors : int
        Number of neighbours to consider. Higher = stricter.
    std_ratio : float
        Multiplier for the standard deviation threshold. Lower = stricter.

    Returns
    -------
    MMLPointCloud with outliers removed.
    """
    if len(pc) <= nb_neighbors:
        return MMLPointCloud(
            points=pc.points.copy(),
            intensity=pc.intensity.copy(),
            crs=pc.crs,
        )

    tree = KDTree(pc.points)
    # Query k+1 because the point itself is always the closest match
    distances, _ = tree.query(pc.points, k=nb_neighbors + 1)
    mean_dists = distances[:, 1:].mean(axis=1)  # exclude self (distance=0)

    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    mask = mean_dists <= threshold

    return MMLPointCloud(
        points=pc.points[mask],
        intensity=pc.intensity[mask],
        crs=pc.crs,
    )


def clip_bbox(
    pc: MMLPointCloud,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> MMLPointCloud:
    """
    Clip to a 2-D bounding box (XY plane, Z is untouched).

    Parameters
    ----------
    pc : MMLPointCloud
    x_min, y_min, x_max, y_max : float
        Bounding box in the cloud's CRS.

    Returns
    -------
    MMLPointCloud containing only points inside the box (inclusive bounds).
    """
    x, y = pc.points[:, 0], pc.points[:, 1]
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    return MMLPointCloud(
        points=pc.points[mask],
        intensity=pc.intensity[mask],
        crs=pc.crs,
    )
