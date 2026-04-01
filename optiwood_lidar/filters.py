"""
Filters for point cloud cleaning and clipping.

remove_noise — Statistical Outlier Removal (SOR) via scipy KDTree.
clip_bbox    — clip to a 2-D bounding box.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt

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


def normalize_height(pc: MMLPointCloud, ground: MMLPointCloud) -> MMLPointCloud:
    """
    Normalize Z to height above ground.

    For each point, the ground elevation at (X, Y) is interpolated from
    the ground point cloud and subtracted from Z.

    Typical use with MML data (ground = classification 2):

        ground = MMLPointCloud(
            points=las.xyz[las.classification == 2],
            intensity=las.intensity[las.classification == 2].astype(float),
        )
        pc_norm = normalize_height(pc, ground)

    Parameters
    ----------
    pc : MMLPointCloud
        Full point cloud with absolute Z values.
    ground : MMLPointCloud
        Ground points only (classification 2 in MML data).

    Returns
    -------
    MMLPointCloud with Z replaced by height above ground (metres).
    Negative values are clipped to 0.
    """
    if len(ground) == 0:
        raise ValueError("ground point cloud is empty")

    # Rasterize ground points to a 1m grid, then look up each point — O(n)
    resolution = 1.0
    x_g, y_g, z_g = ground.points[:, 0], ground.points[:, 1], ground.points[:, 2]
    x_min, y_min = x_g.min(), y_g.min()

    xi = ((x_g - x_min) / resolution).astype(np.int64)
    yi = ((y_g - y_min) / resolution).astype(np.int64)
    nx, ny = int(xi.max()) + 1, int(yi.max()) + 1

    grid_sum = np.zeros((nx, ny), dtype=np.float64)
    grid_cnt = np.zeros((nx, ny), dtype=np.int32)
    np.add.at(grid_sum, (xi, yi), z_g)
    np.add.at(grid_cnt, (xi, yi), 1)

    filled = grid_cnt > 0
    grid_z = np.where(filled, grid_sum / np.maximum(grid_cnt, 1), 0.0)

    # Fill empty cells with nearest ground value
    if not filled.all():
        _, (ri, ci) = distance_transform_edt(~filled, return_indices=True)
        grid_z = grid_z[ri, ci]

    # Look up ground elevation for every point in pc
    xi_q = np.clip(((pc.points[:, 0] - x_min) / resolution).astype(np.int64), 0, nx - 1)
    yi_q = np.clip(((pc.points[:, 1] - y_min) / resolution).astype(np.int64), 0, ny - 1)
    ground_z_q = grid_z[xi_q, yi_q]

    new_points = pc.points.copy()
    new_points[:, 2] = np.maximum(pc.points[:, 2] - ground_z_q, 0.0)

    return MMLPointCloud(points=new_points, intensity=pc.intensity.copy(), crs=pc.crs)


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
