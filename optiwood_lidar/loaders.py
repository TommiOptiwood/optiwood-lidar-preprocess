"""
MMLPointCloud — point cloud from MML (National Land Survey of Finland) LiDAR data.

MML data arrives pre-classified and pre-normalized, so ground classification
and height normalization are not needed.
"""

from __future__ import annotations

import numpy as np


class MMLPointCloud:
    """
    Point cloud loaded from MML LiDAR data (LAS/LAZ).

    Parameters
    ----------
    points : array-like, shape (n, 3)
        XYZ coordinates in metres.
    intensity : array-like, shape (n,)
        Return intensity values.
    crs : str
        Coordinate reference system, default EPSG:3067 (Finnish national grid).
    """

    def __init__(
        self,
        points: np.ndarray,
        intensity: np.ndarray,
        crs: str = "EPSG:3067",
    ) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self.intensity = np.asarray(intensity, dtype=np.float32)
        self.crs = crs

        if self.points.ndim == 2 and self.points.shape[1] != 3:
            raise ValueError(f"points must have shape (n, 3), got {self.points.shape}")
        if len(self.points) != len(self.intensity):
            raise ValueError("points and intensity must have the same length")

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"MMLPointCloud(n={len(self)}, crs={self.crs!r})"

    def filter_height(self, min_height: float = -np.inf, max_height: float = np.inf) -> MMLPointCloud:
        """Return a new cloud with only points within [min_height, max_height] (Z axis)."""
        z = self.points[:, 2]
        mask = (z >= min_height) & (z <= max_height)
        return MMLPointCloud(
            points=self.points[mask],
            intensity=self.intensity[mask],
            crs=self.crs,
        )

    def normalize_density(self, target_density: float) -> MMLPointCloud:
        """
        Subsample to approximate target density (points/m²) using a regular grid.
        Keeps one point per grid cell — deterministic after shuffling.
        """
        if len(self) == 0:
            return MMLPointCloud(
                points=self.points.copy(),
                intensity=self.intensity.copy(),
                crs=self.crs,
            )

        cell_size = 1.0 / np.sqrt(target_density)
        x, y = self.points[:, 0], self.points[:, 1]
        ix = ((x - x.min()) / cell_size).astype(np.int64)
        iy = ((y - y.min()) / cell_size).astype(np.int64)

        # Shuffle so the kept point per cell is random, not always the first in file
        rng = np.random.default_rng()
        order = rng.permutation(len(self))

        n_cols = int(iy.max()) + 1
        cell_keys = ix[order] * n_cols + iy[order]
        _, first_in_cell = np.unique(cell_keys, return_index=True)
        keep = np.sort(order[first_in_cell])

        return MMLPointCloud(
            points=self.points[keep],
            intensity=self.intensity[keep],
            crs=self.crs,
        )

    def to_array(self) -> np.ndarray:
        """Return (n, 4) array: X, Y, Z, intensity."""
        return np.column_stack([self.points, self.intensity.reshape(-1, 1)])

    def summary(self) -> str:
        """Human-readable summary of the point cloud."""
        if len(self) == 0:
            return "MMLPointCloud (empty)"
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        xy_area = (x.max() - x.min()) * (y.max() - y.min())
        density = len(self) / xy_area if xy_area > 0 else float("nan")
        return (
            f"MMLPointCloud: {len(self):,} points | "
            f"CRS: {self.crs} | "
            f"Z: [{z.min():.1f}, {z.max():.1f}] m | "
            f"density: {density:.1f} pts/m²"
        )
