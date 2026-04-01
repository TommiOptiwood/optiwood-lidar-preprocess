"""
Tiling — splits a point cloud into regular grid tiles.

Typical use: 50x50m tiles before feeding into the ML pipeline.
"""

from __future__ import annotations

import numpy as np
from typing import Iterator

from .loaders import MMLPointCloud


def tile(
    pc: MMLPointCloud,
    tile_size: float = 50.0,
    origin_x: float | None = None,
    origin_y: float | None = None,
    min_points: int = 0,
) -> dict[tuple[int, int], MMLPointCloud]:
    """
    Split a point cloud into square tiles and return a dict keyed by (col, row).

    Empty tiles are omitted. Column/row indices start at (0, 0) at the origin.

    Parameters
    ----------
    pc : MMLPointCloud
    tile_size : float
        Side length of each tile in metres. Default 50.0.
    origin_x, origin_y : float, optional
        Grid origin. Defaults to the minimum X/Y of the cloud.
    min_points : int
        Discard tiles with fewer than this many points. Useful for dropping
        edge fragments at the boundary of a survey area. Default 0 (keep all).

    Returns
    -------
    dict[(col, row) -> MMLPointCloud]
    """
    if len(pc) == 0:
        return {}

    x, y = pc.points[:, 0], pc.points[:, 1]
    ox = x.min() if origin_x is None else origin_x
    oy = y.min() if origin_y is None else origin_y

    cols = np.floor((x - ox) / tile_size).astype(np.int64)
    rows = np.floor((y - oy) / tile_size).astype(np.int64)

    # Encode (col, row) as single int64 for fast sorting
    n_rows = int(rows.max()) + 1
    key = cols * n_rows + rows

    sort_idx = np.argsort(key, kind="stable")
    sorted_key = key[sort_idx]
    boundaries = np.flatnonzero(np.diff(sorted_key)) + 1
    groups = np.split(sort_idx, boundaries)
    unique_keys = sorted_key[np.concatenate([[0], boundaries])]

    tiles: dict[tuple[int, int], MMLPointCloud] = {}
    for k, idx in zip(unique_keys, groups):
        if len(idx) < min_points:
            continue
        col, row = int(k // n_rows), int(k % n_rows)
        tiles[(col, row)] = MMLPointCloud(
            points=pc.points[idx],
            intensity=pc.intensity[idx],
            crs=pc.crs,
        )

    return tiles


def iter_tiles(
    pc: MMLPointCloud,
    tile_size: float = 50.0,
    origin_x: float | None = None,
    origin_y: float | None = None,
    min_points: int = 0,
) -> Iterator[tuple[tuple[int, int], MMLPointCloud]]:
    """Yield (col, row), MMLPointCloud one tile at a time (memory-friendly)."""
    yield from tile(pc, tile_size=tile_size, origin_x=origin_x, origin_y=origin_y, min_points=min_points).items()
