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

    # Group indices by (col, row)
    keys = np.stack([cols, rows], axis=1)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    tiles: dict[tuple[int, int], MMLPointCloud] = {}
    for i, (col, row) in enumerate(unique_keys):
        mask = inverse == i
        tiles[(int(col), int(row))] = MMLPointCloud(
            points=pc.points[mask],
            intensity=pc.intensity[mask],
            crs=pc.crs,
        )

    return tiles


def iter_tiles(
    pc: MMLPointCloud,
    tile_size: float = 50.0,
    origin_x: float | None = None,
    origin_y: float | None = None,
) -> Iterator[tuple[tuple[int, int], MMLPointCloud]]:
    """Yield (col, row), MMLPointCloud one tile at a time (memory-friendly)."""
    yield from tile(pc, tile_size=tile_size, origin_x=origin_x, origin_y=origin_y).items()
