"""
Exporters — write MMLPointCloud to various formats.

to_numpy   — .npz (points + intensity, direct numpy/PyTorch input)
to_geojson — GeoJSON FeatureCollection (visualisation, GIS tools)
to_ply     — ASCII PLY (CloudCompare, MeshLab, Open3D)
to_las     — LAS 1.4 via laspy
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .loaders import MMLPointCloud


def to_numpy(pc: MMLPointCloud, path: str | Path) -> None:
    """
    Save as compressed .npz with arrays 'points' (n,3) and 'intensity' (n,).

    Load back with:
        data = np.load("file.npz")
        points, intensity = data["points"], data["intensity"]
    """
    np.savez_compressed(path, points=pc.points, intensity=pc.intensity)


def to_geojson(
    pc: MMLPointCloud,
    path: str | Path,
    to_wgs84: bool = False,
) -> None:
    """
    Save as GeoJSON FeatureCollection. Each point becomes a Feature with
    geometry type Point (X, Y, Z) and 'intensity' as a property.

    Parameters
    ----------
    to_wgs84 : bool
        Reproject coordinates to WGS84 (EPSG:4326) before writing.
        Requires pyproj (already a core dependency).
        Set True for compatibility with web maps (Leaflet, Mapbox, etc.).
    """
    x, y, z = pc.points[:, 0], pc.points[:, 1], pc.points[:, 2]

    if to_wgs84:
        from pyproj import Transformer
        transformer = Transformer.from_crs(pc.crs, "EPSG:4326", always_xy=True)
        x, y = transformer.transform(x, y)
        output_crs = "EPSG:4326"
    else:
        output_crs = pc.crs

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(x[i]), float(y[i]), float(z[i])],
            },
            "properties": {"intensity": float(pc.intensity[i])},
        }
        for i in range(len(pc))
    ]

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": output_crs}},
        "features": features,
    }

    Path(path).write_text(json.dumps(geojson), encoding="utf-8")


def to_ply(pc: MMLPointCloud, path: str | Path) -> None:
    """
    Save as ASCII PLY. Compatible with CloudCompare, MeshLab, and Open3D.
    """
    n = len(pc)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float intensity\n"
        "end_header\n"
    )

    rows = np.column_stack([pc.points, pc.intensity.reshape(-1, 1)])

    with open(path, "w", encoding="ascii") as f:
        f.write(header)
        np.savetxt(f, rows, fmt="%.6f", delimiter=" ")


def to_las(pc: MMLPointCloud, path: str | Path) -> None:
    """
    Save as LAS 1.4 (Point Format 0) using laspy.
    """
    import laspy

    header = laspy.LasHeader(point_format=0, version="1.4")
    header.offsets = pc.points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision

    las = laspy.LasData(header=header)
    las.x = pc.points[:, 0]
    las.y = pc.points[:, 1]
    las.z = pc.points[:, 2]
    las.intensity = np.clip(pc.intensity, 0, 65535).astype(np.uint16)

    las.write(str(path))
