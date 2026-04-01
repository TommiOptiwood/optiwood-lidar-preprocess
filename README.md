# optiwood-lidar-utils

Open source preprocessing library for airborne LiDAR point clouds, with first-class support for data from the National Land Survey of Finland (MML/Maanmittauslaitos).

## Features

- **Load** LAS/LAZ files from MML or drone surveys
- **Tile** large point clouds into regular grids (default 50×50 m)
- **Filter** by height and remove statistical outliers (SOR)
- **Clip** to bounding boxes
- **Export** to NumPy (.npz), GeoJSON, PLY, and LAS

All steps are designed to run without PDAL — core dependencies are `laspy`, `numpy`, `scipy`, and `pyproj`, all pip-installable.

## Installation

```bash
pip install optiwood-lidar-utils
```

## Quick start

```python
import laspy
from optiwood_lidar import MMLPointCloud, tile, remove_noise, to_numpy

# Load
las = laspy.read("mml_tile.laz")
pc = MMLPointCloud(
    points=las.xyz,
    intensity=las.intensity.astype(float),
)

# MML data is pre-classified and pre-normalized — skip ground classification
print(pc.summary())
# MMLPointCloud: 142 037 points | CRS: EPSG:3067 | Z: [0.1, 28.4] m | density: 4.8 pts/m²

# Split into 50×50 m tiles
tiles = tile(pc, tile_size=50.0)

# Clean noise (optional for MML, useful for drone data)
tiles = {k: remove_noise(v) for k, v in tiles.items()}

# Export each tile for the ML pipeline
for (col, row), t in tiles.items():
    to_numpy(t, f"tile_{col}_{row}.npz")
```

## Processing modes

| Step | MML (avodata) | Drone |
|------|:---:|:---:|
| Read LAS/LAZ | ✅ | ✅ |
| Tiling | ✅ | ✅ |
| Noise removal | optional | ✅ |
| Ground classification | skip* | ✅ |
| Height normalisation | skip* | ✅ |
| CRS conversion | ✅ | ✅ |

\* MML data arrives pre-classified and pre-normalised.

## API reference

### `MMLPointCloud`

```python
MMLPointCloud(points, intensity, crs="EPSG:3067")
```

| Method | Description |
|--------|-------------|
| `filter_height(min, max)` | Keep points within Z range |
| `normalize_density(target)` | Subsample to target pts/m² |
| `to_array()` | Return (n, 4) NumPy array: X Y Z intensity |
| `summary()` | Human-readable statistics |

### Tiling

```python
tile(pc, tile_size=50.0, origin_x=None, origin_y=None)  # → dict[(col, row) → MMLPointCloud]
iter_tiles(pc, tile_size=50.0)                           # → generator
```

### Filters

```python
remove_noise(pc, nb_neighbors=20, std_ratio=2.0)         # Statistical Outlier Removal
clip_bbox(pc, x_min, y_min, x_max, y_max)
```

### Exporters

```python
to_numpy(pc, path)               # .npz
to_geojson(pc, path, to_wgs84=False)
to_ply(pc, path)                 # ASCII PLY
to_las(pc, path)                 # LAS 1.4
```

## License

MIT — free for commercial use.

---

Part of the [Optiwood](https://optiwood.io) tree-level LiDAR analysis pipeline.
