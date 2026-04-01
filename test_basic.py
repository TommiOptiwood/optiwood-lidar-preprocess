"""
Perustestit optiwood-lidar-utils kirjastolle.
Aja: pytest tests/
"""

import numpy as np
import pytest
import json
import tempfile
from pathlib import Path

from optiwood_lidar import MMLPointCloud, tile, iter_tiles, remove_noise, clip_bbox
from optiwood_lidar import to_numpy, to_geojson, to_ply, to_las


def make_test_cloud(n: int = 1000) -> MMLPointCloud:
    """Luo synteettinen pistepilvi testausta varten."""
    rng = np.random.default_rng(42)
    points = rng.uniform([0, 0, 0], [100, 100, 30], size=(n, 3))
    intensity = rng.uniform(0, 65535, size=n).astype(np.float32)
    return MMLPointCloud(points=points, intensity=intensity, crs="EPSG:3067")


class TestMMLPointCloud:

    def test_create(self):
        pc = make_test_cloud()
        assert len(pc) == 1000
        assert pc.crs == "EPSG:3067"

    def test_filter_height(self):
        pc = make_test_cloud(1000)
        filtered = pc.filter_height(min_height=5.0, max_height=25.0)
        assert len(filtered) < len(pc)
        assert filtered.points[:, 2].min() >= 5.0
        assert filtered.points[:, 2].max() <= 25.0

    def test_normalize_density(self):
        pc = make_test_cloud(10000)
        normalized = pc.normalize_density(target_density=1.0)
        assert len(normalized) < len(pc)

    def test_to_array_shape(self):
        pc = make_test_cloud(100)
        arr = pc.to_array()
        assert arr.shape == (100, 4)  # X, Y, Z, intensity

    def test_summary_returns_string(self):
        pc = make_test_cloud()
        summary = pc.summary()
        assert isinstance(summary, str)
        assert "MMLPointCloud" in summary

    def test_empty_cloud(self):
        pc = MMLPointCloud(
            points=np.zeros((0, 3)),
            intensity=np.zeros(0),
        )
        assert len(pc) == 0
        assert "empty" in pc.summary()

    def test_repr(self):
        pc = make_test_cloud(500)
        assert "500" in repr(pc)


class TestTiling:

    def test_tile_count(self):
        # 100x100m cloud, 50m tiles → 2x2 = 4 tiles
        pc = make_test_cloud(4000)
        tiles = tile(pc, tile_size=50.0)
        assert len(tiles) == 4

    def test_tile_keys(self):
        pc = make_test_cloud(4000)
        tiles = tile(pc, tile_size=50.0)
        assert set(tiles.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_all_points_preserved(self):
        pc = make_test_cloud(1000)
        tiles = tile(pc, tile_size=50.0)
        total = sum(len(t) for t in tiles.values())
        assert total == len(pc)

    def test_points_within_tile_bounds(self):
        pc = make_test_cloud(2000)
        tiles = tile(pc, tile_size=50.0)
        ox = pc.points[:, 0].min()
        oy = pc.points[:, 1].min()
        for (col, row), t in tiles.items():
            x_min = ox + col * 50.0
            x_max = ox + (col + 1) * 50.0
            y_min = oy + row * 50.0
            y_max = oy + (row + 1) * 50.0
            assert t.points[:, 0].min() >= x_min
            assert t.points[:, 0].max() < x_max
            assert t.points[:, 1].min() >= y_min
            assert t.points[:, 1].max() < y_max

    def test_empty_cloud_returns_empty_dict(self):
        pc = MMLPointCloud(points=np.zeros((0, 3)), intensity=np.zeros(0))
        assert tile(pc) == {}

    def test_crs_preserved(self):
        pc = make_test_cloud(100)
        tiles = tile(pc, tile_size=50.0)
        for t in tiles.values():
            assert t.crs == pc.crs

    def test_iter_tiles(self):
        pc = make_test_cloud(1000)
        result = list(iter_tiles(pc, tile_size=50.0))
        assert len(result) == 4
        assert all(isinstance(key, tuple) for key, _ in result)
        assert all(isinstance(t, MMLPointCloud) for _, t in result)


class TestFilters:

    def test_remove_noise_reduces_points(self):
        rng = np.random.default_rng(0)
        good = rng.uniform([0, 0, 0], [50, 50, 20], size=(500, 3))
        # Selkeät outlierit kaukana pilvestä
        outliers = rng.uniform([200, 200, 100], [300, 300, 200], size=(20, 3))
        points = np.vstack([good, outliers])
        intensity = np.ones(len(points), dtype=np.float32)
        pc = MMLPointCloud(points=points, intensity=intensity)
        cleaned = remove_noise(pc, nb_neighbors=10, std_ratio=1.0)
        assert len(cleaned) < len(pc)
        assert len(cleaned) >= 480  # hyvät pisteet säilyvät

    def test_remove_noise_preserves_crs(self):
        pc = make_test_cloud(200)
        cleaned = remove_noise(pc)
        assert cleaned.crs == pc.crs

    def test_remove_noise_small_cloud(self):
        # Vähemmän pisteitä kuin nb_neighbors → palautetaan sellaisenaan
        pc = make_test_cloud(5)
        cleaned = remove_noise(pc, nb_neighbors=20)
        assert len(cleaned) == len(pc)

    def test_clip_bbox_filters_correctly(self):
        pc = make_test_cloud(1000)  # X,Y uniform [0,100]
        clipped = clip_bbox(pc, x_min=25.0, y_min=25.0, x_max=75.0, y_max=75.0)
        assert len(clipped) < len(pc)
        assert clipped.points[:, 0].min() >= 25.0
        assert clipped.points[:, 0].max() <= 75.0
        assert clipped.points[:, 1].min() >= 25.0
        assert clipped.points[:, 1].max() <= 75.0

    def test_clip_bbox_preserves_crs(self):
        pc = make_test_cloud(100)
        clipped = clip_bbox(pc, 0, 0, 50, 50)
        assert clipped.crs == pc.crs

    def test_clip_bbox_empty_result(self):
        pc = make_test_cloud(100)  # X,Y in [0,100]
        clipped = clip_bbox(pc, x_min=200, y_min=200, x_max=300, y_max=300)
        assert len(clipped) == 0


class TestExporters:

    def test_to_numpy_roundtrip(self):
        pc = make_test_cloud(50)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        to_numpy(pc, path)
        data = np.load(path)
        np.testing.assert_array_equal(data["points"], pc.points)
        np.testing.assert_array_equal(data["intensity"], pc.intensity)

    def test_to_geojson_structure(self):
        pc = make_test_cloud(10)
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False, mode="w") as f:
            path = f.name
        to_geojson(pc, path)
        gj = json.loads(Path(path).read_text())
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 10
        feat = gj["features"][0]
        assert feat["geometry"]["type"] == "Point"
        assert len(feat["geometry"]["coordinates"]) == 3
        assert "intensity" in feat["properties"]

    def test_to_geojson_crs_preserved(self):
        pc = make_test_cloud(5)
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False, mode="w") as f:
            path = f.name
        to_geojson(pc, path)
        gj = json.loads(Path(path).read_text())
        assert gj["crs"]["properties"]["name"] == "EPSG:3067"

    def test_to_ply_header_and_count(self):
        pc = make_test_cloud(20)
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w") as f:
            path = f.name
        to_ply(pc, path)
        content = Path(path).read_text()
        assert "ply" in content
        assert "element vertex 20" in content
        assert "property float x" in content
        lines = [l for l in content.splitlines() if l and not l.startswith(("ply", "format", "element", "property", "end_header"))]
        assert len(lines) == 20

    def test_to_las_roundtrip(self):
        laspy = pytest.importorskip("laspy")
        pc = make_test_cloud(100)
        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            path = f.name
        to_las(pc, path)
        las = laspy.read(path)
        assert len(las.points) == 100
        np.testing.assert_allclose(las.x, pc.points[:, 0], atol=0.001)
        np.testing.assert_allclose(las.y, pc.points[:, 1], atol=0.001)
        np.testing.assert_allclose(las.z, pc.points[:, 2], atol=0.001)
