"""Tests for roadflow.data using synthetic SpaceNet-like GeoTIFF fixtures.

The fixtures write real GeoTIFF tags, native GeoJSON centerlines, and small
tile grids.  No real SpaceNet data and no full training are needed, so the
tests are fast and hermetic.
"""
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from roadflow.data import (
    RoadDataset,
    crop_specs,
    discover_sources,
    load_centerlines,
    prepare,
    rasterize_crop,
    read_geotiff_info,
)
from roadflow.verifier import Verifier

TILE_SIZE = 64
PIXEL_DEG = 1e-5
ORIGIN_LON = -115.3000
ORIGIN_LAT = 36.1000
CROP = 16
BUFFER_M = 2.0
SPLITS = ("train", "dev", "holdout")
# Deliberately not monotonic with geographic position.
IMAGE_IDS = (
    (310, 101, 250),
    (430, 160, 205),
)
EMPTY_ID = 250


def _lonlat(lon0, lat0, col, row):
    return [lon0 + col * PIXEL_DEG, lat0 - row * PIXEL_DEG]


def _write_geotiff(path, lon0, lat0, tile_size=TILE_SIZE):
    values = np.empty((tile_size, tile_size, 3), dtype=np.uint16)
    values[:, :, 0] = np.arange(tile_size, dtype=np.uint16)[None, :]
    values[:, :, 1] = np.arange(tile_size, dtype=np.uint16)[:, None] * 2
    values[:, :, 2] = 77
    keys = (1, 1, 0, 3, 1024, 0, 1, 2, 2048, 0, 1, 4326, 2054, 0, 1, 9102)
    extratags = [
        (33550, "d", 3, (PIXEL_DEG, PIXEL_DEG, 0.0), True),
        (33922, "d", 6, (0.0, 0.0, 0.0, lon0, lat0, 0.0), True),
        (34735, "H", len(keys), keys, True),
    ]
    tifffile.imwrite(
        path, values, extratags=extratags, photometric="rgb", metadata=None
    )


def _write_geojson(path, lon0, lat0, tile_size=TILE_SIZE, dense=False):
    if dense:
        step = max(2, tile_size // 8)
        paths = [
            [_lonlat(lon0, lat0, 0, row), _lonlat(lon0, lat0, tile_size - 1, row)]
            for row in range(step, tile_size, step)
        ] + [
            [_lonlat(lon0, lat0, col, 0), _lonlat(lon0, lat0, col, tile_size - 1)]
            for col in range(step, tile_size, step)
        ]
    else:
        paths = [
            [_lonlat(lon0, lat0, col, 20) for col in (4, 24, 44, 60)],
            [_lonlat(lon0, lat0, 40, row) for row in (30, 45, 60)],
        ]
    features = [
        {
            "type": "Feature",
            "properties": {"road_id": index},
            "geometry": {"type": "LineString", "coordinates": coordinates},
        }
        for index, coordinates in enumerate(paths)
    ]
    payload = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_synthetic_dataset(root):
    aoi = root / "train" / "AOI_2_Vegas"
    image_dir = aoi / "PS-RGB"
    geojson_dir = aoi / "geojson_roads"
    image_dir.mkdir(parents=True)
    geojson_dir.mkdir(parents=True)
    for row, id_row in enumerate(IMAGE_IDS):
        for col, image_id in enumerate(id_row):
            lon0 = ORIGIN_LON + col * TILE_SIZE * PIXEL_DEG
            lat0 = ORIGIN_LAT - row * TILE_SIZE * PIXEL_DEG
            stem = f"SN3_roads_train_AOI_2_Vegas_PS-RGB_img{image_id}"
            _write_geotiff(image_dir / f"{stem}.tif", lon0, lat0)
            if image_id != EMPTY_ID:
                label_stem = (
                    f"SN3_roads_train_AOI_2_Vegas_geojson_roads_img{image_id}"
                )
                _write_geojson(geojson_dir / f"{label_stem}.geojson", lon0, lat0)
    return root


def build_grid_dataset(root, rows=5, cols=5, tile_size=TILE_SIZE, dense=True):
    """Square grid with unique image ids and (optionally) a dense road grid."""
    aoi = root / "train" / "AOI_2_Vegas"
    image_dir = aoi / "PS-RGB"
    geojson_dir = aoi / "geojson_roads"
    image_dir.mkdir(parents=True)
    geojson_dir.mkdir(parents=True)
    ids = []
    for row in range(rows):
        id_row = []
        for col in range(cols):
            # gcd(37, 101) == 1 keeps ids unique for these grid sizes.
            image_id = 1000 + 37 * row + 101 * col
            id_row.append(image_id)
            lon0 = ORIGIN_LON + col * tile_size * PIXEL_DEG
            lat0 = ORIGIN_LAT - row * tile_size * PIXEL_DEG
            stem = f"SN3_roads_train_AOI_2_Vegas_PS-RGB_img{image_id}"
            _write_geotiff(image_dir / f"{stem}.tif", lon0, lat0, tile_size)
            label_stem = (
                f"SN3_roads_train_AOI_2_Vegas_geojson_roads_img{image_id}"
            )
            _write_geojson(
                geojson_dir / f"{label_stem}.geojson",
                lon0,
                lat0,
                tile_size,
                dense=dense,
            )
        ids.append(id_row)
    return root, ids


def _source_id(image_id):
    return f"AOI_2_Vegas_img{image_id}"


def _chebyshev(a, b):
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _connected(cells):
    cells = {tuple(cell) for cell in cells}
    visited = {next(iter(cells))}
    frontier = list(visited)
    while frontier:
        row, col = frontier.pop()
        for neighbour in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if neighbour in cells and neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return visited == cells


class GeoRasterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = build_synthetic_dataset(Path(self.temp.name) / "dataset")
        self.image_path = (
            self.root
            / "train"
            / "AOI_2_Vegas"
            / "PS-RGB"
            / "SN3_roads_train_AOI_2_Vegas_PS-RGB_img101.tif"
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_geotiff_tags_bounds_and_metric_pixel_size(self):
        info = read_geotiff_info(self.image_path)
        self.assertEqual((info.width, info.height), (TILE_SIZE, TILE_SIZE))
        self.assertEqual(info.crs_epsg, 4326)
        self.assertTrue(info.is_geographic)
        self.assertAlmostEqual(info.transform[0], PIXEL_DEG, places=15)
        self.assertAlmostEqual(info.transform[4], -PIXEL_DEG, places=15)
        # img101 is row 0, column 1 of the synthetic grid.
        lon0 = ORIGIN_LON + TILE_SIZE * PIXEL_DEG
        lat0 = ORIGIN_LAT
        xmin, ymin, xmax, ymax = info.geo_bounds()
        self.assertAlmostEqual(xmin, lon0, places=12)
        self.assertAlmostEqual(xmax, lon0 + TILE_SIZE * PIXEL_DEG, places=12)
        self.assertAlmostEqual(ymax, lat0, places=12)
        self.assertAlmostEqual(ymin, lat0 - TILE_SIZE * PIXEL_DEG, places=12)
        # EPSG:4326 pixels are anisotropic in metres at this latitude.
        row_m, col_m = info.pixel_size_m(lat0 - TILE_SIZE * PIXEL_DEG / 2)
        self.assertTrue(1.05 < row_m < 1.15, row_m)
        self.assertTrue(0.85 < col_m < 0.95, col_m)

    def test_geojson_parsing_and_missing_annotation(self):
        label_path = (
            self.root
            / "train"
            / "AOI_2_Vegas"
            / "geojson_roads"
            / "SN3_roads_train_AOI_2_Vegas_geojson_roads_img101.geojson"
        )
        lines = load_centerlines(label_path)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.shape[1] == 2 for line in lines))
        self.assertEqual(load_centerlines(None), [])

        multi_path = Path(self.temp.name) / "multi.geojson"
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": [
                            [[-115.3, 36.1], [-115.299, 36.1]],
                            [[-115.3, 36.1], [-115.3, 36.099]],
                        ],
                    },
                }
            ],
        }
        multi_path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(len(load_centerlines(multi_path)), 2)

    def test_empty_source_discovered_as_empty_label(self):
        sources = {source.source_id: source for source in discover_sources(self.root)}
        self.assertEqual(len(sources), 6)
        empty = sources[_source_id(EMPTY_ID)]
        self.assertIsNone(empty.geojson_path)
        self.assertEqual(load_centerlines(empty.geojson_path), [])

    def test_rasterize_buffered_centerline_and_empty_crop(self):
        info = read_geotiff_info(self.image_path)
        lat0 = ORIGIN_LAT
        lon0 = ORIGIN_LON + TILE_SIZE * PIXEL_DEG
        road = np.asarray(
            [
                _lonlat(lon0, lat0, 0, 32),
                _lonlat(lon0, lat0, TILE_SIZE - 1, 32),
            ]
        )
        labels = rasterize_crop(info, [road], x0=16, y0=16, size=32, buffer_m=2.5)
        self.assertEqual(labels.mask.shape, (32, 32))
        self.assertEqual(labels.mask.dtype, np.float32)
        self.assertEqual(labels.centerline.dtype, np.uint8)
        centerline = labels.centerline.astype(bool)
        self.assertGreater(centerline.sum(), 0)
        # The buffered mask must always contain the native centerline.
        self.assertTrue(np.all(labels.mask[centerline] == 1.0))
        # One-pixel horizontal line at crop-local row 16; row_m ~ 1.11 m, so a
        # 2.5 m buffer spans about five rows, not the whole crop.
        column_profile = labels.mask[:, 16]
        self.assertGreaterEqual(column_profile.sum(), 3)
        self.assertLessEqual(column_profile.sum(), 7)
        self.assertTrue(labels.mask[16, :].all())
        self.assertEqual(labels.polyline_offsets.tolist(), [0, 2])
        self.assertEqual(labels.polyline_points.shape, (2, 2))
        np.testing.assert_allclose(labels.polyline_points[:, 1], [16.0, 16.0])

        empty = rasterize_crop(info, [], x0=16, y0=16, size=32, buffer_m=2.5)
        self.assertEqual(empty.mask.sum(), 0)
        self.assertEqual(empty.centerline.sum(), 0)
        self.assertEqual(empty.polyline_points.shape, (0, 2))
        self.assertEqual(empty.polyline_offsets.tolist(), [0])


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_splits_manifest_npz_dataset_and_buffer(self):
        root, _ids = build_grid_dataset(
            self.base / "grid", rows=5, cols=5, dense=True
        )
        output = self.base / "prepared"
        manifest = prepare(
            root,
            output,
            crop_size=CROP,
            train_limit=48,
            dev_limit=32,
            holdout_limit=32,
            seed=5,
            buffer_m=BUFFER_M,
            spatial_buffer_tiles=1,
            query_pool_size=32,
        )
        self.assertEqual(manifest["format_version"], 2)
        self.assertEqual(manifest["splits"], list(SPLITS))
        self.assertNotIn("val", manifest)
        self.assertEqual(manifest["spatial_buffer_tiles"], 1)

        expected = {"train": 48, "dev": 32, "holdout": 32}
        for split, count in expected.items():
            self.assertEqual(len(manifest[split]), count)
            self.assertEqual(manifest["counts"][split]["crops"], count)
            self.assertEqual(manifest["shortfall"][split]["requested"], count)
            self.assertEqual(manifest["shortfall"][split]["shortfall"], 0)

        source_sets = {
            split: {record["source_id"] for record in manifest[split]}
            for split in SPLITS
        }
        self.assertFalse(source_sets["train"] & source_sets["dev"])
        self.assertFalse(source_sets["train"] & source_sets["holdout"])
        self.assertFalse(source_sets["dev"] & source_sets["holdout"])

        # Dev and holdout are whole-tile, 4-connected geographic regions.
        strategy = manifest["split_strategy"]
        dev_cells = {tuple(cell) for cell in strategy["dev_grid_cells"]}
        holdout_cells = {tuple(cell) for cell in strategy["holdout_grid_cells"]}
        self.assertGreaterEqual(len(dev_cells), 2)
        self.assertGreaterEqual(len(holdout_cells), 2)
        self.assertTrue(_connected(dev_cells))
        self.assertTrue(_connected(holdout_cells))
        self.assertFalse(dev_cells & holdout_cells)

        # The spatial buffer is explicit and keeps train away from both
        # evaluation regions.
        self.assertEqual(
            manifest["counts"]["excluded_buffer_sources"],
            len(manifest["excluded_sources"]),
        )
        self.assertGreater(len(manifest["excluded_sources"]), 0)
        self.assertEqual(
            sorted(row["source_id"] for row in manifest["excluded_sources"]),
            sorted(strategy["buffer_source_ids"]),
        )
        for row in manifest["excluded_sources"]:
            self.assertEqual(row["reason"], "spatial_buffer")
            self.assertIn(row["nearest_eval_region"], ("dev", "holdout"))
            self.assertGreaterEqual(row["grid_distance"], 1)
        eval_cells = dev_cells | holdout_cells
        for source_id, meta in manifest["sources"].items():
            if meta["split"] != "train":
                continue
            cell = tuple(meta["grid_cell"])
            self.assertGreater(
                min(_chebyshev(cell, other) for other in eval_cells), 1
            )

        # Records, npz schema, query caches and applicability fields.
        for record in manifest["train"] + manifest["dev"] + manifest["holdout"]:
            path = output / record["npz"]
            self.assertTrue(path.is_file(), path)
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(
                    set(payload.files),
                    {
                        "image",
                        "mask",
                        "centerline",
                        "polyline_points",
                        "polyline_offsets",
                        "crop_origin",
                        "pixel_size_m",
                    },
                )
                self.assertEqual(payload["image"].shape, (3, CROP, CROP))
                self.assertEqual(payload["centerline"].shape, (CROP, CROP))
            self.assertEqual(record["split"], record["npz"].split("/")[0])
            self.assertIn("applicability", record)
            self.assertEqual(
                record["query_pool_size"], record["applicability"]["n_queries"]
            )
            if record["query_pool_requested"] > 0:
                self.assertIsNotNone(record["query_cache"])
                self.assertTrue((output / record["query_cache"]).is_file())
            else:
                self.assertIsNone(record["query_cache"])

        # Dataset item contract used by the CLI.
        dataset = RoadDataset(output / "manifest.json", "train")
        self.assertEqual(len(dataset), 48)
        item = dataset[0]
        self.assertEqual(
            set(item),
            {
                "image",
                "mask",
                "centerline",
                "id",
                "source_id",
                "query_pool",
                "query_cache",
                "npz",
                "stratum",
            },
        )
        import torch

        self.assertIsInstance(item["image"], torch.Tensor)
        self.assertEqual(tuple(item["image"].shape), (3, CROP, CROP))
        self.assertEqual(tuple(item["mask"].shape), (1, CROP, CROP))
        self.assertEqual(tuple(item["centerline"].shape), (1, CROP, CROP))
        self.assertEqual(item["centerline"].dtype, torch.uint8)
        self.assertEqual(item["source_id"], dataset.records[0]["source_id"])
        self.assertEqual(
            item["stratum"],
            dataset.records[0]["applicability"]["category"],
        )
        self.assertIsInstance(item["query_pool"], list)
        for query in item["query_pool"]:
            self.assertEqual(len(query), 3)
            self.assertEqual(len(query[0]), 2)
            self.assertEqual(len(query[1]), 2)
            self.assertGreater(query[2], 0)
        if item["query_pool"]:
            mask = item["mask"].squeeze().numpy()
            verifier = Verifier(mask, queries=item["query_pool"])
            self.assertAlmostEqual(verifier.score(mask)["path"], 1.0)

        # New manifests expose dev/holdout; legacy "val" resolves to dev and is
        # never silently replaced by holdout.
        val_dataset = RoadDataset(output / "manifest.json", "val")
        self.assertEqual(val_dataset.split_key, "dev")
        self.assertEqual(val_dataset.ids, dataset_ids(manifest["dev"]))
        self.assertEqual(
            len(RoadDataset(output / "manifest.json", "holdout")), 32
        )
        from_dict = RoadDataset(manifest, "holdout", root=output)
        self.assertEqual(len(from_dict), 32)
        with self.assertRaises(ValueError):
            RoadDataset(manifest, "test")
        missing_root = RoadDataset(manifest, "train", root=self.base / "missing")
        with self.assertRaises(FileNotFoundError):
            _ = missing_root[0]

        # Applicability report exists, is explicit about nulls, and sums to the
        # split sizes.
        report = json.loads((output / "applicability.json").read_text())
        self.assertEqual(report["query_pool_requested"], 32)
        self.assertEqual(len(report["crops"]), 48 + 32 + 32)
        for split in SPLITS:
            summary = report["splits"][split]
            self.assertEqual(summary["crops"], expected[split])
            self.assertEqual(sum(summary["categories"].values()), expected[split])
            self.assertAlmostEqual(
                sum(summary["category_fraction"].values()), 1.0, places=12
            )
            if summary["query_length_px"] is None:
                self.assertIsNotNone(summary["query_length_px_reason"])
            if summary["complete_visible_path_ratio"] is None:
                self.assertIsNotNone(
                    summary["complete_visible_path_ratio_reason"]
                )
            else:
                self.assertGreaterEqual(
                    summary["complete_visible_path_ratio"], 0.0
                )
                self.assertLessEqual(summary["complete_visible_path_ratio"], 1.0)

    def test_prepare_is_seeded_and_reproducible(self):
        root, _ids = build_grid_dataset(
            self.base / "grid", rows=5, cols=5, dense=True
        )
        params = dict(
            crop_size=CROP,
            train_limit=16,
            dev_limit=16,
            holdout_limit=16,
            seed=11,
            buffer_m=BUFFER_M,
            spatial_buffer_tiles=1,
            query_pool_size=8,
        )
        first = prepare(root, self.base / "one", **params)
        second = prepare(root, self.base / "two", **params)
        for split in SPLITS:
            self.assertEqual(
                [record["id"] for record in first[split]],
                [record["id"] for record in second[split]],
            )
            self.assertEqual(
                [record["query_pool_size"] for record in first[split]],
                [record["query_pool_size"] for record in second[split]],
            )
        for key in (
            "dev_source_ids",
            "holdout_source_ids",
            "buffer_source_ids",
        ):
            self.assertEqual(
                first["split_strategy"][key], second["split_strategy"][key]
            )
        self.assertEqual(first["shortfall"], second["shortfall"])

    def test_all_crops_path_and_missing_geojson_is_zero_mask(self):
        root = build_synthetic_dataset(self.base / "dataset")
        output = self.base / "all"
        manifest = prepare(
            root,
            output,
            crop_size=CROP,
            train_limit=10**6,
            dev_limit=0,
            holdout_limit=0,
            seed=1,
            buffer_m=BUFFER_M,
            spatial_buffer_tiles=0,
            query_pool_size=8,
        )
        total = 6 * (TILE_SIZE // CROP) ** 2
        self.assertEqual(len(manifest["train"]), total)
        self.assertEqual(manifest["dev"], [])
        self.assertEqual(manifest["holdout"], [])
        self.assertNotIn("val", manifest)
        self.assertEqual(len(RoadDataset(manifest, "dev", root=output)), 0)
        self.assertEqual(len(RoadDataset(manifest, "val", root=output)), 0)

        empty_source = _source_id(EMPTY_ID)
        empty_records = [
            record
            for record in manifest["train"]
            if record["source_id"] == empty_source
        ]
        self.assertEqual(len(empty_records), (TILE_SIZE // CROP) ** 2)
        self.assertIsNone(empty_records[0]["source_geojson"])
        self.assertEqual(empty_records[0]["label_source"], "missing_geojson_empty")
        self.assertTrue(all(record["empty"] for record in empty_records))
        for record in empty_records:
            with np.load(output / record["npz"], allow_pickle=False) as payload:
                self.assertEqual(payload["mask"].sum(), 0)
                self.assertEqual(payload["centerline"].sum(), 0)
                self.assertEqual(payload["polyline_points"].shape, (0, 2))
                self.assertEqual(payload["polyline_offsets"].tolist(), [0])
            self.assertEqual(record["query_pool_size"], 0)
            self.assertEqual(
                record["query_pool_shortfall_reason"], "empty_mask"
            )
            applicability = record["applicability"]
            self.assertEqual(applicability["category"], "empty")
            self.assertIsNone(applicability["query_length_px"])
            self.assertEqual(
                applicability["query_length_px_reason"], "empty_mask"
            )
            self.assertIsNone(applicability["complete_visible_path_ratio"])
            self.assertEqual(
                applicability["complete_visible_path_ratio_reason"],
                "empty_mask",
            )
            item = RoadDataset(output / "manifest.json", "train").__getitem__(
                next(
                    index
                    for index, candidate in enumerate(manifest["train"])
                    if candidate["id"] == record["id"]
                )
            )
            self.assertEqual(item["query_pool"], [])
            self.assertIsNotNone(item["query_cache"])
        self.assertEqual(
            manifest["counts"]["train"]["empty_crops"],
            sum(bool(record["empty"]) for record in manifest["train"]),
        )

    def test_spatial_buffer_shortfall_is_explicit(self):
        root, _ids = build_grid_dataset(
            self.base / "grid", rows=4, cols=4, dense=True
        )
        output = self.base / "shortfall"
        manifest = prepare(
            root,
            output,
            crop_size=CROP,
            train_limit=1000,
            dev_limit=32,
            holdout_limit=32,
            seed=7,
            buffer_m=BUFFER_M,
            spatial_buffer_tiles=1,
            query_pool_size=8,
        )
        train = manifest["shortfall"]["train"]
        self.assertEqual(train["requested"], 1000)
        self.assertLess(train["actual"], 1000)
        self.assertEqual(train["shortfall"], 1000 - train["actual"])
        self.assertEqual(
            train["reason"],
            "insufficient_train_crop_capacity_after_spatial_buffer",
        )
        self.assertEqual(train["capacity_crops"], len(manifest["train"]))
        self.assertEqual(manifest["capacity"]["train"], len(manifest["train"]))
        self.assertGreater(len(manifest["excluded_sources"]), 0)
        self.assertEqual(
            manifest["counts"]["excluded_buffer_sources"],
            len(manifest["excluded_sources"]),
        )
        for split in ("dev", "holdout"):
            self.assertEqual(manifest["shortfall"][split]["shortfall"], 0)

    def test_query_pool_288_verifier_self_path_and_native_refs(self):
        root, _ids = build_grid_dataset(
            self.base / "grid", rows=5, cols=5, tile_size=TILE_SIZE, dense=True
        )
        output = self.base / "queries"
        manifest = prepare(
            root,
            output,
            crop_size=TILE_SIZE,
            train_limit=1,
            dev_limit=1,
            holdout_limit=1,
            seed=3,
            buffer_m=BUFFER_M,
            spatial_buffer_tiles=1,
            query_pool_size=288,
        )
        record = manifest["train"][0]
        self.assertEqual(record["query_pool_requested"], 288)
        self.assertEqual(record["query_pool_shortfall"], 0)
        self.assertIsNone(record["query_pool_shortfall_reason"])
        self.assertEqual(record["query_pool_size"], 288)
        self.assertEqual(record["applicability"]["n_queries"], 288)
        self.assertEqual(record["applicability"]["category"], "complex")
        self.assertGreater(record["applicability"]["junction_pixels"], 0)
        self.assertGreater(record["applicability"]["cycles"], 0)

        dataset = RoadDataset(output / "manifest.json", "train")
        item = dataset[0]
        pool = item["query_pool"]
        self.assertIsInstance(pool, list)
        self.assertEqual(len(pool), 288)
        self.assertEqual(item["stratum"], "complex")
        self.assertTrue(all(isinstance(query, tuple) for query in pool))
        self.assertTrue(all(len(query) == 3 for query in pool))
        self.assertTrue(all(isinstance(value, int) for u, v, _d in pool for value in u + v))
        self.assertEqual(dataset.query_pool(0), pool)
        self.assertTrue(Path(item["query_cache"]).is_file())
        self.assertEqual(dataset.query_cache_path(0), Path(item["query_cache"]))

        mask = item["mask"].squeeze().numpy()
        self.assertAlmostEqual(
            Verifier(mask, queries=pool).score(mask)["path"], 1.0
        )

        cache = json.loads(Path(item["query_cache"]).read_text())
        self.assertEqual(cache["pool_size"], 288)
        self.assertEqual(len(cache["queries"]), 288)
        self.assertEqual(len(cache["query_meta"]), 288)
        self.assertEqual(cache["query_meta_alignment"], "parallel_to_queries")
        self.assertEqual(cache["centerline_key"], "centerline")
        self.assertEqual(cache["native_polylines"]["points_key"], "polyline_points")
        self.assertEqual(cache["native_polylines"]["offsets_key"], "polyline_offsets")
        self.assertTrue(cache["native_polylines"]["may_extend_outside_crop"])
        self.assertIsNotNone(cache["source_geojson"])
        self.assertTrue((output / cache["npz"]).is_file())
        for meta in cache["query_meta"]:
            self.assertIn(meta["visible"], (True, False, None))
            self.assertIn(meta["visibility"], (
                "path_fully_inside_crop",
                "path_touches_crop_border",
                "no_raster_path",
                "crop_too_small_for_interior_visibility",
            ))
            self.assertEqual(len(meta["u_geo"]), 2)
            self.assertEqual(len(meta["v_geo"]), 2)
            if meta["u_native"] is not None:
                self.assertTrue(
                    {"line", "vertex", "x", "y", "distance_px"}
                    <= set(meta["u_native"])
                )
                self.assertTrue(
                    {"line", "vertex", "x", "y", "distance_px"}
                    <= set(meta["v_native"])
                )

        # The native centerline is retained separately for roundtrip checks.
        with np.load(item["npz"], allow_pickle=False) as payload:
            np.testing.assert_array_equal(
                item["centerline"].squeeze().numpy(), payload["centerline"]
            )
            self.assertGreater(int(payload["centerline"].sum()), 0)
            self.assertGreater(len(payload["polyline_offsets"]), 1)

    def test_legacy_val_manifest_and_holdout_never_used_as_dev(self):
        root, _ids = build_grid_dataset(
            self.base / "grid", rows=3, cols=3, tile_size=TILE_SIZE, dense=True
        )
        output = self.base / "legacy"
        manifest = prepare(
            root,
            output,
            crop_size=TILE_SIZE,
            train_limit=1,
            val_limit=1,
            holdout_limit=1,
            seed=2,
            buffer_m=BUFFER_M,
            spatial_buffer_tiles=0,
            query_pool_size=4,
        )
        self.assertEqual(manifest["legacy_val_limit"], 1)
        self.assertEqual(len(manifest["dev"]), 1)
        self.assertEqual(len(manifest["holdout"]), 1)
        self.assertNotIn("val", manifest)

        with self.assertRaises(ValueError):
            prepare(
                root,
                self.base / "conflict",
                crop_size=TILE_SIZE,
                dev_limit=5,
                val_limit=1,
                holdout_limit=1,
                query_pool_size=0,
            )

        # New manifest: val aliases dev; holdout is separate.
        new_dataset = RoadDataset(output / "manifest.json", "val")
        self.assertEqual(new_dataset.split_key, "dev")
        self.assertEqual(
            new_dataset.ids, [record["id"] for record in manifest["dev"]]
        )
        self.assertEqual(
            RoadDataset(output / "manifest.json", "dev").ids,
            [record["id"] for record in manifest["dev"]],
        )

        # Vintage manifest with a val list stays readable.
        legacy_record = dict(manifest["dev"][0])
        legacy_record.pop("query_cache", None)
        legacy_record.pop("applicability", None)
        legacy = {
            "format_version": 1,
            "train": [],
            "val": [legacy_record],
        }
        legacy_dataset = RoadDataset(legacy, "val", root=output)
        self.assertEqual(legacy_dataset.split_key, "val")
        self.assertEqual(len(legacy_dataset), 1)
        legacy_item = legacy_dataset[0]
        self.assertEqual(legacy_item["source_id"], legacy_record["source_id"])
        self.assertEqual(tuple(legacy_item["centerline"].shape), (1, TILE_SIZE, TILE_SIZE))
        self.assertEqual(legacy_item["query_pool"], [])
        self.assertIsNone(legacy_item["query_cache"])
        self.assertEqual(legacy_item["stratum"], "unclassified")
        # Old val also serves as the development split.
        self.assertEqual(
            RoadDataset(legacy, "dev", root=output).split_key, "val"
        )

        # A holdout-only manifest must never be silently used as val/dev.
        holdout_only = {"format_version": 2, "holdout": manifest["holdout"]}
        with self.assertRaises(ValueError):
            RoadDataset(holdout_only, "val", root=output)
        with self.assertRaises(ValueError):
            RoadDataset(holdout_only, "dev", root=output)

    def test_prepare_defaults_match_model_test_targets(self):
        signature = inspect.signature(prepare)
        self.assertEqual(signature.parameters["train_limit"].default, 400)
        self.assertEqual(signature.parameters["dev_limit"].default, 60)
        self.assertEqual(signature.parameters["holdout_limit"].default, 40)
        self.assertEqual(
            signature.parameters["spatial_buffer_tiles"].default, 1
        )
        self.assertEqual(signature.parameters["query_pool_size"].default, 288)

    def test_crop_specs_are_fixed_non_overlapping_native_pixels(self):
        root = build_synthetic_dataset(self.base / "dataset")
        info = read_geotiff_info(
            root
            / "train"
            / "AOI_2_Vegas"
            / "PS-RGB"
            / "SN3_roads_train_AOI_2_Vegas_PS-RGB_img101.tif"
        )
        specs = crop_specs(info, CROP)
        self.assertEqual(len(specs), (TILE_SIZE // CROP) ** 2)
        self.assertEqual(specs[0], (0, 0))
        self.assertEqual(specs[-1], (TILE_SIZE - CROP, TILE_SIZE - CROP))
        # Stride is exactly one crop; the last 64 % 16 == 0 pixels are included.
        for row, col in specs:
            self.assertEqual(row % CROP, 0)
            self.assertEqual(col % CROP, 0)


def dataset_ids(records):
    return [str(record["id"]) for record in records]


if __name__ == "__main__":
    unittest.main()
