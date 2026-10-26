"""SpaceNet 3 road data preparation and a manifest-backed dataset.

This module only prepares data.  It does not train a model and it does not
implement rewards or verification.

Pipeline implemented here
-------------------------
1. Discover the native PS-RGB GeoTIFF tiles and their native
   ``geojson_roads`` centerline labels.
2. Read each tile's GeoTIFF georeferencing directly from its TIFF tags.  No
   rasterio/GDAL installation is required.
3. Split *whole source tiles* into disjoint ``train``/``dev``/``holdout``
   geographic regions before any cropping.  Two seeded connected grow-regions
   are selected over the tile grid derived from geographic bounds, and a
   Chebyshev ``spatial_buffer_tiles`` ring around both evaluation regions is
   excluded from training (and from the other evaluation region).  Image
   numbers are never used as a proxy for location.
4. Cut fixed, non-overlapping ``crop_size`` windows at native resolution from
   each selected source tile.  No random cropping and no resizing.
5. Rasterize native GeoJSON centerlines into a metric buffer mask.  Centerlines
   outside a crop are still considered by the distance transform when their
   buffer could reach into the crop.
6. Build an offline query pool per crop from the *GT mask skeleton* using the
   same graph and sampler as :mod:`roadflow.verifier`.  The pool contains at
   least ``query_pool_size`` (default 288) unique verifier-compatible queries
   when the graph admits them; every shortfall is recorded explicitly.  Native
   crop-local polylines and endpoint references are retained for separately
   measuring paths on the native graph.
7. Compute a per-crop data-applicability report (query count/length, junction,
   component and cycle counts, empty/single/complex category, and the fraction
   of queries with a complete visible path).  Metrics that cannot be measured
   are written as JSON ``null`` together with an explicit reason.
8. Save one compressed ``.npz`` per crop, one query-cache JSON per crop, an
   ``applicability.json`` report, and a JSON manifest with split provenance.

NPZ contents
------------
``image``            float32, shape (3, H, W), raw sensor DN (not normalized)
``mask``             float32, shape (H, W), binary buffered-centerline mask
``centerline``       uint8, shape (H, W), one-pixel native centerline raster
``polyline_points``  float32, shape (P, 2), crop-local (x, y) native polylines
``polyline_offsets`` int64, shape (E + 1,), offsets into ``polyline_points``
``crop_origin``      int32, shape (2,), source-image (x0, y0)
``pixel_size_m``     float32, shape (2,), (row_metres, column_metres)

Graph polylines are a convenience for native-graph measurements.  Their
coordinates are crop-local and may extend outside ``[0, W) x [0, H)``; callers
that build native path queries must enforce their own in-crop visibility rule.

``RoadDataset`` returns the image divided by the fixed constant 2047.0
(SpaceNet PS-RGB is 11-bit).  The npz stores raw DN; normalization is never
computed per crop.

Usage
-----
Command line::

    python -m roadflow.data --root SpaceNet3_AOI_2_Vegas \
        --output data/spacenet_vegas_256 --crop-size 256 \
        --train-limit 400 --dev-limit 60 --holdout-limit 40 --seed 0

Python::

    from roadflow.data import prepare, RoadDataset
    manifest = prepare("SpaceNet3_AOI_2_Vegas", "data/spacenet_vegas_256")
    train = RoadDataset("data/spacenet_vegas_256/manifest.json", "train")
    item = train[0]  # image (3,H,W), mask (1,H,W), source_id, centerline,
                     # query_pool [(u, v, distance), ...], query_cache path

Split compatibility
-------------------
New manifests write only ``train``, ``dev`` and ``holdout``.  ``RoadDataset``
still reads vintage manifests with a legacy ``val`` list (``split="val"``),
and aliases ``val`` to ``dev`` for new manifests.  It never falls back to
``holdout`` for ``val`` or ``dev``.

Limitations / fixed assumptions
-------------------------------
* CRS support is limited to geographic CRS (degrees), as used by SpaceNet 3
  (EPSG:4326 / CRS84), plus a simple projected-CRS fallback that assumes the
  linear unit recorded in the GeoTIFF GeoKeys.  No datum reprojection is done.
* GeoJSON coordinates are assumed to be longitude/latitude in degrees.
* Tiles without a ``geojson_roads`` file are treated as empty-road negatives,
  matching SpaceNet 3's documented eight empty training tiles.
* Buffer width is a physical width in metres, fixed for every crop.  Pixel
  spacing is anisotropic for EPSG:4326 tiles, so the buffer is applied with
  separate row/column metre scales.
* The complete-visible-path flag uses the centerline-raster shortest path
  inside the crop: it excludes paths that use the outermost crop row/column.
  It cannot detect a native road that leaves and re-enters the crop along a
  different alignment; native polylines and endpoint references are cached for
  that separate roundtrip check.
* Missing source TIFFs raise; small tiles that cannot fit one crop are skipped
  and reported in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as _skimage_line

SPLITS = ("train", "dev", "holdout")
_LEGACY_SPLIT_ALIASES = {"val": "dev"}
_DATASET_SPLITS = SPLITS + tuple(_LEGACY_SPLIT_ALIASES)
_FORMAT_VERSION = 2
_QUERY_FORMAT_VERSION = 1
_APPLICABILITY_FORMAT_VERSION = 1
_DEFAULT_BUFFER_M = 6.0
_DEFAULT_QUERY_POOL_SIZE = 288
_DEFAULT_SPATIAL_BUFFER_TILES = 1
_MAX_QUERY_ATTEMPTS = 16
_IMAGE_DIVISOR = 2047.0  # PS-RGB is 11-bit; fixed normalization, no per-crop stats
_AOI_RE = re.compile(r"AOI_\d+_[A-Za-z0-9]+")
_PRODUCT_TOKEN_RE = re.compile(
    r"_(?:PS-RGB|PS-MS|PAN|geojson_roads_speed|geojson_roads)_(.+)$"
)
_VISIBILITY_RULE = (
    "A query is complete_visible iff every pixel on the crop-local GT-mask "
    "skeleton shortest path lies strictly inside the crop (not on the outer "
    "row/column). This excludes paths truncated at the crop edge. It cannot "
    "detect a native road that leaves and re-enters along a different "
    "alignment; use the cached crop-local native polylines and endpoint "
    "references for that separate native-graph roundtrip."
)


# ---------------------------------------------------------------------------
# GeoTIFF reading (stdlib + tifffile; no rasterio)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeoRaster:
    """Minimal affine georeferencing for one raster.

    The transform maps pixel coordinates to CRS coordinates::

        X = a * col + b * row + c
        Y = d * col + e * row + f
    """

    width: int
    height: int
    transform: tuple[float, float, float, float, float, float]
    crs_epsg: int | None = None
    is_geographic: bool = True
    linear_unit_m: float = 1.0

    def pixel_to_geo(self, col: float, row: float) -> tuple[float, float]:
        a, b, c, d, e, f = self.transform
        return a * col + b * row + c, d * col + e * row + f

    def geo_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        a, b, c, d, e, f = self.transform
        det = a * e - b * d
        if abs(det) < 1e-20:
            raise ValueError("Degenerate GeoTIFF affine transform")
        dx, dy = x - c, y - f
        return (e * dx - b * dy) / det, (-d * dx + a * dy) / det

    def geo_bounds(self) -> tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax) of the raster edges."""
        corners = (
            self.pixel_to_geo(0.0, 0.0),
            self.pixel_to_geo(float(self.width), 0.0),
            self.pixel_to_geo(0.0, float(self.height)),
            self.pixel_to_geo(float(self.width), float(self.height)),
        )
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def center(self) -> tuple[float, float]:
        return self.pixel_to_geo(self.width / 2.0, self.height / 2.0)

    def pixel_size_m(self, latitude: float | None = None) -> tuple[float, float]:
        """Approximate ground size (row_metres, column_metres) of a pixel.

        For geographic CRS the local meridian/parallel scale at ``latitude``
        is used.  For projected CRS the GeoKey linear unit is applied.
        """
        a, b, c, d, e, f = self.transform
        col_units = math.hypot(a, d)
        row_units = math.hypot(b, e)
        if self.is_geographic:
            if latitude is None:
                latitude = self.center()[1]
            lat = math.radians(float(latitude))
            m_per_deg_lat = (
                111132.92
                - 559.82 * math.cos(2.0 * lat)
                + 1.175 * math.cos(4.0 * lat)
                - 0.0023 * math.cos(6.0 * lat)
            )
            m_per_deg_lon = (
                111412.84 * math.cos(lat)
                - 93.5 * math.cos(3.0 * lat)
                + 0.118 * math.cos(5.0 * lat)
            )
            return row_units * m_per_deg_lat, col_units * m_per_deg_lon
        unit = float(self.linear_unit_m or 1.0)
        return row_units * unit, col_units * unit


def _parse_geokeys(value: Sequence[int] | None) -> dict[int, int]:
    if not value or len(value) < 4:
        return {}
    count = int(value[3])
    keys: dict[int, int] = {}
    for index in range(count):
        offset = 4 + 4 * index
        if offset + 3 >= len(value):
            break
        key, location, key_count, key_value = (int(v) for v in value[offset : offset + 4])
        if location == 0 and key_count == 1:
            keys[key] = key_value
    return keys


def _linear_unit_metres(unit_code: int | None) -> float:
    if unit_code == 9002:  # international foot
        return 0.3048
    if unit_code == 9003:  # US survey foot
        return 1200.0 / 3937.0
    return 1.0


def _info_from_page(page: Any) -> GeoRaster:
    width = int(getattr(page, "imagewidth"))
    height = int(getattr(page, "imagelength"))
    tags = page.tags

    transformation = tags.get("ModelTransformationTag")
    if transformation is not None:
        matrix = [float(v) for v in transformation.value]
        if len(matrix) < 8:
            raise ValueError("Malformed ModelTransformationTag")
        a, b, c = matrix[0], matrix[1], matrix[3]
        d, e, f = matrix[4], matrix[5], matrix[7]
    else:
        scale_tag = tags.get("ModelPixelScaleTag")
        tiepoint_tag = tags.get("ModelTiepointTag")
        if scale_tag is None or tiepoint_tag is None:
            raise ValueError("GeoTIFF has no usable georeferencing tags")
        sx, sy, _sz = (float(v) for v in scale_tag.value[:3])
        tie = [float(v) for v in tiepoint_tag.value[:6]]
        i, j, _k, x, y, _z = tie
        a, b, c = sx, 0.0, x - i * sx
        d, e, f = 0.0, -sy, y + j * sy

    geo_keys_tag = tags.get("GeoKeyDirectoryTag")
    keys = _parse_geokeys(geo_keys_tag.value if geo_keys_tag is not None else None)
    gt_model = keys.get(1024)
    geographic_epsg = keys.get(2048)
    projected_epsg = keys.get(3072)
    unit_code = keys.get(3076)

    if gt_model == 1:
        is_geographic = False
        crs_epsg = projected_epsg
    elif gt_model == 2:
        is_geographic = True
        crs_epsg = geographic_epsg
    else:
        # Fall back to inspecting the tiepoint magnitudes.
        is_geographic = (
            abs(c) <= 180.0
            and abs(f) <= 90.0
            and abs(a) <= 1.0
            and abs(e) <= 1.0
        )
        crs_epsg = projected_epsg if not is_geographic else geographic_epsg

    if geographic_epsg in (4326, 4269, 4258) and gt_model != 1:
        is_geographic = True
        crs_epsg = geographic_epsg

    return GeoRaster(
        width=width,
        height=height,
        transform=(a, b, c, d, e, f),
        crs_epsg=crs_epsg,
        is_geographic=is_geographic,
        linear_unit_m=_linear_unit_metres(unit_code),
    )


def read_geotiff_info(path: str | Path) -> GeoRaster:
    """Read only the georeferencing metadata of a GeoTIFF."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        return _info_from_page(tif.pages[0])


def _to_chw(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF array shape: {arr.shape}")
    # tifffile normally returns HWC.  Also accept CHW when unambiguous.
    if arr.shape[0] <= 4 and arr.shape[-1] > 4:
        arr = np.transpose(arr, (1, 2, 0))
    channels = arr.shape[2]
    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif channels > 3:
        arr = arr[:, :, :3]
    elif channels == 2:
        arr = np.repeat(arr[:, :, :1], 3, axis=2)
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)), dtype=np.float32)


def read_geotiff_image(path: str | Path) -> np.ndarray:
    """Read a GeoTIFF as float32 raw DN in CHW order."""
    import tifffile

    return _to_chw(tifffile.imread(path))


# ---------------------------------------------------------------------------
# Native GeoJSON centerlines
# ---------------------------------------------------------------------------


def _iter_line_coordinates(geometry: Mapping[str, Any] | None) -> Iterable[Any]:
    if not geometry:
        return
    kind = geometry.get("type")
    if kind == "LineString":
        yield geometry.get("coordinates") or []
    elif kind == "MultiLineString":
        for coordinates in geometry.get("coordinates") or []:
            yield coordinates
    elif kind == "GeometryCollection":
        for child in geometry.get("geometries") or []:
            yield from _iter_line_coordinates(child)


def load_centerlines(path: str | Path | None) -> list[np.ndarray]:
    """Load native GeoJSON centerlines as a list of (N, 2) lon/lat arrays.

    A missing path means "no annotation" and returns an empty list.  Only line
    geometries are used; polygons and points are ignored.
    """
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        geometries = [f.get("geometry") for f in payload.get("features") or []]
    elif payload.get("type") == "Feature":
        geometries = [payload.get("geometry")]
    else:
        geometries = [payload]

    lines: list[np.ndarray] = []
    for geometry in geometries:
        for coordinates in _iter_line_coordinates(geometry):
            array = np.asarray(coordinates, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] < 2 or array.shape[0] < 2:
                continue
            array = array[:, :2]
            if np.isfinite(array).all():
                lines.append(array)
    return lines


# ---------------------------------------------------------------------------
# Source discovery and geographic region split
# ---------------------------------------------------------------------------


@dataclass
class SourceTile:
    image_path: Path
    geojson_path: Path | None
    source_id: str
    token: str
    info: GeoRaster


def _extract_aoi(path: Path) -> str:
    for part in path.parts:
        match = _AOI_RE.search(part)
        if match:
            return match.group(0)
    return ""


def _product_token(path: Path) -> str:
    match = _PRODUCT_TOKEN_RE.search(path.stem)
    return match.group(1) if match else path.stem


def _source_id_for(path: Path, token: str) -> str:
    aoi = _extract_aoi(path)
    if aoi and not token.startswith(aoi):
        return f"{aoi}_{token}"
    return token


def _geojson_rank(path: Path) -> tuple[int, str]:
    parts = path.parts
    if "geojson_roads" in parts:
        return 0, str(path)
    if "geojson_roads_speed" in parts:
        return 2, str(path)
    return 1, str(path)


def _geojson_index(search_root: Path, image_paths: Sequence[Path]) -> dict[tuple[str, str], Path]:
    files: set[Path] = set(search_root.rglob("*.geojson"))
    for image_path in image_paths:
        # Also support root pointing directly at .../PS-RGB.
        for candidate in (
            image_path.parent.parent / "geojson_roads",
            image_path.parent.parent / "geojson_roads_speed",
        ):
            if candidate.is_dir():
                files.update(candidate.glob("*.geojson"))
    index: dict[tuple[str, str], Path] = {}
    for path in sorted(files, key=_geojson_rank):
        key = (_extract_aoi(path), _product_token(path))
        index.setdefault(key, path)
    return index


def discover_sources(root: str | Path) -> list[SourceTile]:
    """Discover PS-RGB TIFFs and matching native centerline GeoJSON files.

    If ``root`` contains a ``train`` directory, only that subtree is searched,
    so the unlabeled ``test_public`` imagery is never mixed into preparation.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    search_root = root / "train" if (root / "train").is_dir() else root
    image_paths = sorted(
        p
        for p in search_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".tif", ".tiff")
        and "PS-RGB" in p.parent.name
        and "test_public" not in p.parts
    )
    if not image_paths:
        image_paths = sorted(
            p
            for p in search_root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in (".tif", ".tiff")
            and "test_public" not in p.parts
        )
    if not image_paths:
        raise FileNotFoundError(f"No GeoTIFF imagery found under: {root}")

    geojson_by_key = _geojson_index(search_root, image_paths)
    sources: list[SourceTile] = []
    seen: set[str] = set()
    for image_path in image_paths:
        token = _product_token(image_path)
        geojson_path = geojson_by_key.get((_extract_aoi(image_path), token))
        source_id = _source_id_for(image_path, token)
        if source_id in seen:
            raise ValueError(
                f"Duplicate source id {source_id!r}; check AOI/image naming under {root}"
            )
        seen.add(source_id)
        sources.append(
            SourceTile(
                image_path=image_path,
                geojson_path=geojson_path,
                source_id=source_id,
                token=token,
                info=read_geotiff_info(image_path),
            )
        )
    return sources


def crop_specs(info: GeoRaster, crop_size: int) -> list[tuple[int, int]]:
    """Fixed non-overlapping crop grid, in (row, col) source pixels."""
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    rows = range(0, info.height - crop_size + 1, crop_size)
    cols = range(0, info.width - crop_size + 1, crop_size)
    return [(int(row), int(col)) for row in rows for col in cols]


def _grid_cells(sources: Sequence[SourceTile]) -> dict[str, tuple[int, int]]:
    bounds = np.asarray([source.info.geo_bounds() for source in sources], dtype=np.float64)
    centers = np.asarray([source.info.center() for source in sources], dtype=np.float64)
    widths = bounds[:, 2] - bounds[:, 0]
    heights = bounds[:, 3] - bounds[:, 1]
    tile_w = float(np.median(widths))
    tile_h = float(np.median(heights))
    if not (math.isfinite(tile_w) and tile_w > 0 and math.isfinite(tile_h) and tile_h > 0):
        raise ValueError("Cannot derive a geographic tile grid from degenerate bounds")
    origin_x = float(centers[:, 0].min()) - tile_w / 2.0
    top_y = float(centers[:, 1].max()) + tile_h / 2.0
    cells: dict[str, tuple[int, int]] = {}
    for source, (cx, cy) in zip(sources, centers):
        col = int(round((cx - origin_x - tile_w / 2.0) / tile_w))
        row = int(round((top_y - cy - tile_h / 2.0) / tile_h))
        cells[source.source_id] = (row, col)
    return cells


def _neighbours(cell: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = cell
    return [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]


@dataclass(frozen=True)
class GeographicSplit:
    """Whole-source-tile train/dev/holdout assignment and isolation buffer."""

    cells: dict[str, tuple[int, int]]
    train_source_ids: list[str]
    dev_source_ids: list[str]
    holdout_source_ids: list[str]
    buffer_source_ids: list[str]
    excluded_sources: list[dict[str, Any]]
    dev_grid_cells: list[tuple[int, int]]
    holdout_grid_cells: list[tuple[int, int]]
    dev_anchor: str | None
    holdout_anchor: str | None
    capacity: dict[str, int]


def _cell_capacity(
    cell: tuple[int, int],
    cell_sources: Mapping[tuple[int, int], Sequence[str]],
    source_crops: Mapping[str, int],
) -> int:
    return sum(source_crops[source_id] for source_id in cell_sources[cell])


def _grow_region(
    cell_sources: Mapping[tuple[int, int], Sequence[str]],
    cell_centers: Mapping[tuple[int, int], tuple[float, float]],
    source_crops: Mapping[str, int],
    allowed: Iterable[tuple[int, int]],
    limit: int | None,
    seed: int,
    anchor: tuple[int, int] | None = None,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    """Grow a 4-connected region of whole grid cells to cover ``limit`` crops.

    The anchor is either supplied or chosen by ``seed``.  If the connected
    frontier is exhausted the growth jumps to the nearest unvisited allowed
    cell, so a non-contiguous layout still terminates deterministically.
    """
    allowed_cells = sorted(set(allowed))
    if not allowed_cells or (limit is not None and limit <= 0):
        return [], None
    allowed_set = set(allowed_cells)
    rng = random.Random(int(seed))
    if anchor is None or anchor not in allowed_cells:
        anchor = allowed_cells[rng.randrange(len(allowed_cells))]

    selected: set[tuple[int, int]] = set()
    selected_order: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = {anchor}
    queue: deque[tuple[int, int]] = deque([anchor])
    capacity = 0

    while limit is None or capacity < limit:
        if not queue:
            unvisited = [cell for cell in allowed_cells if cell not in visited]
            if not unvisited:
                break
            if selected_order:
                px = float(np.mean([cell_centers[c][0] for c in selected_order]))
                py = float(np.mean([cell_centers[c][1] for c in selected_order]))
            else:
                px, py = cell_centers[anchor]
            next_cell = min(
                unvisited,
                key=lambda c: (
                    (cell_centers[c][0] - px) ** 2 + (cell_centers[c][1] - py) ** 2,
                    c,
                ),
            )
            visited.add(next_cell)
            queue.append(next_cell)

        cell = queue.popleft()
        if cell in selected:
            continue
        selected.add(cell)
        selected_order.append(cell)
        capacity += _cell_capacity(cell, cell_sources, source_crops)
        for neighbour in _neighbours(cell):
            if neighbour in cell_sources and neighbour in allowed_set and neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return sorted(selected), anchor


def _farthest_anchor(
    allowed: Iterable[tuple[int, int]],
    reference_cells: Sequence[tuple[int, int]],
    cell_centers: Mapping[tuple[int, int], tuple[float, float]],
    seed: int,
) -> tuple[int, int] | None:
    allowed_cells = sorted(set(allowed))
    if not allowed_cells:
        return None
    rng = random.Random(int(seed))
    if not reference_cells:
        return allowed_cells[rng.randrange(len(allowed_cells))]
    px = float(np.mean([cell_centers[c][0] for c in reference_cells]))
    py = float(np.mean([cell_centers[c][1] for c in reference_cells]))
    ranked = sorted(
        allowed_cells,
        key=lambda c: (
            -((cell_centers[c][0] - px) ** 2 + (cell_centers[c][1] - py) ** 2),
            c,
        ),
    )
    # Choose among the farthest tenth so the seed still matters while keeping
    # the two evaluation regions clearly separated.
    top = max(1, len(ranked) // 10)
    return ranked[rng.randrange(top)]


def _cells_within(
    base_cells: Iterable[tuple[int, int]],
    all_cells: Iterable[tuple[int, int]],
    radius: int,
) -> set[tuple[int, int]]:
    """Chebyshev grid cells within ``radius`` of ``base_cells`` (base excluded)."""
    if radius <= 0:
        return set()
    base = set(base_cells)
    if not base:
        return set()
    return {
        cell
        for cell in set(all_cells)
        if cell not in base
        and min(
            max(abs(cell[0] - b[0]), abs(cell[1] - b[1])) for b in base
        ) <= radius
    }


def _nearest_cell_distance(
    cell: tuple[int, int], cells: Sequence[tuple[int, int]]
) -> int | None:
    if not cells:
        return None
    return min(max(abs(cell[0] - b[0]), abs(cell[1] - b[1])) for b in cells)


def select_geographic_regions(
    sources: Sequence[SourceTile],
    crop_size: int,
    dev_limit: int | None,
    holdout_limit: int | None,
    seed: int,
    spatial_buffer_tiles: int = _DEFAULT_SPATIAL_BUFFER_TILES,
) -> GeographicSplit:
    """Assign whole source tiles to train/dev/holdout before cropping.

    ``dev`` and ``holdout`` are disjoint seeded connected grow-regions on the
    tile grid.  A Chebyshev ring of ``spatial_buffer_tiles`` cells around each
    evaluation region is excluded from training and from the other evaluation
    region, preventing adjacent-tile leakage.  Buffer sources are returned
    explicitly so the manifest can record them.
    """
    if spatial_buffer_tiles < 0:
        raise ValueError("spatial_buffer_tiles must be non-negative")
    cells = _grid_cells(sources)
    source_by_id = {source.source_id: source for source in sources}
    source_crops = {
        source.source_id: len(crop_specs(source.info, crop_size)) for source in sources
    }
    cell_sources: dict[tuple[int, int], list[str]] = {}
    for source in sources:
        cell_sources.setdefault(cells[source.source_id], []).append(source.source_id)
    for value in cell_sources.values():
        value.sort()

    cell_centers: dict[tuple[int, int], tuple[float, float]] = {}
    for cell, ids in cell_sources.items():
        coords = np.asarray(
            [source_by_id[source_id].info.center() for source_id in ids],
            dtype=np.float64,
        )
        cell_centers[cell] = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

    all_cells = set(cell_sources)
    dev_cells, dev_anchor_cell = _grow_region(
        cell_sources, cell_centers, source_crops, all_cells, dev_limit, seed
    )
    dev_buffer = _cells_within(dev_cells, all_cells, spatial_buffer_tiles)

    holdout_allowed = all_cells - set(dev_cells) - dev_buffer
    if holdout_limit is not None and holdout_limit <= 0:
        holdout_anchor_cell = None
    else:
        holdout_anchor_cell = _farthest_anchor(
            holdout_allowed, dev_cells, cell_centers, seed + 1
        )
    holdout_cells, _ = _grow_region(
        cell_sources,
        cell_centers,
        source_crops,
        holdout_allowed,
        holdout_limit,
        seed + 2,
        anchor=holdout_anchor_cell,
    )
    holdout_buffer = (
        _cells_within(holdout_cells, all_cells, spatial_buffer_tiles)
        - set(dev_cells)
        - dev_buffer
        - set(holdout_cells)
    )
    buffer_cells = dev_buffer | holdout_buffer
    train_cells = all_cells - set(dev_cells) - dev_buffer - set(holdout_cells) - holdout_buffer

    def ids_for(selected_cells: Iterable[tuple[int, int]]) -> list[str]:
        return sorted(
            source_id
            for cell in selected_cells
            for source_id in cell_sources[cell]
        )

    dev_source_ids = ids_for(dev_cells)
    holdout_source_ids = ids_for(holdout_cells)
    train_source_ids = ids_for(train_cells)
    buffer_source_ids = ids_for(buffer_cells)

    excluded_sources: list[dict[str, Any]] = []
    for source_id in buffer_source_ids:
        cell = cells[source_id]
        distance_dev = _nearest_cell_distance(cell, dev_cells)
        distance_holdout = _nearest_cell_distance(cell, holdout_cells)
        if distance_dev is not None and (
            distance_holdout is None or distance_dev <= distance_holdout
        ):
            nearest, distance = "dev", distance_dev
        else:
            nearest, distance = "holdout", distance_holdout
        excluded_sources.append(
            {
                "source_id": source_id,
                "reason": "spatial_buffer",
                "nearest_eval_region": nearest,
                "grid_cell": [int(cell[0]), int(cell[1])],
                "grid_distance": int(distance) if distance is not None else None,
            }
        )

    def capacity(selected_ids: Iterable[str]) -> int:
        return sum(source_crops[source_id] for source_id in selected_ids)

    return GeographicSplit(
        cells=cells,
        train_source_ids=train_source_ids,
        dev_source_ids=dev_source_ids,
        holdout_source_ids=holdout_source_ids,
        buffer_source_ids=buffer_source_ids,
        excluded_sources=excluded_sources,
        dev_grid_cells=[tuple(map(int, cell)) for cell in sorted(dev_cells)],
        holdout_grid_cells=[tuple(map(int, cell)) for cell in sorted(holdout_cells)],
        dev_anchor=cell_sources[dev_anchor_cell][0] if dev_anchor_cell else None,
        holdout_anchor=(
            cell_sources[holdout_anchor_cell][0] if holdout_anchor_cell else None
        ),
        capacity={
            "train": capacity(train_source_ids),
            "dev": capacity(dev_source_ids),
            "holdout": capacity(holdout_source_ids),
        },
    )


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CropLabels:
    mask: np.ndarray
    centerline: np.ndarray
    polyline_points: np.ndarray
    polyline_offsets: np.ndarray


def _clip_segment(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Liang-Barsky clip of a segment to [0, width-1] x [0, height-1]."""
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0),
        (dx, width - 1 - x0),
        (-dy, y0),
        (dy, height - 1 - y0),
    ):
        if p == 0.0:
            if q < 0.0:
                return None
        else:
            ratio = q / p
            if p < 0.0:
                if ratio > t1:
                    return None
                t0 = max(t0, ratio)
            else:
                if ratio < t0:
                    return None
                t1 = min(t1, ratio)
    if t0 > t1:
        return None
    return (x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy)


def _draw_polyline(canvas: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
    height, width = canvas.shape
    if len(xs) == 1:
        col, row = int(round(xs[0])), int(round(ys[0]))
        if 0 <= row < height and 0 <= col < width:
            canvas[row, col] = 1
        return
    for index in range(len(xs) - 1):
        clipped = _clip_segment(
            float(xs[index]),
            float(ys[index]),
            float(xs[index + 1]),
            float(ys[index + 1]),
            width,
            height,
        )
        if clipped is None:
            continue
        x0, y0, x1, y1 = clipped
        rows, cols = _skimage_line(
            int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1))
        )
        keep = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        canvas[rows[keep], cols[keep]] = 1


def rasterize_crop(
    info: GeoRaster,
    polylines: Sequence[np.ndarray],
    x0: int,
    y0: int,
    size: int,
    buffer_m: float,
) -> CropLabels:
    """Rasterize crop-local buffered centerlines and optional graph polylines.

    The buffer is measured in metres using the local latitude.  A margin of at
    least ``buffer_m`` is rasterized around the crop so roads just outside the
    crop cannot be lost from the buffer.
    """
    if buffer_m <= 0:
        raise ValueError("buffer_m must be positive")
    if size <= 0:
        raise ValueError("size must be positive")

    latitude = info.pixel_to_geo(x0 + size / 2.0, y0 + size / 2.0)[1]
    row_m, col_m = info.pixel_size_m(latitude)
    if not (row_m > 0 and col_m > 0):
        raise ValueError("GeoTIFF has non-positive pixel spacing")

    margin = max(1, int(math.ceil(buffer_m / min(row_m, col_m))) + 1)
    canvas_size = size + 2 * margin
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    a, b, c, d, e, f = info.transform
    det = a * e - b * d
    if abs(det) < 1e-20:
        raise ValueError("Degenerate GeoTIFF affine transform")
    graph_lines: list[np.ndarray] = []

    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] == 0:
            continue
        lon, lat = points[:, 0], points[:, 1]
        dx, dy = lon - c, lat - f
        pixel_x = (e * dx - b * dy) / det
        pixel_y = (-d * dx + a * dy) / det
        if not (np.isfinite(pixel_x).all() and np.isfinite(pixel_y).all()):
            continue

        # Keep a graph copy when the native polyline intersects the crop.
        if (
            pixel_x.max() >= x0
            and pixel_x.min() <= x0 + size
            and pixel_y.max() >= y0
            and pixel_y.min() <= y0 + size
        ):
            graph_lines.append(
                np.stack((pixel_x - x0, pixel_y - y0), axis=1).astype(np.float32)
            )

        local_x = pixel_x - (x0 - margin)
        local_y = pixel_y - (y0 - margin)
        _draw_polyline(canvas, local_x, local_y)

    if canvas.any():
        distance = ndi.distance_transform_edt(
            canvas == 0, sampling=(row_m, col_m)
        )
        buffered = distance <= buffer_m
    else:
        # scipy's EDT uses implicit zeros outside an all-True array; an empty
        # centerline must produce an empty mask instead of edge artifacts.
        buffered = np.zeros_like(canvas, dtype=bool)

    central = (slice(margin, margin + size), slice(margin, margin + size))
    mask = buffered[central].astype(np.float32)
    centerline = canvas[central].copy()

    if graph_lines:
        lengths = np.asarray([len(line) for line in graph_lines], dtype=np.int64)
        points = np.concatenate(graph_lines, axis=0)
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    else:
        points = np.zeros((0, 2), dtype=np.float32)
        offsets = np.zeros(1, dtype=np.int64)

    return CropLabels(
        mask=mask,
        centerline=centerline,
        polyline_points=points,
        polyline_offsets=offsets,
    )


# ---------------------------------------------------------------------------
# Crop selection
# ---------------------------------------------------------------------------


def _hash_key(*parts: Any) -> bytes:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()


def select_crops(
    sources: Sequence[SourceTile],
    crop_size: int,
    limit: int | None,
    seed: int,
    split: str,
) -> list[tuple[SourceTile, tuple[int, int]]]:
    """Choose fixed crop windows, minimizing source TIFF reads.

    Whole source tiles are taken in a seeded order until their crop capacity
    covers ``limit``; crops are then taken round-robin across those tiles, in a
    seeded per-tile order.  This keeps spatial coverage within each tile while
    reading only as many source TIFFs as needed.
    """
    ordered = sorted(
        sources, key=lambda s: _hash_key("source", seed, split, s.source_id)
    )
    chosen: list[SourceTile] = []
    capacity = 0
    for source in ordered:
        if limit is not None and capacity >= limit:
            break
        chosen.append(source)
        capacity += len(crop_specs(source.info, crop_size))

    target = capacity if limit is None else min(limit, capacity)
    if target <= 0 or not chosen:
        return []

    specs_by_id: dict[str, list[tuple[int, int]]] = {}
    for source in chosen:
        specs = crop_specs(source.info, crop_size)
        specs.sort(
            key=lambda rc: _hash_key(
                "crop", seed, split, source.source_id, rc[0], rc[1]
            )
        )
        specs_by_id[source.source_id] = specs

    selected: list[tuple[SourceTile, tuple[int, int]]] = []
    positions = {source.source_id: 0 for source in chosen}
    while len(selected) < target:
        progressed = False
        for source in chosen:
            index = positions[source.source_id]
            specs = specs_by_id[source.source_id]
            if index >= len(specs):
                continue
            selected.append((source, specs[index]))
            positions[source.source_id] = index + 1
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected


# ---------------------------------------------------------------------------
# Offline query pool and applicability report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CropQueryContext:
    """Per-crop information needed to cache a verifier-compatible query pool."""

    crop_id: str
    split: str
    source_id: str
    npz: str
    source_geojson: str | None
    info: GeoRaster
    x0: int
    y0: int
    crop_size: int
    pixel_size_m: tuple[float, float]
    requested: int
    seed: int
    control_spacing: int


def _nearest_native(
    point_yx: tuple[int, int],
    polyline_points: np.ndarray,
    polyline_offsets: np.ndarray,
) -> dict[str, Any] | None:
    """Nearest native polyline vertex to a raster query endpoint."""
    if polyline_points.size == 0 or len(polyline_offsets) < 2:
        return None
    row, col = point_yx
    delta_x = polyline_points[:, 0].astype(np.float64) - float(col)
    delta_y = polyline_points[:, 1].astype(np.float64) - float(row)
    distances_sq = delta_x * delta_x + delta_y * delta_y
    flat_index = int(np.argmin(distances_sq))
    line_index = int(np.searchsorted(polyline_offsets, flat_index, side="right") - 1)
    line_index = min(max(line_index, 0), len(polyline_offsets) - 2)
    vertex_index = int(flat_index - polyline_offsets[line_index])
    return {
        "line": line_index,
        "vertex": vertex_index,
        "x": float(polyline_points[flat_index, 0]),
        "y": float(polyline_points[flat_index, 1]),
        "distance_px": float(math.sqrt(float(distances_sq[flat_index]))),
    }


def _path_visibility(
    graph: Any,
    u: tuple[int, int],
    v: tuple[int, int],
    crop_size: int,
) -> tuple[bool | None, str]:
    """Whether the crop-local skeleton shortest path avoids the outer border."""
    import networkx as nx

    if crop_size <= 2:
        return None, "crop_too_small_for_interior_visibility"
    try:
        _distance, path = nx.bidirectional_dijkstra(graph, u, v, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, "no_raster_path"
    if all(
        0 < row < crop_size - 1 and 0 < col < crop_size - 1 for row, col in path
    ):
        return True, "path_fully_inside_crop"
    return False, "path_touches_crop_border"


def _length_stats(lengths: Sequence[float]) -> dict[str, float] | None:
    if not lengths:
        return None
    values = np.asarray(lengths, dtype=np.float64)
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _metric_stats(values: Sequence[int | float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _build_query_cache(
    graph: Any,
    labels: CropLabels,
    context: CropQueryContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one verifier-compatible query cache and per-crop applicability.

    The query graph is exactly ``roadflow.verifier.graph_from_mask(mask)``, so
    replaying a query with ``Verifier(mask, queries=pool)`` on the same mask
    yields path score 1 for the self check.
    """
    from .verifier import make_queries

    requested = int(context.requested)
    queries: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    attempts = 0
    no_progress = 0

    if requested > 0:
        while len(queries) < requested and attempts < _MAX_QUERY_ATTEMPTS:
            need = requested - len(queries)
            batch = make_queries(
                graph,
                count=max(need, 1),
                seed=context.seed + attempts,
                spacing=context.control_spacing,
            )
            added = 0
            for u, v, distance in batch:
                key = (
                    (int(u[0]), int(u[1])),
                    (int(v[0]), int(v[1])),
                )
                if key in seen:
                    continue
                seen.add(key)
                queries.append((key[0], key[1], float(distance)))
                added += 1
                if len(queries) >= requested:
                    break
            attempts += 1
            if not batch:
                break
            if added == 0:
                no_progress += 1
                if no_progress >= 2:
                    break
            else:
                no_progress = 0

    shortfall = max(0, requested - len(queries)) if requested > 0 else 0
    if requested <= 0:
        shortfall_reason = "query_pool_disabled"
    elif len(queries) >= requested:
        shortfall_reason = None
    elif graph.number_of_nodes() == 0:
        shortfall_reason = "empty_mask"
    elif graph.number_of_edges() == 0:
        shortfall_reason = "isolated_skeleton_points_only"
    elif not queries:
        shortfall_reason = "no_reachable_verifier_control_pairs"
    else:
        shortfall_reason = "insufficient_unique_reachable_queries"

    query_meta: list[dict[str, Any]] = []
    lengths: list[float] = []
    visible_queries = 0
    measurable_queries = 0
    for u, v, distance in queries:
        visible, visibility = _path_visibility(graph, u, v, context.crop_size)
        if visible is not None:
            measurable_queries += 1
            if visible:
                visible_queries += 1
        u_geo = context.info.pixel_to_geo(
            float(context.x0 + u[1] + 0.5), float(context.y0 + u[0] + 0.5)
        )
        v_geo = context.info.pixel_to_geo(
            float(context.x0 + v[1] + 0.5), float(context.y0 + v[0] + 0.5)
        )
        query_meta.append(
            {
                "gt_distance_px": float(distance),
                "visible": visible,
                "visibility": visibility,
                "u_geo": [float(u_geo[0]), float(u_geo[1])],
                "v_geo": [float(v_geo[0]), float(v_geo[1])],
                "u_source_px": [float(context.x0 + u[1]), float(context.y0 + u[0])],
                "v_source_px": [float(context.x0 + v[1]), float(context.y0 + v[0])],
                "u_native": _nearest_native(
                    u, labels.polyline_points, labels.polyline_offsets
                ),
                "v_native": _nearest_native(
                    v, labels.polyline_points, labels.polyline_offsets
                ),
            }
        )
        lengths.append(float(distance))

    import networkx as nx

    node_count = int(graph.number_of_nodes())
    edge_count = int(graph.number_of_edges())
    components = int(nx.number_connected_components(graph)) if node_count else 0
    junction_pixels = int(sum(1 for _, degree in graph.degree if degree >= 3))
    cycles = max(0, edge_count - node_count + components) if node_count else 0
    if node_count == 0:
        category = "empty"
    elif components == 1 and junction_pixels == 0 and cycles == 0 and edge_count > 0:
        category = "single"
    else:
        category = "complex"

    length_stats = _length_stats(lengths)
    if length_stats is None:
        length_reason = shortfall_reason or "no_queries"
    else:
        length_reason = None

    if measurable_queries:
        visible_ratio: float | None = visible_queries / measurable_queries
        visible_reason: str | None = None
    else:
        visible_ratio = None
        visible_reason = shortfall_reason if not queries else "no_measurable_queries"

    applicability: dict[str, Any] = {
        "n_queries": len(queries),
        "query_pool_requested": requested,
        "query_pool_shortfall": shortfall,
        "query_pool_shortfall_reason": shortfall_reason,
        "query_length_px": length_stats,
        "query_length_px_reason": length_reason,
        "junction_pixels": junction_pixels,
        "connected_components": components,
        "cycles": cycles,
        "category": category,
        "complete_visible_path_ratio": visible_ratio,
        "complete_visible_path_ratio_reason": visible_reason,
        "visible_queries": visible_queries,
        "measurable_queries": measurable_queries,
        "centerline_pixels": int(np.asarray(labels.centerline).sum()),
        "mask_pixels": int(np.asarray(labels.mask).sum()),
    }

    cache: dict[str, Any] = {
        "format_version": _QUERY_FORMAT_VERSION,
        "crop_id": context.crop_id,
        "split": context.split,
        "source_id": context.source_id,
        "requested": requested,
        "pool_size": len(queries),
        "shortfall": shortfall,
        "shortfall_reason": shortfall_reason,
        "query_graph": "roadflow.verifier.graph_from_mask(mask)",
        "control_spacing": int(context.control_spacing),
        "visibility_rule": _VISIBILITY_RULE,
        "queries": [
            [[int(u[0]), int(u[1])], [int(v[0]), int(v[1])], float(distance)]
            for u, v, distance in queries
        ],
        "query_meta": query_meta,
        "query_meta_alignment": "parallel_to_queries",
        "native_polylines": {
            "points_key": "polyline_points",
            "offsets_key": "polyline_offsets",
            "coordinate_frame": "crop-local pixel (x, y)",
            "may_extend_outside_crop": True,
        },
        "centerline_key": "centerline",
        "npz": context.npz,
        "source_geojson": context.source_geojson,
        "crop": {
            "x0": int(context.x0),
            "y0": int(context.y0),
            "width": int(context.crop_size),
            "height": int(context.crop_size),
            "pixel_size_m": [float(context.pixel_size_m[0]), float(context.pixel_size_m[1])],
        },
        "pixel_center_offset": 0.5,
        "crs_epsg": context.info.crs_epsg,
        "transform": [float(value) for value in context.info.transform],
    }
    return cache, applicability


def _summarize_applicability(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    lengths_by_split: Mapping[str, Sequence[float]],
    requested: int,
) -> dict[str, Any]:
    """Aggregate per-crop applicability, keeping null + reason for no evidence."""
    summary: dict[str, Any] = {}
    for split in SPLITS:
        rows = list(rows_by_split.get(split, ()))
        lengths = list(lengths_by_split.get(split, ()))
        if not rows:
            summary[split] = {
                "crops": 0,
                "reason": "empty_split",
                "categories": {"empty": 0, "single": 0, "complex": 0},
                "category_fraction": None,
                "category_fraction_reason": "empty_split",
                "query_count": None,
                "query_count_reason": "empty_split",
                "query_length_px": None,
                "query_length_px_reason": "empty_split",
                "junction_pixels": None,
                "junction_pixels_reason": "empty_split",
                "connected_components": None,
                "connected_components_reason": "empty_split",
                "cycles": None,
                "cycles_reason": "empty_split",
                "complete_visible_path_ratio": None,
                "complete_visible_path_ratio_reason": "empty_split",
                "visible_queries": 0,
                "measurable_queries": 0,
                "query_pool_requested": requested,
                "query_pool_shortfall_crops": 0,
                "query_pool_shortfall_total": 0,
            }
            continue

        crops = len(rows)
        categories = {
            name: sum(1 for row in rows if row.get("category") == name)
            for name in ("empty", "single", "complex")
        }
        query_counts = [int(row.get("n_queries", 0)) for row in rows]
        visible = sum(int(row.get("visible_queries", 0)) for row in rows)
        measurable = sum(int(row.get("measurable_queries", 0)) for row in rows)
        length_stats = _length_stats(lengths)
        summary[split] = {
            "crops": crops,
            "reason": None,
            "categories": categories,
            "category_fraction": {
                name: count / crops for name, count in categories.items()
            },
            "category_fraction_reason": None,
            "query_count": {
                "total": int(sum(query_counts)),
                "min": int(min(query_counts)),
                "max": int(max(query_counts)),
                "mean": float(np.mean(query_counts)),
            },
            "query_count_reason": None,
            "query_length_px": length_stats,
            "query_length_px_reason": None if length_stats is not None else "no_queries",
            "junction_pixels": _metric_stats([row.get("junction_pixels", 0) for row in rows]),
            "junction_pixels_reason": None,
            "connected_components": _metric_stats(
                [row.get("connected_components", 0) for row in rows]
            ),
            "connected_components_reason": None,
            "cycles": _metric_stats([row.get("cycles", 0) for row in rows]),
            "cycles_reason": None,
            "complete_visible_path_ratio": (
                visible / measurable if measurable else None
            ),
            "complete_visible_path_ratio_reason": (
                None
                if measurable
                else ("no_queries" if not any(query_counts) else "no_measurable_queries")
            ),
            "visible_queries": visible,
            "measurable_queries": measurable,
            "query_pool_requested": requested,
            "query_pool_shortfall_crops": sum(
                1 for row in rows if int(row.get("query_pool_shortfall", 0)) > 0
            ),
            "query_pool_shortfall_total": int(
                sum(int(row.get("query_pool_shortfall", 0)) for row in rows)
            ),
        }
    return summary


def _write_json(path: Path, payload: Mapping[str, Any], indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload, indent=indent, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _crop_geo_bounds(info: GeoRaster, x0: int, y0: int, size: int) -> list[float]:
    corners = (
        info.pixel_to_geo(x0, y0),
        info.pixel_to_geo(x0 + size, y0),
        info.pixel_to_geo(x0, y0 + size),
        info.pixel_to_geo(x0 + size, y0 + size),
    )
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def prepare(
    root: str | Path,
    output: str | Path,
    crop_size: int = 256,
    train_limit: int | None = 400,
    dev_limit: int | None = 60,
    holdout_limit: int | None = 40,
    seed: int = 0,
    buffer_m: float = _DEFAULT_BUFFER_M,
    spatial_buffer_tiles: int = _DEFAULT_SPATIAL_BUFFER_TILES,
    query_pool_size: int = _DEFAULT_QUERY_POOL_SIZE,
    val_limit: int | None = None,
) -> dict[str, Any]:
    """Prepare fixed-resolution SpaceNet crops for train/dev/holdout.

    Source tiles are assigned to whole-tile geographic regions *before* any
    cropping.  ``dev`` and ``holdout`` are disjoint connected regions; a
    Chebyshev ring of ``spatial_buffer_tiles`` grid cells around both is
    excluded from training and from the other evaluation region.  The manifest
    records the excluded buffer sources and every requested/actual shortfall.

    Args:
        root: Dataset root containing ``train/.../PS-RGB`` and
            ``train/.../geojson_roads`` (or any directory of GeoTIFFs).
        output: Output directory for split subdirectories, ``queries/``,
            ``applicability.json`` and ``manifest.json``.
        crop_size: Native-resolution square crop size.
        train_limit: Maximum training crops; ``None`` means all available.
        dev_limit: Maximum development crops.
        holdout_limit: Maximum final holdout crops.
        seed: Seed for geographic region anchors, crop ordering and queries.
        buffer_m: Fixed physical centerline buffer width in metres.
        spatial_buffer_tiles: Chebyshev ring of source-grid cells excluded
            around dev and holdout.  ``0`` disables the spatial buffer.
        query_pool_size: Unique verifier-compatible queries cached per crop.
            The default 288 supports 24/48/96-query resampling experiments.
        val_limit: Legacy alias for ``dev_limit``.  Supplying it together with
            a non-default ``dev_limit`` is an error, and it never selects the
            holdout region.

    Returns:
        The manifest dictionary written to ``output/manifest.json``.
    """
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if buffer_m <= 0:
        raise ValueError("buffer_m must be positive")
    if train_limit is not None and train_limit < 0:
        raise ValueError("train_limit must be non-negative or None")
    if dev_limit is not None and dev_limit < 0:
        raise ValueError("dev_limit must be non-negative or None")
    if holdout_limit is not None and holdout_limit < 0:
        raise ValueError("holdout_limit must be non-negative or None")
    if spatial_buffer_tiles < 0:
        raise ValueError("spatial_buffer_tiles must be non-negative")
    if query_pool_size < 0:
        raise ValueError("query_pool_size must be non-negative")
    if val_limit is not None:
        if val_limit < 0:
            raise ValueError("val_limit must be non-negative or None")
        if dev_limit != 60:
            raise ValueError(
                "val_limit is a legacy alias for dev_limit; pass only one of "
                "dev_limit / val_limit"
            )
        # Legacy callers meant "validation" in the old two-split scheme, which
        # now plays the development role.  It is never mapped to holdout.
        dev_limit = val_limit

    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    from .verifier import VerifierConfig, graph_from_mask

    control_spacing = int(VerifierConfig().control_spacing)

    discovered = discover_sources(root)
    usable = [source for source in discovered if crop_specs(source.info, crop_size)]
    skipped = [source.source_id for source in discovered if source not in usable]
    if not usable:
        raise ValueError(
            f"No source tiles are at least {crop_size}x{crop_size} pixels under {root}"
        )

    plan = select_geographic_regions(
        usable,
        crop_size,
        dev_limit,
        holdout_limit,
        seed,
        spatial_buffer_tiles,
    )
    source_by_id = {source.source_id: source for source in usable}

    def sources_for(ids: Iterable[str]) -> list[SourceTile]:
        return [source_by_id[source_id] for source_id in ids if source_id in source_by_id]

    train_sources = sources_for(plan.train_source_ids)
    dev_sources = sources_for(plan.dev_source_ids)
    holdout_sources = sources_for(plan.holdout_source_ids)

    selections = {
        "train": select_crops(train_sources, crop_size, train_limit, seed, "train"),
        "dev": select_crops(dev_sources, crop_size, dev_limit, seed, "dev"),
        "holdout": select_crops(
            holdout_sources, crop_size, holdout_limit, seed, "holdout"
        ),
    }

    grouped: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        for source, (row, col) in selections[split]:
            entry = grouped.setdefault(
                source.source_id, {"source": source, "items": []}
            )
            entry["items"].append((split, row, col))

    split_of_source = {source_id: "train" for source_id in plan.train_source_ids}
    split_of_source.update({source_id: "dev" for source_id in plan.dev_source_ids})
    split_of_source.update(
        {source_id: "holdout" for source_id in plan.holdout_source_ids}
    )

    records: list[dict[str, Any]] = []
    sources_meta: dict[str, dict[str, Any]] = {}
    app_rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    lengths_by_split: dict[str, list[float]] = {split: [] for split in SPLITS}

    for source_id in sorted(grouped):
        entry = grouped[source_id]
        source: SourceTile = entry["source"]
        info = source.info
        image = read_geotiff_image(source.image_path)
        if image.shape[1:] != (info.height, info.width):
            raise ValueError(
                f"TIFF metadata/image mismatch for {source.image_path}: "
                f"{image.shape[1:]} vs {(info.height, info.width)}"
            )
        polylines = load_centerlines(source.geojson_path)
        if source.geojson_path is None:
            label_source = "missing_geojson_empty"
        elif polylines:
            label_source = "geojson_roads"
        else:
            label_source = "geojson_roads_empty"

        sources_meta[source_id] = {
            "image": _relative(source.image_path, root),
            "geojson": _relative(source.geojson_path, root),
            "split": split_of_source[source_id],
            "grid_cell": list(plan.cells[source_id]),
            "width": info.width,
            "height": info.height,
            "crs_epsg": info.crs_epsg,
            "is_geographic": info.is_geographic,
            "transform": list(info.transform),
            "bounds": list(info.geo_bounds()),
            "n_crops_total": len(crop_specs(info, crop_size)),
            "n_crops_selected": len(entry["items"]),
            "label_source": label_source,
        }

        for split, row, col in entry["items"]:
            # crop_specs returns source-pixel origins (row, col).
            y0, x0 = row, col
            grid_row, grid_col = y0 // crop_size, x0 // crop_size
            labels = rasterize_crop(info, polylines, x0, y0, crop_size, buffer_m)
            crop_image = np.ascontiguousarray(
                image[:, y0 : y0 + crop_size, x0 : x0 + crop_size]
            )
            crop_id = _safe_id(f"{source_id}__r{grid_row}_c{grid_col}")
            destination = output / split / f"{crop_id}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            row_m, col_m = info.pixel_size_m(
                info.pixel_to_geo(x0 + crop_size / 2.0, y0 + crop_size / 2.0)[1]
            )
            np.savez_compressed(
                destination,
                image=crop_image,
                mask=labels.mask,
                centerline=labels.centerline,
                polyline_points=labels.polyline_points,
                polyline_offsets=labels.polyline_offsets,
                crop_origin=np.asarray([x0, y0], dtype=np.int32),
                pixel_size_m=np.asarray([row_m, col_m], dtype=np.float32),
            )

            # The query graph is the verifier's mask skeleton, so a self check
            # with Verifier(mask, queries=pool) gives path score 1.
            graph, _skeleton = graph_from_mask(labels.mask)
            query_seed = int.from_bytes(
                _hash_key("queries", seed, crop_id)[:4], "big"
            ) % (2**31 - 1)
            context = CropQueryContext(
                crop_id=crop_id,
                split=split,
                source_id=source_id,
                npz=f"{split}/{crop_id}.npz",
                source_geojson=_relative(source.geojson_path, root),
                info=info,
                x0=x0,
                y0=y0,
                crop_size=crop_size,
                pixel_size_m=(float(row_m), float(col_m)),
                requested=query_pool_size,
                seed=query_seed,
                control_spacing=control_spacing,
            )
            cache, applicability = _build_query_cache(graph, labels, context)
            if query_pool_size > 0:
                query_cache_rel = f"queries/{crop_id}.json"
                _write_json(output / query_cache_rel, cache, indent=None)
            else:
                query_cache_rel = None
            app_rows_by_split[split].append(applicability)
            lengths_by_split[split].extend(
                float(query[2]) for query in cache["queries"]
            )

            mask_pixels = int(labels.mask.sum())
            records.append(
                {
                    "id": crop_id,
                    "npz": f"{split}/{crop_id}.npz",
                    "split": split,
                    "source_id": source_id,
                    "source_image": _relative(source.image_path, root),
                    "source_geojson": _relative(source.geojson_path, root),
                    "label_source": label_source,
                    "query_cache": query_cache_rel,
                    "query_pool_size": int(applicability["n_queries"]),
                    "query_pool_requested": int(applicability["query_pool_requested"]),
                    "query_pool_shortfall": int(applicability["query_pool_shortfall"]),
                    "query_pool_shortfall_reason": applicability[
                        "query_pool_shortfall_reason"
                    ],
                    "applicability": applicability,
                    "crop": {
                        "row": grid_row,
                        "col": grid_col,
                        "x0": x0,
                        "y0": y0,
                        "x1": x0 + crop_size,
                        "y1": y0 + crop_size,
                        "width": crop_size,
                        "height": crop_size,
                    },
                    "geo_bounds": _crop_geo_bounds(info, x0, y0, crop_size),
                    "pixel_size_m": [float(row_m), float(col_m)],
                    "buffer_m": float(buffer_m),
                    "crs_epsg": info.crs_epsg,
                    "n_polylines": int(len(labels.polyline_offsets) - 1),
                    "centerline_pixels": int(labels.centerline.sum()),
                    "has_native_centerline": bool(labels.centerline.any()),
                    "native_centerline_key": "centerline",
                    "polyline_points_key": "polyline_points",
                    "polyline_offsets_key": "polyline_offsets",
                    "mask_pixels": mask_pixels,
                    "empty": mask_pixels == 0,
                }
            )

    split_records: dict[str, list[dict[str, Any]]] = {
        split: sorted(
            (record for record in records if record["split"] == split),
            key=lambda record: record["id"],
        )
        for split in SPLITS
    }

    def counts(records_in_split: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "crops": len(records_in_split),
            "source_images": len(
                {record["source_id"] for record in records_in_split}
            ),
            "empty_crops": sum(bool(record["empty"]) for record in records_in_split),
            "query_pool_total": int(
                sum(int(record["query_pool_size"]) for record in records_in_split)
            ),
            "query_pool_shortfall_total": int(
                sum(int(record["query_pool_shortfall"]) for record in records_in_split)
            ),
            "crops_below_query_pool": sum(
                int(record["query_pool_shortfall"]) > 0
                for record in records_in_split
            ),
        }

    requested = {
        "train": train_limit,
        "dev": dev_limit,
        "holdout": holdout_limit,
    }
    shortfall_reason_by_split = {
        "train": "insufficient_train_crop_capacity_after_spatial_buffer",
        "dev": "insufficient_dev_region_crop_capacity",
        "holdout": "insufficient_holdout_region_crop_capacity",
    }
    shortfall: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        actual = len(split_records[split])
        limit = requested[split]
        if limit is None:
            gap = 0
            reason = None
        else:
            gap = max(0, int(limit) - actual)
            reason = shortfall_reason_by_split[split] if gap else None
        shortfall[split] = {
            "requested": limit,
            "actual": actual,
            "shortfall": gap,
            "reason": reason,
            "capacity_crops": int(plan.capacity[split]),
        }

    applicability_summary = {
        "format_version": _APPLICABILITY_FORMAT_VERSION,
        "query_pool_requested": int(query_pool_size),
        "query_graph": "roadflow.verifier.graph_from_mask(mask)",
        "visibility_rule": _VISIBILITY_RULE,
        "splits": _summarize_applicability(
            app_rows_by_split, lengths_by_split, int(query_pool_size)
        ),
    }
    applicability_crops = [
        {
            "id": record["id"],
            "split": record["split"],
            "source_id": record["source_id"],
            **record["applicability"],
        }
        for record in records
    ]
    _write_json(
        output / "applicability.json",
        {**applicability_summary, "crops": applicability_crops},
    )

    manifest: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "source_root": str(root),
        "crop_size": crop_size,
        "buffer_m": float(buffer_m),
        "spatial_buffer_tiles": int(spatial_buffer_tiles),
        "seed": int(seed),
        "splits": list(SPLITS),
        "legacy_val_limit": val_limit,
        "requested": requested,
        "shortfall": shortfall,
        "capacity": {split: int(plan.capacity[split]) for split in SPLITS},
        "excluded_sources": plan.excluded_sources,
        "image_normalization": {
            "method": "divide",
            "divisor": _IMAGE_DIVISOR,
            "applied_in": "RoadDataset.__getitem__ (npz image is raw DN)",
            "note": "Fixed 11-bit PS-RGB normalization; no per-crop percentile.",
        },
        "npz_schema": {
            "image": "float32 (3, crop_size, crop_size), raw sensor DN",
            "mask": "float32 (crop_size, crop_size), {0,1} buffered centerline",
            "centerline": "uint8 (crop_size, crop_size), native one-pixel centerline",
            "polyline_points": "float32 (P, 2), crop-local native polylines",
            "polyline_offsets": "int64 (E + 1,), offsets into polyline_points",
            "crop_origin": "int32 (2,), source-image (x0, y0)",
            "pixel_size_m": "float32 (2,), (row_m, column_m)",
        },
        "query_pool": {
            "requested_per_crop": int(query_pool_size),
            "cache_dir": "queries",
            "cache_schema": {
                "queries": "list of [[[u_row, u_col], [v_row, v_col], distance_px]]",
                "query_meta": "parallel list; visibility, geo endpoints, native refs",
                "native_polylines": "points_key/offsets_key into the crop npz",
            },
            "query_graph": "roadflow.verifier.graph_from_mask(mask)",
            "visibility_rule": _VISIBILITY_RULE,
            "note": (
                "Pools are generated from the GT mask skeleton with "
                "roadflow.verifier.make_queries and are verifier-compatible. "
                "At least query_pool_size unique queries are requested; any "
                "shortfall is explicit in the manifest and applicability report."
            ),
        },
        "split_strategy": {
            "name": "geographic_regions_with_spatial_buffer",
            "description": (
                "Whole source tiles first form disjoint seeded 4-neighbour "
                "grow-regions for dev and holdout on a grid derived from TIFF "
                "geographic bounds. A Chebyshev spatial buffer around both "
                "regions is excluded from train and from the other evaluation "
                "region. Cropping happens only after this source assignment; "
                "image numbers are not used."
            ),
            "dev_anchor_source": plan.dev_anchor,
            "holdout_anchor_source": plan.holdout_anchor,
            "dev_source_ids": plan.dev_source_ids,
            "holdout_source_ids": plan.holdout_source_ids,
            "dev_grid_cells": [list(cell) for cell in plan.dev_grid_cells],
            "holdout_grid_cells": [list(cell) for cell in plan.holdout_grid_cells],
            "buffer_source_ids": plan.buffer_source_ids,
            "spatial_buffer_tiles": int(spatial_buffer_tiles),
            "crop_grid": {
                "stride": crop_size,
                "overlap": False,
                "origin": "top-left source pixels",
            },
        },
        "counts": {
            "train": counts(split_records["train"]),
            "dev": counts(split_records["dev"]),
            "holdout": counts(split_records["holdout"]),
            "total_crops": len(records),
            "sources_discovered": len(discovered),
            "sources_usable": len(usable),
            "sources_skipped_too_small": skipped,
            "excluded_buffer_sources": len(plan.buffer_source_ids),
        },
        "applicability": applicability_summary,
        "sources": sources_meta,
        "train": split_records["train"],
        "dev": split_records["dev"],
        "holdout": split_records["holdout"],
    }

    # Keep the manifest atomic so an interrupted run cannot leave valid-looking
    # but truncated JSON.  New manifests deliberately have no legacy "val" key.
    manifest_path = output / "manifest.json"
    temporary = output / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RoadDataset:
    """Manifest-backed dataset returning image, mask, centerline and queries.

    Args:
        manifest: Path to ``manifest.json`` or an already-loaded manifest dict.
        split: ``"train"``, ``"dev"``, ``"holdout"`` or legacy ``"val"``.
            ``val`` resolves to a vintage ``val`` list when present, otherwise
            to ``dev`` on a new manifest.  It is never resolved to ``holdout``;
            likewise ``dev`` never falls back to ``holdout`` in either
            direction.  ``dev`` reads a vintage ``val`` list when that is the
            only available development split.
        root: Directory used to resolve relative paths.  Defaults to the
            manifest file's directory, or the current directory for a dict.

    Each item is a dict with:

    ``image``        float32 tensor, shape (3, H, W), raw DN divided by 2047.0.
    ``mask``         float32 tensor, shape (1, H, W), values in {0, 1}.
    ``centerline``   uint8 tensor, shape (1, H, W), native one-pixel raster.
    ``id``           str crop identifier from the manifest.
    ``source_id``    str source-tile identifier.
    ``query_pool``   Python list ``[(u, v, gt_distance_px), ...]`` generated
                     from the GT mask skeleton and directly consumable by
                     :class:`roadflow.verifier.Verifier` via ``queries=``.
                     Empty for vintage manifests without a query cache.
    ``query_cache``  absolute path to the per-crop query-cache JSON, or None.
    ``npz``          absolute path to the crop npz.
    ``stratum``      H1 stratification label mapped from the cached data
                     applicability ``category`` (``"empty"``, ``"single"``,
                     ``"complex"``).  Vintage manifests without applicability
                     metadata return ``"unclassified"``; no occlusion stratum
                     is invented.

    Variable-length ``query_pool`` and ``source_id`` are returned per item; the
    training loop is expected to stack ``image``/``mask`` itself.
    """

    def __init__(
        self,
        manifest: str | Path | Mapping[str, Any],
        split: str = "train",
        root: str | Path | None = None,
    ) -> None:
        if split not in _DATASET_SPLITS:
            raise ValueError(
                f"split must be one of {_DATASET_SPLITS}, got {split!r}"
            )
        if isinstance(manifest, Mapping):
            payload = dict(manifest)
            default_base = Path(".")
        else:
            manifest_path = Path(manifest).expanduser().resolve()
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            default_base = manifest_path.parent

        if split in payload:
            split_key = split
        elif split == "val" and "dev" in payload:
            split_key = "dev"
        elif split == "dev" and "val" in payload:
            split_key = "val"
        else:
            available = [name for name in _DATASET_SPLITS if name in payload]
            raise ValueError(
                f"manifest has no {split!r} split; available={available}. "
                "holdout is never used as dev or val."
            )

        self.manifest = payload
        self.split = split
        self.split_key = split_key
        self.records = list(payload.get(split_key) or [])
        self.root = (
            Path(root).expanduser().resolve() if root is not None else default_base
        )
        self._paths = [self.root / record["npz"] for record in self.records]
        self._query_pool_cache: dict[int, list[tuple[tuple[int, int], tuple[int, int], float]]] = {}

    def __len__(self) -> int:
        return len(self.records)

    @property
    def ids(self) -> list[str]:
        return [
            str(record.get("id", path.stem))
            for record, path in zip(self.records, self._paths)
        ]

    @property
    def source_ids(self) -> list[str]:
        return [str(record.get("source_id", "")) for record in self.records]

    def query_cache_path(self, index: int) -> Path | None:
        """Absolute path of the cached query pool, or None if not cached."""
        relative = self.records[index].get("query_cache")
        return self.root / relative if relative else None

    def query_pool(
        self, index: int
    ) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
        """Load and return the verifier-compatible query pool for one crop.

        Returns a new list of ``((u_row, u_col), (v_row, v_col), distance_px)``
        tuples.  Vintage manifests without ``query_cache`` return ``[]``.
        """
        if index in self._query_pool_cache:
            return list(self._query_pool_cache[index])
        path = self.query_cache_path(index)
        if path is None:
            pool: list[tuple[tuple[int, int], tuple[int, int], float]] = []
        else:
            if not path.is_file():
                raise FileNotFoundError(f"Missing query cache: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            pool = []
            for entry in payload.get("queries") or []:
                if len(entry) != 3:
                    raise ValueError(
                        f"Malformed query in {path}: expected [u, v, distance]"
                    )
                u = tuple(int(value) for value in entry[0])
                v = tuple(int(value) for value in entry[1])
                if len(u) != 2 or len(v) != 2:
                    raise ValueError(f"Malformed query endpoints in {path}: {entry}")
                pool.append((u, v, float(entry[2])))
        self._query_pool_cache[index] = pool
        return list(pool)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        path = self._paths[index]
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared crop: {path}")
        with np.load(path, allow_pickle=False) as payload:
            image = np.asarray(payload["image"], dtype=np.float32)
            mask = np.asarray(payload["mask"], dtype=np.float32)
            if "centerline" not in payload:
                raise ValueError(
                    f"Expected native centerline raster in {path}; "
                    "re-prepare this crop"
                )
            centerline = np.asarray(payload["centerline"], dtype=np.uint8)
        if image.ndim != 3:
            raise ValueError(f"Expected CHW image in {path}, got {image.shape}")
        image = image / _IMAGE_DIVISOR
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"Expected HW mask in {path}, got {mask.shape}")
        if centerline.ndim == 3 and centerline.shape[0] == 1:
            centerline = centerline[0]
        if centerline.ndim != 2:
            raise ValueError(
                f"Expected HW native centerline in {path}, got {centerline.shape}"
            )
        crop_id = str(self.records[index].get("id", path.stem))
        cache_path = self.query_cache_path(index)
        applicability = self.records[index].get("applicability") or {}
        stratum = applicability.get("category") or "unclassified"
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "centerline": torch.from_numpy(
                np.ascontiguousarray(centerline)
            ).unsqueeze(0),
            "id": crop_id,
            "source_id": str(self.records[index].get("source_id", "")),
            "query_pool": self.query_pool(index),
            "query_cache": str(cache_path) if cache_path is not None else None,
            "npz": str(path),
            "stratum": str(stratum),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare fixed-resolution SpaceNet road crops and manifest."
    )
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=400)
    parser.add_argument("--dev-limit", type=int, default=60)
    parser.add_argument("--holdout-limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-m", type=float, default=_DEFAULT_BUFFER_M)
    parser.add_argument(
        "--spatial-buffer-tiles",
        type=int,
        default=_DEFAULT_SPATIAL_BUFFER_TILES,
        help="Chebyshev tile ring excluded around dev and holdout (0 disables)",
    )
    parser.add_argument(
        "--query-pool-size",
        type=int,
        default=_DEFAULT_QUERY_POOL_SIZE,
        help="Unique verifier-compatible queries cached per crop",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Legacy alias for --dev-limit; never selects holdout",
    )
    args = parser.parse_args(argv)

    manifest = prepare(
        root=args.root,
        output=args.output,
        crop_size=args.crop_size,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        holdout_limit=args.holdout_limit,
        seed=args.seed,
        buffer_m=args.buffer_m,
        spatial_buffer_tiles=args.spatial_buffer_tiles,
        query_pool_size=args.query_pool_size,
        val_limit=args.val_limit,
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(json.dumps(manifest["shortfall"], indent=2))
    print(f"manifest: {Path(args.output).expanduser().resolve() / 'manifest.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
