"""Fixed, non-repairing skeleton verifier; scores are NOT official APLS.

The fixed Stage 0 pipeline is::

    threshold -> skeletonize -> pixel graph -> junction aggregation ->
    degree-2 compression -> polyline-preserving weighted multigraph -> path score

No morphological closing, gap bridging, or A* repair is performed.  The pixel
graph is retained for compatibility (``graph_from_mask``) and as a precise
pixel scaffold.  The compressed graph used for shortest paths stores every
skeleton pixel along each polyline edge, so a projected query point is located
by its exact arc-length offset; projection is never collapsed to the nearest
compressed endpoint.  Distances are pixels at the original crop resolution.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import itertools
import math
import time

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.draw import line as _raster_line
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class VerifierConfig:
    threshold: float = 0.5
    match_radius: float = 4.0
    boundary_tolerance: float = 2.0
    centerline_tolerance: float = 2.0
    query_count: int = 48
    control_spacing: int = 8
    length_error_tolerance: float = 0.1
    weights: tuple = (0.5, 0.2, 0.2, 0.1)
    # Short terminal branches are a structural-noise proxy, not a true spur rate.
    short_terminal_branch_max_length: float = 8.0

    def __post_init__(self):
        if len(self.weights) != 4 or min(self.weights) < 0 or not np.isclose(sum(self.weights), 1):
            raise ValueError('Four nonnegative reward weights must sum to one')
        if self.query_count < 1 or self.control_spacing < 1:
            raise ValueError('Query count and control spacing must be positive')
        if min(self.match_radius, self.boundary_tolerance, self.centerline_tolerance) < 0:
            raise ValueError('Tolerances must be nonnegative')
        if self.short_terminal_branch_max_length < 0:
            raise ValueError('Terminal branch length must be nonnegative')


def _pixel_graph_from_skeleton(skeleton):
    """Build the legacy 8-connected pixel graph without re-skeletonizing."""
    graph = nx.Graph()
    points = [tuple(map(int, p)) for p in np.argwhere(skeleton)]
    graph.add_nodes_from(points)
    for y, x in points:
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            q = (y + dy, x + dx)
            if q not in graph:
                continue
            # Avoid diagonal triangles at orthogonal junctions, retaining diagonals.
            if dy and dx and ((y, x + dx) in graph or (y + dy, x) in graph):
                continue
            graph.add_edge((y, x), q, weight=math.hypot(dy, dx))
    return graph


def graph_from_mask(mask):
    """Return the legacy pixel graph and skeleton (kept for data/tests).

    Every skeleton pixel is a node and every 8-connected skeleton edge is an
    edge.  Loops and isolated pixels are preserved verbatim.  Nothing is added,
    closed, bridged or repaired.
    """
    skeleton = skeletonize(np.asarray(mask, dtype=bool))
    return _pixel_graph_from_skeleton(skeleton), skeleton


def _undirected_pixel_key(a, b):
    return (a, b) if a <= b else (b, a)


def _polyline_offsets(points):
    offsets = [0.0]
    for (y0, x0), (y1, x1) in zip(points, points[1:]):
        offsets.append(offsets[-1] + math.hypot(y1 - y0, x1 - x0))
    return offsets


def _add_polyline_edge(graph, u, v, path):
    offsets = _polyline_offsets(path)
    weight = offsets[-1]
    if weight <= 0:
        return None
    key = graph.add_edge(
        u, v, weight=float(weight),
        polyline=tuple((int(y), int(x)) for y, x in path),
        offsets=tuple(float(value) for value in offsets),
        start=tuple(map(int, path[0])), end=tuple(map(int, path[-1])))
    return key


def _compress_pixel_graph(pixel_graph, skeleton):
    """Compress degree-2 chains while preserving exact pixel-graph distances.

    Adjacent junction pixels are identified as one logical cluster
    (``graph.graph['junction_clusters']``), but every junction *port* pixel
    remains its own graph node.  This keeps the internal distance between two
    ports exact; collapsing a cluster into one node would silently change path
    lengths and could conflict with the self-score-equals-one requirement.
    The returned graph is a ``networkx.MultiGraph`` so parallel corridors and
    closed loops are never collapsed.  Node labels are integer ``(row, col)``
    skeleton pixels, which keeps query caches JSON serializable.
    """
    compressed = nx.MultiGraph()
    degrees = dict(pixel_graph.degree())
    junction_pixels = {p for p, d in degrees.items() if d >= 3}
    endpoint_pixels = {p for p, d in degrees.items() if d == 1}
    isolated_pixels = {p for p, d in degrees.items() if d == 0}

    # A pure cycle has no junction or endpoint.  Keep a few anchor pixels so the
    # loop survives compression and can still provide reachable query pairs.
    anchors = set()
    for component in nx.connected_components(pixel_graph):
        nodes = sorted(component)
        if len(nodes) >= 3 and all(degrees[p] == 2 for p in nodes):
            anchor_count = min(4, max(2, len(nodes) // 4))
            indices = np.linspace(0, len(nodes) - 1, num=anchor_count, dtype=int)
            anchors.update(nodes[int(i)] for i in indices)

    # Logical junction clusters are retained as metadata only.  Port pixels stay
    # separate nodes so internal junction distances remain exact.
    cluster_label = {}
    junction_clusters = {}
    visited_junctions = set()
    for pixel in sorted(junction_pixels):
        if pixel in visited_junctions:
            continue
        cluster = []
        stack = [pixel]
        while stack:
            current = stack.pop()
            if current in visited_junctions:
                continue
            visited_junctions.add(current)
            cluster.append(current)
            for neighbour in pixel_graph.neighbors(current):
                if neighbour in junction_pixels and neighbour not in visited_junctions:
                    stack.append(neighbour)
        representative = min(cluster)
        junction_clusters[representative] = tuple(sorted(cluster))
        for member in cluster:
            cluster_label[member] = representative

    pixel_to_node = {}
    node_attributes = {}
    for pixel in sorted(junction_pixels):
        pixel_to_node[pixel] = pixel
        node_attributes[pixel] = ('junction', (pixel,), cluster_label[pixel])
    for pixel in sorted(endpoint_pixels | isolated_pixels | anchors):
        if pixel in pixel_to_node:
            continue
        pixel_to_node[pixel] = pixel
        kind = ('endpoint' if pixel in endpoint_pixels else
                'loop_anchor' if pixel in anchors else 'isolated')
        node_attributes[pixel] = (kind, (pixel,), None)

    for node in sorted(node_attributes):
        kind, pixels, cluster = node_attributes[node]
        compressed.add_node(node, pixel=node, kind=kind, pixels=pixels,
                            cluster=cluster)

    special = set(pixel_to_node)
    visited_edges = set()
    for start in sorted(special):
        for first in sorted(pixel_graph.neighbors(start)):
            edge_key = _undirected_pixel_key(start, first)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            path = [start, first]
            if first in special:
                end = first
            else:
                previous, current = start, first
                end = None
                while True:
                    neighbours = [n for n in pixel_graph.neighbors(current)
                                  if n != previous]
                    if len(neighbours) != 1:
                        break
                    nxt = neighbours[0]
                    next_key = _undirected_pixel_key(current, nxt)
                    if next_key in visited_edges:
                        break
                    visited_edges.add(next_key)
                    path.append(nxt)
                    previous, current = current, nxt
                    if current in special:
                        end = current
                        break
                if end is None:
                    continue
            # Direct junction-port edges keep their exact pixel weight, so no
            # internal cluster distance is lost by the logical aggregation.
            _add_polyline_edge(compressed, pixel_to_node[start],
                               pixel_to_node[end], path)

    junction_map = {}
    for node in compressed.nodes:
        kind, _, _ = node_attributes[node]
        if kind == 'junction':
            junction_map[node] = (node,)
    compressed.graph.update(dict(
        skeleton=skeleton,
        pixel_graph=pixel_graph,
        node_pixels={node: node for node in compressed.nodes},
        junction_pixels=junction_map,
        junction_clusters=junction_clusters,
        source_shape=tuple(int(v) for v in skeleton.shape),
        pipeline='threshold->skeleton->pixel_graph->junction_aggregation->degree2_compression',
        repair=False))
    return compressed


def vectorize(mask, threshold=0.5):
    """Threshold and compress a mask into a polyline-preserving graph.

    ``mask`` may be boolean or numeric.  Numeric input is thresholded (a
    ``VerifierConfig`` may be passed as ``threshold`` for convenience).  Each
    graph edge carries ``weight`` (arc length), ``polyline`` (all skeleton
    pixels from the start port to the end port) and cumulative ``offsets`` from
    the start port.  Adjacent junction pixels are recorded as one logical
    cluster in ``graph.graph['junction_clusters']`` while their port nodes are
    retained, so compressed-graph distances exactly match the pixel scaffold.
    The pixel scaffold and skeleton are stored in ``graph.graph``.  Loops,
    parallel corridors and isolated pixels are kept; no repair of any kind is
    performed.
    """
    if isinstance(threshold, VerifierConfig):
        threshold = threshold.threshold
    array = np.asarray(mask)
    binary = array.astype(bool) if array.dtype == bool else array >= threshold
    skeleton = skeletonize(binary)
    pixel_graph = _pixel_graph_from_skeleton(skeleton)
    return _compress_pixel_graph(pixel_graph, skeleton)


def subdivide_graph(graph, spacing=4):
    """Insert degree-2 nodes along polyline edges without changing geometry.

    This is the node-splitting invariance helper: shortest-path lengths and
    projected point distances are unchanged by subdivision.  Used by tests and
    useful when an external consumer needs regularly sampled graph nodes.
    """
    if spacing <= 0:
        raise ValueError('Subdivision spacing must be positive')
    result = nx.MultiGraph()
    result.graph.update(graph.graph)
    result.graph['subdivided_spacing'] = float(spacing)
    node_pixels = dict(graph.graph.get('node_pixels', {}))
    for node, data in graph.nodes(data=True):
        result.add_node(node, **data)
    for u, v, key, data in graph.edges(keys=True, data=True):
        polyline = [tuple(map(int, p)) for p in data['polyline']]
        offsets = [float(x) for x in data['offsets']]
        breaks = [0]
        last = 0.0
        for index in range(1, len(polyline) - 1):
            if offsets[index] - last >= spacing:
                breaks.append(index)
                last = offsets[index]
        breaks.append(len(polyline) - 1)
        for position, (i0, i1) in enumerate(zip(breaks, breaks[1:])):
            start_node = u if i0 == 0 else tuple(polyline[i0])
            end_node = v if i1 == len(polyline) - 1 else tuple(polyline[i1])
            if start_node not in result:
                result.add_node(start_node, pixel=start_node, kind='split',
                                pixels=(start_node,))
            if end_node not in result:
                result.add_node(end_node, pixel=end_node, kind='split',
                                pixels=(end_node,))
            segment = polyline[i0:i1 + 1]
            segment_offsets = [offsets[i] - offsets[i0] for i in range(i0, i1 + 1)]
            weight = segment_offsets[-1]
            if weight <= 0:
                continue
            result.add_edge(
                start_node, end_node, weight=float(weight),
                polyline=tuple(segment),
                offsets=tuple(float(x) for x in segment_offsets),
                start=tuple(segment[0]), end=tuple(segment[-1]))
    result.graph['node_pixels'] = {
        node: node for node in result.nodes}
    return result


@dataclass(frozen=True)
class _Location:
    kind: str
    pixel: tuple
    distance: float = 0.0
    node: tuple = None
    edge: tuple = None
    edge_u: tuple = None
    edge_v: tuple = None
    edge_length: float = 0.0
    offset: float = 0.0
    self_loop: bool = False
    point: tuple = None


class _ProjectionIndex:
    """Exact projection of a pixel onto compressed polyline edges.

    The nearest edge segment is found with a KD-tree over polyline vertices
    (segments are unit-length skeleton steps), then the point is projected onto
    that segment to recover the arc-length offset used by shortest paths.
    """

    def __init__(self, graph, search_vertices=8):
        self.graph = graph
        self.search_vertices = int(search_vertices)
        self.node_map = {}
        for node, data in graph.nodes(data=True):
            pixel = tuple(map(int, data.get('pixel', node)))
            if pixel not in self.node_map:
                self.node_map[pixel] = node
        self.segments = []
        vertex_segments = {}
        for u, v, key, data in graph.edges(keys=True, data=True):
            polyline = np.asarray(data['polyline'], dtype=float)
            offsets = np.asarray(data['offsets'], dtype=float)
            length = float(data['weight'])
            for i in range(len(polyline) - 1):
                a = polyline[i]
                b = polyline[i + 1]
                segment_length = float(np.linalg.norm(b - a))
                if segment_length <= 0:
                    continue
                index = len(self.segments)
                self.segments.append(dict(
                    a=a, b=b, offset=float(offsets[i]),
                    edge=(u, v, key), u=u, v=v, length=length,
                    self_loop=(u == v), segment_length=segment_length))
                for point in (tuple(a), tuple(b)):
                    vertex_segments.setdefault(point, set()).add(index)
        self._vertex_segments = vertex_segments
        self._vertices = list(vertex_segments)
        self._tree = (cKDTree(np.asarray(self._vertices, dtype=float))
                      if self._vertices else None)

    def locate(self, pixel):
        point = np.asarray(pixel, dtype=float).reshape(2)
        rounded = tuple(int(round(float(value))) for value in point)
        if (tuple(point) == tuple(float(value) for value in rounded)
                and rounded in self.node_map):
            return _Location(kind='node', pixel=rounded, distance=0.0,
                             node=self.node_map[rounded])
        if self._tree is None:
            return None
        count = min(self.search_vertices, len(self._vertices))
        _, indices = self._tree.query(point, k=count)
        indices = np.atleast_1d(indices)
        candidates = set()
        for index in indices:
            candidates.update(self._vertex_segments[self._vertices[int(index)]])
        best = None
        best_location = None
        for segment_index in sorted(candidates):
            segment = self.segments[segment_index]
            a, b = segment['a'], segment['b']
            ab = b - a
            denominator = float(np.dot(ab, ab))
            if denominator <= 0:
                fraction = 0.0
            else:
                fraction = float(np.clip(np.dot(point - a, ab) / denominator, 0.0, 1.0))
            projection = a + fraction * ab
            distance = float(np.linalg.norm(point - projection))
            if (best is None or distance < best - 1e-12
                    or (abs(distance - best) <= 1e-12
                        and best_location is not None
                        and segment_index < best_location[0])):
                best = distance
                best_location = (segment_index, fraction, projection, segment)
        if best_location is None:
            return None
        segment_index, fraction, projection, segment = best_location
        offset = segment['offset'] + fraction * segment['segment_length']
        return _Location(
            kind='edge', pixel=pixel, distance=best,
            edge=segment['edge'], edge_u=segment['u'], edge_v=segment['v'],
            edge_length=segment['length'], offset=float(offset),
            self_loop=segment['self_loop'],
            point=(float(projection[0]), float(projection[1])))


def _node_distances(graph, node, cache):
    if node not in cache:
        cache[node] = nx.single_source_dijkstra_path_length(
            graph, node, weight='weight')
    return cache[node]


def _location_to_node_distance(graph, location, target, cache):
    if location.kind == 'node':
        return float(_node_distances(graph, location.node, cache).get(target, math.inf))
    source_distances = _node_distances(graph, location.edge_u, cache)
    if location.self_loop:
        return (float(source_distances.get(target, math.inf))
                + min(location.offset, location.edge_length - location.offset))
    other_distances = _node_distances(graph, location.edge_v, cache)
    return min(
        float(source_distances.get(target, math.inf)) + location.offset,
        float(other_distances.get(target, math.inf))
        + (location.edge_length - location.offset))


def _location_distance(graph, first, second, cache):
    """Shortest-path length between two projected locations (or None)."""
    if first.kind == 'node' and second.kind == 'node':
        value = _node_distances(graph, first.node, cache).get(second.node, math.inf)
        return float(value) if np.isfinite(value) else None
    if first.kind == 'node':
        value = _location_to_node_distance(graph, second, first.node, cache)
        return value if np.isfinite(value) else None
    if second.kind == 'node':
        value = _location_to_node_distance(graph, first, second.node, cache)
        return value if np.isfinite(value) else None

    candidates = []
    if first.edge == second.edge:
        direct = abs(first.offset - second.offset)
        if first.self_loop:
            direct = min(direct, first.edge_length - direct)
        candidates.append(direct)
    to_u = _location_to_node_distance(graph, first, second.edge_u, cache)
    if second.self_loop:
        candidates.append(to_u + min(second.offset,
                                     second.edge_length - second.offset))
    else:
        candidates.append(to_u + second.offset)
        to_v = _location_to_node_distance(graph, first, second.edge_v, cache)
        candidates.append(to_v + (second.edge_length - second.offset))
    finite = [value for value in candidates if np.isfinite(value)]
    return float(min(finite)) if finite else None


def pixel_distance(graph, u, v):
    """Exact graph distance between two skeleton pixel coordinates.

    Query points may be compressed nodes or arbitrary pixels on a polyline.
    Projection retains the exact arc-length offset instead of snapping to the
    nearest compressed endpoint.
    """
    index = _ProjectionIndex(graph)
    first = index.locate(u)
    second = index.locate(v)
    if first is None or second is None:
        return None
    return _location_distance(graph, first, second, {})


def short_terminal_branch_density(graph, max_length=8.0):
    """Count short terminal branches attached to junctions.

    ``density`` is the count per 100 pixels of graph length.  This is a
    structural proxy for terminal noise; it is explicitly not a measured or
    ground-truth spur rate.  Compare a candidate value with the GT reference
    (``Verifier`` reports both) rather than treating the candidate value alone
    as an error count.
    """
    total_length = 0.0
    for _, _, _, data in graph.edges(keys=True, data=True):
        total_length += float(data.get('weight', 0.0))
    count = 0
    for node in sorted(n for n, degree in graph.degree if degree == 1):
        incident = sorted(graph.edges(node, keys=True, data=True))
        if len(incident) != 1:
            continue
        u, v, _, data = incident[0]
        previous = node
        current = v if u == node else u
        length = float(data.get('weight', 0.0))
        while graph.degree(current) == 2 and length <= max_length:
            options = []
            for a, b, key, edge_data in sorted(graph.edges(current, keys=True, data=True)):
                other = b if a == current else a
                if other != previous:
                    options.append((other, edge_data))
            if len(options) != 1:
                break
            other, edge_data = options[0]
            previous, current = current, other
            length += float(edge_data.get('weight', 0.0))
        if graph.degree(current) >= 3 and length <= max_length:
            count += 1
    density = float(count / total_length * 100.0) if total_length > 0 else 0.0
    return dict(count=int(count), density=density,
                total_length=float(total_length), max_length=float(max_length))


def make_queries(graph, count=48, seed=0, spacing=8):
    """Sample reachable pairs balanced across per-crop length tertiles.

    Length strata are per-crop for this quick-test implementation, rather than
    dataset-wide quantiles.  No full-scene shortest path is used after cropping.
    The API is shared by the pixel graph and the compressed vector graph; node
    labels are integer ``(row, col)`` pairs so cached queries remain JSON safe.
    """
    rng = np.random.default_rng(seed)
    nodes = sorted(graph.nodes)
    controls = set(n for n, degree in graph.degree if degree != 2)
    controls.update(nodes[::spacing])
    # Ensure at least two controls per connected component, so pure loops and
    # multi-component masks still yield reachable query pairs.
    for component in nx.connected_components(graph):
        component_nodes = sorted(component)
        if len(component_nodes) < 2:
            continue
        present = [node for node in component_nodes if node in controls]
        if len(present) < 2:
            for index in (0, len(component_nodes) // 2, len(component_nodes) - 1):
                controls.add(component_nodes[index])
    controls = sorted(controls)
    if len(controls) < 2:
        return []
    # Bound source Dijkstra calls even for dense masks.
    sources = rng.choice(len(controls), min(len(controls), max(12, count)), replace=False)
    pool = {}
    for index in sources:
        u = controls[int(index)]
        distances = nx.single_source_dijkstra_path_length(graph, u, weight='weight')
        targets = rng.choice(len(controls), min(len(controls), max(12, count)), replace=False)
        for j in targets:
            v = controls[int(j)]
            if v != u and v in distances and distances[v] > 0:
                pair = tuple(sorted((u, v)))
                pool[pair] = (pair[0], pair[1], float(distances[v]))
    ordered = sorted(pool.values(), key=lambda query: query[2])
    if not ordered:
        return []
    bins = np.array_split(np.arange(len(ordered)), 3)
    queries = []
    for i in range(count):
        bucket = bins[i % 3]
        if len(bucket):
            queries.append(ordered[int(rng.choice(bucket))])
    return queries


def dice_score(pred, gt):
    denominator = pred.sum() + gt.sum()
    return float(2 * np.logical_and(pred, gt).sum() / denominator) if denominator else 1.0


def _near_fraction(source, target, tolerance):
    if not source.any():
        return 1.0 if not target.any() else 0.0
    if not target.any():
        return 0.0
    return float((ndi.distance_transform_edt(~target)[source] <= tolerance).mean())


def boundary_f1(pred, gt, tolerance=2):
    p = pred ^ ndi.binary_erosion(pred, border_value=0)
    g = gt ^ ndi.binary_erosion(gt, border_value=0)
    precision = _near_fraction(p, g, tolerance)
    recall = _near_fraction(g, p, tolerance)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class Verifier:
    def __init__(self, gt, config=None, seed=0, queries=None, centerline=None):
        self.config = config or VerifierConfig()
        self.gt = np.asarray(gt, dtype=bool)
        source = self.gt if centerline is None else np.asarray(centerline, dtype=bool)
        self.skeleton = skeletonize(source)
        self.pixel_graph = _pixel_graph_from_skeleton(self.skeleton)
        # Legacy attribute: ``graph_from_mask`` always returned the pixel graph.
        self.graph = self.pixel_graph
        self.vectorized = _compress_pixel_graph(self.pixel_graph, self.skeleton)
        self._projection = _ProjectionIndex(self.vectorized)
        self.gt_short_terminal_branch = short_terminal_branch_density(
            self.vectorized, self.config.short_terminal_branch_max_length)
        if queries is None:
            raw_queries = make_queries(self.pixel_graph, self.config.query_count,
                                       seed, self.config.control_spacing)
            distance_cache = {}
            self.queries = []
            for u, v, _ in raw_queries:
                u = tuple(int(x) for x in u)
                v = tuple(int(x) for x in v)
                first = self._projection.locate(u)
                second = self._projection.locate(v)
                if first is None or second is None:
                    continue
                length = _location_distance(self.vectorized, first, second,
                                            distance_cache)
                # Keep the pixel-derived control points but store the exact
                # compressed-graph distance used later by score().
                if length is not None and length > 0:
                    self.queries.append((u, v, float(length)))
        else:
            self.queries = []
            for query in queries:
                u, v, distance = query
                self.queries.append((tuple(int(x) for x in u),
                                     tuple(int(x) for x in v), float(distance)))

    def cache(self):
        return {'config': asdict(self.config), 'queries': self.queries}

    def score(self, candidate):
        cfg = self.config
        pred = np.asarray(candidate) >= cfg.threshold
        if pred.shape != self.gt.shape:
            raise ValueError('Prediction and GT must have identical shapes')
        # Public vectorization path: threshold is already applied above, then
        # skeleton -> junction aggregation -> degree-2 compression.
        graph = vectorize(pred, threshold=cfg.threshold)
        skeleton = graph.graph['skeleton']
        pixel_graph = graph.graph['pixel_graph']
        projection = _ProjectionIndex(graph)
        counts = dict(match_failure=0, disconnected=0, detour=0, shortened=0, consistent=0)
        values = []
        distance_cache = {}
        for u, v, gt_distance in self.queries:
            if gt_distance <= 0:
                raise ValueError('Query distance must be positive')
            first = projection.locate(u)
            second = projection.locate(v)
            if (first is None or second is None
                    or max(first.distance, second.distance) > cfg.match_radius):
                counts['match_failure'] += 1
                values.append(0.0)
                continue
            length = _location_distance(graph, first, second, distance_cache)
            if length is None:
                counts['disconnected'] += 1
                values.append(0.0)
                continue
            error = (length - gt_distance) / gt_distance
            key = 'detour' if error > cfg.length_error_tolerance else (
                'shortened' if error < -cfg.length_error_tolerance else 'consistent')
            counts[key] += 1
            values.append(1.0 - min(1.0, abs(error)))
        # A nonempty graph without reachable control pairs has no path evidence.
        path = float(np.mean(values)) if values else 0.0
        precision = _near_fraction(skeleton, self.skeleton, cfg.centerline_tolerance)
        dice = dice_score(pred, self.gt)
        bf1 = boundary_f1(pred, self.gt, cfg.boundary_tolerance)
        components = (path, precision, dice, bf1)
        reward = sum(w * s for w, s in zip(cfg.weights, components))
        if not self.gt.any():
            reward = 1.0 - float(pred.mean())
            path, precision = 0.0, 0.0
        labeled, _ = ndi.label(pred)
        sizes = np.bincount(labeled.ravel())[1:]
        small_fraction = float(sizes[sizes < 16].sum() / max(1, pred.sum()))
        short_endpoints = sum(degree == 1 for _, degree in pixel_graph.degree)
        terminal = short_terminal_branch_density(
            graph, cfg.short_terminal_branch_max_length)
        gt_terminal = self.gt_short_terminal_branch
        return dict(reward=float(reward), path=path, centerline_precision=precision,
                    dice=dice, bf1=bf1, queries=len(values), errors=counts,
                    endpoint_density=short_endpoints / max(1, skeleton.sum()),
                    small_component_fraction=small_fraction,
                    short_terminal_branch_count=terminal['count'],
                    short_terminal_branch_density=terminal['density'],
                    gt_short_terminal_branch_count=gt_terminal['count'],
                    gt_short_terminal_branch_density=gt_terminal['density'],
                    short_terminal_branch_density_excess=float(
                        terminal['density'] - gt_terminal['density']),
                    # Short aliases used by the evaluation/protocol report.
                    short_branch_count=terminal['count'],
                    short_branch_density=terminal['density'],
                    gt_short_branch_density=gt_terminal['density'])


def sample_query_pool(pool, count=48, seed=0):
    """Deterministically sample training queries from a cached query pool.

    ``pool`` is a sequence of ``(u, v, distance)`` entries or a mapping with a
    ``queries`` key (the JSON cache written by :meth:`Verifier.cache`).  Sampling
    is without replacement when the pool is large enough, otherwise with
    replacement.  Returned entries remain JSON serializable.
    """
    if count < 1:
        raise ValueError('Query count must be positive')
    if isinstance(pool, Mapping):
        if 'queries' not in pool:
            raise TypeError("Query pool mapping must contain a 'queries' key")
        pool = pool['queries']
    entries = list(pool)
    if not entries:
        return []
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(entries), size=count,
                         replace=count > len(entries))
    sampled = []
    for index in np.atleast_1d(indices):
        entry = entries[int(index)]
        if isinstance(entry, np.ndarray):
            entry = entry.tolist()
        if isinstance(entry, (list, tuple)):
            entry = tuple(value.tolist() if isinstance(value, np.ndarray)
                          else value.item() if isinstance(value, np.generic)
                          else value for value in entry)
        sampled.append(entry)
    return sampled


def _pairwise_rank_flip(reference, sample):
    flips = 0
    total = 0
    for i, j in itertools.combinations(range(len(sample)), 2):
        ref = reference[i] - reference[j]
        if abs(ref) <= 1e-15:
            continue
        total += 1
        value = sample[i] - sample[j]
        if abs(value) <= 1e-15 or (ref > 0) != (value > 0):
            flips += 1
    return flips / total if total else 0.0


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def query_budget_report(gt, candidates, counts=(24, 48, 96), seeds=(0, 1),
                        config=None, apls=None):
    """Re-sample query sets at several budgets and summarize ranking stability.

    For every query count and seed the same candidates are scored with a fresh
    query draw.  The report includes reward gaps, top-candidate flips, pairwise
    rank flips within the count, rank flips against a reference count, timings,
    and (only when an official APLS callable is supplied) APLS diagnostics.
    The callable uses the existing official adapter signature
    ``apls(gt_mask, predicted_mask, metadata) -> float``.  Without it every
    APLS field is JSON null; no local approximation is substituted.
    """
    candidates = [np.asarray(candidate) for candidate in candidates]
    if not candidates:
        raise ValueError('At least one candidate is required')
    counts = tuple(int(count) for count in counts)
    seeds = tuple(int(seed) for seed in seeds)
    if not counts or not seeds:
        raise ValueError('At least one count and one seed are required')
    base_config = config or VerifierConfig()
    apls_values = None
    if apls is not None:
        apls_values = []
        for index, candidate in enumerate(candidates):
            value = apls(np.asarray(gt, dtype=bool), candidate,
                         {'candidate': index})
            apls_values.append(None if value is None else float(value))

    reports = {}
    mean_rewards = {}
    for count in counts:
        seed_rewards = []
        top_per_seed = []
        gaps = []
        build_times = []
        score_times = []
        query_counts = []
        for seed in seeds:
            start = time.perf_counter()
            verifier = Verifier(gt, config=replace(base_config, query_count=count),
                                seed=seed)
            build_times.append(time.perf_counter() - start)
            scores = []
            start = time.perf_counter()
            for candidate in candidates:
                scores.append(float(verifier.score(candidate)['reward']))
            score_times.append((time.perf_counter() - start) / len(candidates))
            seed_rewards.append(scores)
            query_counts.append(len(verifier.queries))
            if len(scores) >= 2:
                ordered = sorted(scores, reverse=True)
                gaps.append(ordered[0] - ordered[1])
            top_per_seed.append(max(range(len(scores)),
                                    key=lambda index: (scores[index], -index)))
        means = [float(np.mean([row[index] for row in seed_rewards]))
                 for index in range(len(candidates))]
        stds = [float(np.std([row[index] for row in seed_rewards]))
                for index in range(len(candidates))]
        mode_top = max(range(len(candidates)),
                       key=lambda index: (sum(top == index for top in top_per_seed),
                                          -index))
        rank_flips = [_pairwise_rank_flip(means, row) for row in seed_rewards]
        report = dict(
            query_count=count,
            seeds=list(seeds),
            query_counts=query_counts,
            reward_mean=means,
            reward_std=stds,
            reward_gap_mean=_mean(gaps),
            reward_gap_std=float(np.std(gaps)) if gaps else None,
            top_index_per_seed=top_per_seed,
            top_index=mode_top,
            top_flip_fraction=float(np.mean(
                [top != mode_top for top in top_per_seed])),
            rank_flip_fraction=float(np.mean(rank_flips)) if rank_flips else 0.0,
            build_seconds_mean=float(np.mean(build_times)),
            score_seconds_mean=float(np.mean(score_times)),
            total_seconds=float(np.sum(build_times)
                                + np.sum(score_times) * len(candidates)))
        if apls_values is None:
            report.update(apls_mean=None, apls_oracle=None, apls_gap=None,
                          apls_selected_mean=None,
                          apls_selected_minus_mean=None)
        else:
            available = [value for value in apls_values if value is not None]
            selected = []
            for row in seed_rewards:
                top = max(range(len(row)), key=lambda index: (row[index], -index))
                if apls_values[top] is not None:
                    selected.append(apls_values[top])
            report.update(
                apls_mean=_mean(available),
                apls_oracle=max(available) if available else None,
                apls_gap=(max(available) - float(np.mean(available))
                          if len(available) >= 2 else None),
                apls_selected_mean=_mean(selected),
                apls_selected_minus_mean=(
                    float(np.mean(selected) - np.mean(available))
                    if selected and available else None))
        reports[count] = report
        mean_rewards[count] = means

    reference_count = 48 if 48 in reports else counts[-1]
    reference_rewards = mean_rewards[reference_count]
    reference_top = reports[reference_count]['top_index']
    for count, report in reports.items():
        report['rank_flip_vs_reference'] = (
            0.0 if count == reference_count
            else _pairwise_rank_flip(reference_rewards, mean_rewards[count]))
        report['top_differs_from_reference'] = bool(
            report['top_index'] != reference_top)
    return dict(query_counts={str(count): reports[count] for count in counts},
                counts=list(counts),
                reference_count=int(reference_count),
                candidate_count=len(candidates),
                seeds=list(seeds),
                apls_available=bool(apls_values is not None),
                apls_status=('provided_adapter' if apls_values is not None
                             else 'missing_official_adapter'),
                note=('Reward is the fixed Stage 0 verifier reward. Official APLS '
                      'fields are null unless an external adapter is supplied.'))


def _rasterize_polylines(polylines, shape):
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for polyline in polylines:
        points = np.rint(np.stack([polyline[:, 1], polyline[:, 0]], axis=1)).astype(int)
        for (y0, x0), (y1, x1) in zip(points, points[1:]):
            rows, columns = _raster_line(int(y0), int(x0), int(y1), int(x1))
            inside = ((rows >= 0) & (rows < height)
                      & (columns >= 0) & (columns < width))
            mask[rows[inside], columns[inside]] = True
    return mask


def polyline_roundtrip(polylines, shape=None, offsets=None, centerline=None,
                       threshold=0.5):
    """Measure raster -> skeleton-vector-graph quantization against native lines.

    ``polylines`` is a sequence of ``(N, 2)`` ``(x, y)`` native polylines.  The
    flat ``(P, 2)`` ``polyline_points`` plus ``polyline_offsets`` layout written
    by ``roadflow.data.prepare`` is also accepted.  If ``centerline`` (the native
    one-pixel raster) is given it is used directly; otherwise the polylines are
    rasterized.  For every polyline the native arc length is compared with the
    vectorized graph shortest path between the projected endpoints, and every
    native vertex is projected to report its quantization distance.  This is a
    diagnostic only, not an official APLS or native-graph evaluation.
    """
    if offsets is not None:
        points = np.asarray(polylines, dtype=float)
        bounds = np.asarray(offsets, dtype=int)
        lines = [points[bounds[i]:bounds[i + 1]]
                 for i in range(len(bounds) - 1)]
    else:
        array = np.asarray(polylines, dtype=float)
        if array.ndim == 2:
            lines = [array]
        else:
            lines = [np.asarray(line, dtype=float) for line in polylines]
    lines = [line for line in lines if len(line) >= 2]
    if centerline is not None:
        mask = np.asarray(centerline, dtype=bool)
        height, width = mask.shape
    else:
        if shape is None:
            if not lines:
                height = width = 2
            else:
                max_x = max(float(np.max(line[:, 0])) for line in lines)
                max_y = max(float(np.max(line[:, 1])) for line in lines)
                height = max(2, int(math.ceil(max_y)) + 2)
                width = max(2, int(math.ceil(max_x)) + 2)
        else:
            height, width = (int(shape[0]), int(shape[1]))
        mask = _rasterize_polylines(lines, (height, width))
    graph = vectorize(mask, threshold=threshold)
    index = _ProjectionIndex(graph)
    cache = {}
    per_polyline = []
    vertex_distances = []
    length_errors = []
    relative_errors = []
    for polyline in lines:
        native_points = np.stack([polyline[:, 1], polyline[:, 0]], axis=1)
        native_length = float(np.sum(np.linalg.norm(
            np.diff(polyline, axis=0), axis=1)))
        first = index.locate(native_points[0])
        second = index.locate(native_points[-1])
        length_error = None
        relative_error = None
        if (first is not None and second is not None and native_length > 0):
            measured = _location_distance(graph, first, second, cache)
            if measured is not None:
                length_error = float(measured - native_length)
                relative_error = float(length_error / native_length)
                length_errors.append(length_error)
                relative_errors.append(relative_error)
        distances = []
        for point in native_points:
            location = index.locate(point)
            if location is not None:
                distances.append(float(location.distance))
        vertex_distances.extend(distances)
        per_polyline.append(dict(
            vertices=int(len(polyline)),
            native_length=native_length,
            length_error=length_error,
            relative_error=relative_error,
            vertex_distance_mean=_mean(distances),
            vertex_distance_max=max(distances) if distances else None))
    return dict(
        polyline_count=len(lines),
        compared=len(length_errors),
        raster_shape=[int(height), int(width)],
        skeleton_pixels=int(mask.sum()),
        graph_nodes=int(graph.number_of_nodes()),
        graph_edges=int(graph.number_of_edges()),
        vertex_distance_mean=_mean(vertex_distances),
        vertex_distance_max=(max(vertex_distances) if vertex_distances else None),
        length_abs_error_mean=(_mean([abs(value) for value in length_errors])
                               if length_errors else None),
        length_abs_error_max=(max(abs(value) for value in length_errors)
                              if length_errors else None),
        length_relative_error_mean=(_mean([abs(value) for value in relative_errors])
                                    if relative_errors else None),
        length_relative_error_max=(max(abs(value) for value in relative_errors)
                                   if relative_errors else None),
        per_polyline=per_polyline)


native_polyline_roundtrip = polyline_roundtrip
# Stage 0 integration name for the 24/48/96 query re-sampling report.
retest_report = query_budget_report
