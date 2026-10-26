"""Protocol tests for the non-repairing vectorized verifier."""
import json
import math
import unittest

import networkx as nx
import numpy as np

from roadflow.verifier import (
    Verifier, VerifierConfig, graph_from_mask, make_queries, native_polyline_roundtrip,
    pixel_distance, polyline_roundtrip, query_budget_report, retest_report,
    sample_query_pool, short_terminal_branch_density, subdivide_graph, vectorize,
)


def cross_fixture(size=64):
    mask = np.zeros((size, size), bool)
    mask[size // 2, 6:size - 6] = True
    mask[6:size - 6, size // 2] = True
    return mask


def ring_fixture(size=64):
    mask = np.zeros((size, size), bool)
    mask[10, 10:size - 10] = True
    mask[size - 10, 10:size - 10] = True
    mask[10:size - 10, 10] = True
    mask[10:size - 10, size - 10] = True
    return mask


def adjacent_junction_fixture():
    """Skeleton with a three-pixel junction cluster (ports must stay exact)."""
    return np.asarray([
        [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ], dtype=bool)


class VectorizationTests(unittest.TestCase):
    def test_legacy_pixel_graph_and_compressed_polylines(self):
        mask = cross_fixture()
        pixel_graph, skeleton = graph_from_mask(mask)
        expected = {tuple(map(int, point)) for point in np.argwhere(skeleton)}
        self.assertEqual(set(pixel_graph.nodes), expected)
        self.assertEqual(pixel_graph.number_of_nodes(), int(skeleton.sum()))

        graph = vectorize(mask)
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertLess(graph.number_of_nodes(), pixel_graph.number_of_nodes())
        self.assertAlmostEqual(
            sum(data['weight'] for *_, data in graph.edges(keys=True, data=True)),
            sum(data['weight'] for *_, data in pixel_graph.edges(data=True)))
        junctions = [node for node, data in graph.nodes(data=True)
                     if data['kind'] == 'junction']
        self.assertEqual(len(junctions), 1)
        self.assertEqual(graph.degree(junctions[0]), 4)
        self.assertEqual([node for node, degree in graph.degree if degree == 2], [])
        for _, _, _, data in graph.edges(keys=True, data=True):
            self.assertEqual(len(data['polyline']), len(data['offsets']))
            self.assertAlmostEqual(data['offsets'][-1], data['weight'])
            rebuilt = sum(math.dist(data['polyline'][i], data['polyline'][i + 1])
                          for i in range(len(data['polyline']) - 1))
            self.assertAlmostEqual(rebuilt, data['weight'])
            self.assertEqual(data['polyline'][0], data['start'])
            self.assertEqual(data['polyline'][-1], data['end'])

    def test_threshold_and_config_argument(self):
        values = np.zeros((24, 24), dtype=float)
        values[12, 4:20] = 0.4
        values[8:13, 12] = 0.6
        graph = vectorize(values, threshold=0.5)
        self.assertTrue(graph.graph['skeleton'][8:13, 12].all())
        self.assertFalse(graph.graph['skeleton'][12, 4:10].any())
        config_graph = vectorize(values, VerifierConfig(threshold=0.5))
        self.assertEqual(sorted(config_graph.edges), sorted(graph.edges))

    def test_loops_isolated_components_and_no_repair(self):
        ring = ring_fixture()
        graph = vectorize(ring)
        self.assertGreater(graph.number_of_edges(), 0)
        # The loop survives as one closed component rather than being dropped.
        undirected = graph.to_undirected()
        self.assertTrue(nx.is_connected(undirected))
        self.assertAlmostEqual(Verifier(ring).score(ring)['path'], 1)

        mixed = np.zeros((40, 48), bool)
        mixed[20, 4:44] = True
        mixed[3, 3] = True
        mixed[30, 30] = True
        mixed_graph = vectorize(mixed)
        isolated = [node for node, data in mixed_graph.nodes(data=True)
                    if data['kind'] == 'isolated']
        self.assertEqual(len(isolated), 2)
        for node in isolated:
            self.assertEqual(mixed_graph.degree(node), 0)

        # A one-pixel gap is not bridged, closed, or repaired.
        gapped = np.zeros((32, 64), bool)
        gapped[16, 4:30] = True
        gapped[16, 32:60] = True
        gapped_graph = vectorize(gapped)
        self.assertEqual(nx.number_connected_components(gapped_graph), 2)
        self.assertAlmostEqual(Verifier(gapped).score(gapped)['path'], 1)

    def test_projection_preserves_arc_length_offset(self):
        line = np.zeros((32, 64), bool)
        line[16, 8:56] = True
        graph = vectorize(line)
        # Only the two endpoints are compressed nodes; interior points still
        # project by arc length rather than snapping to an endpoint.
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertAlmostEqual(pixel_distance(graph, (16, 20), (16, 40)), 20)
        verifier = Verifier(line, queries=[((16, 20), (16, 40), 20.0)])
        self.assertAlmostEqual(verifier.score(line)['path'], 1)
        self.assertEqual(verifier.score(line)['errors']['consistent'], 1)
        # A one-pixel lateral shift still projects to the exact same arc-length
        # positions; only nearest compressed endpoints would be wrong here.
        shifted = np.zeros_like(line)
        shifted[17, 8:56] = True
        self.assertAlmostEqual(verifier.score(shifted)['path'], 1)
        broken = line.copy()
        broken[16, 28:32] = False
        self.assertLess(verifier.score(broken)['path'], 1)

    def test_node_split_invariance(self):
        line = np.zeros((32, 64), bool)
        line[16, 8:56] = True
        graph = vectorize(line)
        split = subdivide_graph(graph, spacing=4)
        self.assertGreater(split.number_of_nodes(), graph.number_of_nodes())
        self.assertAlmostEqual(
            sum(data['weight'] for *_, data in split.edges(keys=True, data=True)),
            sum(data['weight'] for *_, data in graph.edges(keys=True, data=True)))
        pairs = [((16, 8), (16, 55)), ((16, 10), (16, 50)),
                 ((16, 20), (16, 40)), ((16, 26), (16, 31))]
        for u, v in pairs:
            self.assertAlmostEqual(pixel_distance(graph, u, v),
                                   pixel_distance(split, u, v))
        for u in graph.nodes:
            for v in graph.nodes:
                if u < v:
                    self.assertAlmostEqual(pixel_distance(graph, u, v),
                                           pixel_distance(split, u, v))

    def test_compression_preserves_exact_pixel_distances(self):
        mask = adjacent_junction_fixture()
        pixel_graph, _ = graph_from_mask(mask)
        graph = vectorize(mask)
        clusters = graph.graph['junction_clusters']
        self.assertTrue(any(len(members) > 1 for members in clusters.values()))
        # Port nodes are retained, so every pixel-pair distance is exact even
        # through the interior of an adjacent-junction cluster.
        for first in pixel_graph.nodes:
            for second in pixel_graph.nodes:
                try:
                    expected = nx.shortest_path_length(
                        pixel_graph, first, second, weight='weight')
                except nx.NetworkXNoPath:
                    expected = None
                measured = pixel_distance(graph, first, second)
                if expected is None:
                    self.assertIsNone(measured)
                else:
                    self.assertAlmostEqual(measured, expected)
        self.assertAlmostEqual(Verifier(mask).score(mask)['path'], 1)


class MetricTests(unittest.TestCase):
    def test_short_terminal_branch_density_is_a_gt_comparison_proxy(self):
        main = np.zeros((64, 64), bool)
        main[32, 8:56] = True
        spur = main.copy()
        spur[26:33, 40] = True
        clean = short_terminal_branch_density(vectorize(main))
        noisy = short_terminal_branch_density(vectorize(spur))
        self.assertEqual(clean['count'], 0)
        self.assertEqual(noisy['count'], 1)
        self.assertGreater(noisy['density'], clean['density'])
        self.assertGreater(noisy['density'], 0)

        long_branch = main.copy()
        long_branch[8:33, 40] = True
        self.assertEqual(
            short_terminal_branch_density(vectorize(long_branch), max_length=8)['count'],
            0)

        result = Verifier(spur).score(main)
        self.assertIn('short_terminal_branch_density', result)
        self.assertIn('gt_short_terminal_branch_density', result)
        self.assertGreater(result['gt_short_terminal_branch_density'],
                           result['short_terminal_branch_density'])
        self.assertLess(result['short_terminal_branch_density_excess'], 0)
        # The reported fields are a structural proxy, not a ground-truth spur rate.
        self.assertFalse(any('spur' in key for key in result))

    def test_native_polyline_roundtrip_quantization(self):
        line = np.asarray([[5.0, 5.0], [30.0, 5.0], [30.0, 30.0]])
        report = polyline_roundtrip([line])
        self.assertEqual(report['polyline_count'], 1)
        self.assertGreaterEqual(report['compared'], 1)
        self.assertLess(report['length_relative_error_mean'], 0.05)
        self.assertLessEqual(report['vertex_distance_max'], 1.0)
        self.assertEqual(report['raster_shape'], [32, 32])
        self.assertIs(native_polyline_roundtrip, polyline_roundtrip)

        # ``roadflow.data`` stores a flat point array plus offsets.
        points = np.vstack([line, np.asarray([[35.0, 35.0], [40.0, 35.0]])])
        offsets = np.asarray([0, len(line), len(points)], dtype=np.int64)
        flat = polyline_roundtrip(points, offsets=offsets)
        self.assertEqual(flat['polyline_count'], 2)
        self.assertGreaterEqual(flat['compared'], 1)
        self.assertLess(flat['length_abs_error_mean'], 1.0)

        mask = np.zeros((40, 40), bool)
        mask[5, 5:31] = True
        mask[5:31, 30] = True
        with_centerline = polyline_roundtrip([line], centerline=mask)
        self.assertGreaterEqual(with_centerline['compared'], 1)
        self.assertEqual(with_centerline['raster_shape'], [40, 40])


class QueryProtocolTests(unittest.TestCase):
    def test_query_cache_pool_and_json_roundtrip(self):
        gt = cross_fixture()
        verifier = Verifier(gt, VerifierConfig(query_count=16), seed=2)
        cache = json.loads(json.dumps(verifier.cache()))
        self.assertEqual(Verifier(gt, queries=cache['queries']).score(gt)['path'], 1)
        self.assertEqual(make_queries(graph_from_mask(gt)[0], 8, seed=4),
                         make_queries(graph_from_mask(gt)[0], 8, seed=4))

        sampled = sample_query_pool(cache, count=8, seed=3)
        self.assertEqual(len(sampled), 8)
        self.assertEqual(sampled, sample_query_pool(cache, 8, 3))
        self.assertEqual(json.loads(json.dumps(sampled)),
                         [list(query) for query in sampled])
        self.assertAlmostEqual(Verifier(gt, queries=sampled).score(gt)['path'], 1)
        expanded = sample_query_pool(cache['queries'], len(verifier.queries) + 3, seed=1)
        self.assertEqual(len(expanded), len(verifier.queries) + 3)
        self.assertEqual(sample_query_pool([], 5, seed=0), [])
        with self.assertRaises(ValueError):
            sample_query_pool(cache, 0)

    def test_query_budget_report_and_optional_apls(self):
        gt = cross_fixture(64)
        broken = gt.copy()
        broken[30:34, 30:34] = False
        branch = gt.copy()
        branch[10:14, 8:56] = True
        candidates = [gt, broken, branch]
        report = query_budget_report(gt, candidates, counts=(24, 48, 96), seeds=(0, 1))
        self.assertIs(retest_report, query_budget_report)
        self.assertEqual(set(report['query_counts']), {'24', '48', '96'})
        self.assertEqual(report['reference_count'], 48)
        self.assertFalse(report['apls_available'])
        for key in ('24', '48', '96'):
            row = report['query_counts'][key]
            self.assertGreaterEqual(row['rank_flip_fraction'], 0)
            self.assertLessEqual(row['rank_flip_fraction'], 1)
            self.assertGreaterEqual(row['top_flip_fraction'], 0)
            self.assertLessEqual(row['top_flip_fraction'], 1)
            self.assertIsNotNone(row['reward_gap_mean'])
            self.assertGreaterEqual(row['reward_gap_std'], 0)
            self.assertIsNone(row['apls_mean'])
            self.assertIsNone(row['apls_selected_mean'])
            self.assertGreaterEqual(row['total_seconds'], 0)
            self.assertGreaterEqual(row['score_seconds_mean'], 0)
            self.assertIn('top_differs_from_reference', row)
        json.dumps(report, allow_nan=False)

        def scorer(gt_mask, predicted_mask, metadata):
            self.assertIn('candidate', metadata)
            return float(1.0 - predicted_mask.mean())

        with_apls = query_budget_report(gt, candidates, counts=(24,), seeds=(0,),
                                        apls=scorer)
        self.assertTrue(with_apls['apls_available'])
        row = with_apls['query_counts']['24']
        self.assertIsNotNone(row['apls_mean'])
        self.assertIsNotNone(row['apls_selected_mean'])
        self.assertIsNotNone(row['apls_oracle'])
        json.dumps(with_apls, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
