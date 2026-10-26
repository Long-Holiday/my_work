import copy
import unittest

import numpy as np
from scipy import ndimage as ndi
import torch

from roadflow.verifier import Verifier, VerifierConfig, graph_from_mask, make_queries
from roadflow.models import RoadModel, supervised_loss
from roadflow.flow import sample, policy_loss, group_advantages, log_probability
from roadflow.diagnostics import oracle_report


def road_fixture(size=96):
    center = np.zeros((size, size), bool)
    center[size // 2, 8:size - 8] = True
    center[8:size - 8, size // 2] = True
    return ndi.binary_dilation(center, iterations=2)


class VerifierTests(unittest.TestCase):
    def test_self_damage_branch_and_width(self):
        gt = road_fixture()
        verifier = Verifier(gt, VerifierConfig(query_count=96), seed=13)
        own = verifier.score(gt)
        self.assertAlmostEqual(own['path'], 1)
        self.assertAlmostEqual(own['reward'], 1)
        broken = gt.copy()
        broken[40:57, 40:57] = False
        self.assertLess(verifier.score(broken)['path'], .8)
        branch = gt.copy()
        branch[22:25, 10:49] = True
        self.assertLess(verifier.score(branch)['centerline_precision'], .95)
        wide = ndi.binary_dilation(gt, iterations=6)
        self.assertLess(verifier.score(wide)['dice'], own['dice'])
        self.assertLess(verifier.score(wide)['bf1'], own['bf1'])

    def test_empty_isolated_loop_and_no_repair(self):
        empty = np.zeros((64, 64), bool)
        self.assertEqual(Verifier(empty).score(empty)['reward'], 1)
        self.assertEqual(Verifier(empty).score(np.ones_like(empty))['reward'], 0)
        isolated = empty.copy()
        isolated[10, 10] = True
        self.assertEqual(Verifier(isolated).score(isolated)['path'], 0)
        ring = empty.copy()
        ring[10, 10:51] = ring[50, 10:51] = True
        ring[10:51, 10] = ring[10:51, 50] = True
        self.assertAlmostEqual(Verifier(ring).score(ring)['path'], 1)
        gt = road_fixture()
        split = gt.copy()
        split[:, 47:50] = False
        self.assertLess(Verifier(gt).score(split)['path'], 1)

    def test_parallel_road_shortcut(self):
        gt = np.zeros((96, 96), bool)
        gt[16:81, 24] = True
        gt[16:81, 72] = True
        gt[80, 24:73] = True
        graph, _ = graph_from_mask(gt)
        import networkx as nx
        distance = nx.shortest_path_length(graph, (16, 24), (16, 72), weight='weight')
        verifier = Verifier(gt, queries=[((16, 24), (16, 72), distance)])
        shortcut = gt.copy()
        shortcut[16, 24:73] = True
        result = verifier.score(shortcut)
        self.assertEqual(result['errors']['shortened'], 1)
        self.assertLess(result['path'], .5)

    def test_query_cache_is_json_serializable(self):
        import json
        verifier = Verifier(road_fixture())
        restored = json.loads(json.dumps(verifier.cache()))
        self.assertEqual(Verifier(road_fixture(), queries=restored['queries']).score(road_fixture())['path'], 1)

    def test_cached_queries_and_retest_signal(self):
        gt = road_fixture()
        bad = gt.copy()
        bad[40:57, 40:57] = False
        scores = [Verifier(gt, VerifierConfig(query_count=96), seed=s).score(bad)['path']
                  for s in (1, 2)]
        self.assertLess(abs(scores[0] - scores[1]), 1 - np.mean(scores))
        graph, _ = graph_from_mask(gt)
        self.assertEqual(make_queries(graph, seed=4), make_queries(graph, seed=4))
        cache = Verifier(gt).cache()
        self.assertEqual(Verifier(gt, queries=cache['queries']).score(gt)['path'], 1)


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_model_supervised_backward_and_baseline(self):
        torch.manual_seed(2)
        image = torch.rand(2, 3, 32, 32)
        mask = (torch.rand(2, 1, 32, 32) > .8).float()
        for discriminative in (False, True):
            model = RoadModel(discriminative)
            loss = supervised_loss(model, image, mask)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.head.weight.grad)

    def test_sde_ratio_kl_freezing_and_update(self):
        torch.manual_seed(7)
        model = RoadModel().eval()
        model.freeze_encoder()
        reference = copy.deepcopy(model).requires_grad_(False)
        image = torch.rand(4, 3, 32, 32)
        pred, trajectory = sample(model, image, steps=3, noise=.2, record=True)
        self.assertEqual(pred.shape, (4, 1, 32, 32))
        self.assertEqual(len(trajectory), 2)
        features = model.encode(image)
        ref_features = reference.encode(image)
        advantages = group_advantages(torch.tensor([[0., .2, .5, 1.]])).flatten()
        loss, stats = policy_loss(model, reference, features, ref_features, trajectory,
                                  advantages, noise=.2)
        self.assertLess(stats['ratio_max_error'], 1e-6)
        self.assertAlmostEqual(stats['kl'], 0)
        loss.backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertIsNone(model.encoder.conv1.weight.grad)
        self.assertFalse(model.encoder.training)
        before = model.head.weight.detach().clone()
        torch.optim.Adam(model.parameters(), lr=1e-4).step()
        self.assertFalse(torch.equal(before, model.head.weight))
        with self.assertRaises(ValueError):
            log_probability(pred, pred, 0)
        self.assertTrue(torch.equal(group_advantages(torch.ones(2, 4)), torch.zeros(2, 4)))

    def test_ode_reproducibility(self):
        model = RoadModel().eval()
        image = torch.rand(1, 3, 32, 32)
        a, _ = sample(model, image, steps=2, generator=torch.Generator().manual_seed(1))
        b, _ = sample(model, image, steps=2, generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(a, b))


class DiagnosticTests(unittest.TestCase):
    def test_eligibility_oracle_and_missing_apls(self):
        candidates = [dict(dice=.9, bf1=.9, path=.4, reward=.5, apls=.3),
                      dict(dice=.91, bf1=.9, path=.8, reward=.8, apls=.7),
                      dict(dice=.1, bf1=.1, path=1, reward=1, apls=1)]
        rows = [dict(id='x', reference=dict(dice=.9, bf1=.9), candidates=candidates)]
        report = oracle_report(rows)
        self.assertAlmostEqual(report['apls_gap_mean'], .2)
        self.assertAlmostEqual(report['eligible_fraction'], 2 / 3)
        for c in candidates:
            c['apls'] = None
        self.assertEqual(oracle_report(rows)['decision'], 'blocked_missing_official_apls')


if __name__ == '__main__':
    unittest.main()
