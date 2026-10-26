"""Bounded synthetic protocol integration, with no downloads or full training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from roadflow import cli
from roadflow.diagnostics import oracle_report, select_distillation_candidate
from roadflow.flow import group_advantages
from roadflow.verifier import graph_from_mask, make_queries


class TinyDataset:
    def __init__(self, split):
        self.split = split
        self.ids = [f'{split}_{i}' for i in range(2)]
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[15:18, 3:29] = 1
        mask[3:29, 15:18] = 1
        graph, _ = graph_from_mask(mask)
        self.pool = make_queries(graph, count=120)
        self.mask = torch.from_numpy(mask)[None]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return dict(id=self.ids[index], source_id=self.ids[index], query_pool=self.pool,
                    image=torch.ones(3, 32, 32) * .3, mask=self.mask.clone())


def run_command(*args):
    with patch('sys.argv', ['roadflow', *map(str, args)]), patch(
            'roadflow.cli.get_dataset', side_effect=lambda args, split: TinyDataset(split)):
        cli.main()


class RevisedProtocolTests(unittest.TestCase):
    def test_low_variance_is_not_amplified(self):
        rewards = torch.tensor([[.8, .8000001, .7999999, .8]])
        self.assertTrue(torch.equal(group_advantages(rewards), torch.zeros_like(rewards)))

    def test_geometry_selection_random_and_gt_fallback(self):
        row = dict(reference=dict(dice=.9, bf1=.9), candidates=[
            dict(dice=.1, bf1=.1, reward=1), dict(dice=.9, bf1=.9, reward=.7),
            dict(dice=.9, bf1=.9, reward=.5)])
        rng = np.random.default_rng(9)
        self.assertEqual(select_distillation_candidate(row, 'reward', rng), 1)
        self.assertTrue({select_distillation_candidate(row, 'random', rng) for _ in range(20)} == {1, 2})
        row['reference']['dice'] = 1
        self.assertIsNone(select_distillation_candidate(row, 'reward', rng))

    def test_reward_selected_apls_and_insufficient_pairs(self):
        row = dict(id='a', reference=dict(dice=.9, bf1=.9), stratum='complex', candidates=[
            dict(dice=.9, bf1=.9, reward=.8, path=.7, apls=.9),
            dict(dice=.9, bf1=.9, reward=.4, path=.3, apls=.3)])
        report = oracle_report([row])
        self.assertAlmostEqual(report['reward_selection_apls_gain_mean'], .3)
        self.assertAlmostEqual(report['reward_selected_apls_mean'], .9)
        self.assertIn('complex', report['strata'])
        row['candidates'].pop()
        self.assertIsNone(oracle_report([row])['reward_selection_apls_gain_mean'])

    def test_shared_batch_schedule(self):
        self.assertEqual(cli.batch_indices(20, 4, 3, 10), cli.batch_indices(20, 4, 3, 10))
        self.assertNotEqual(cli.batch_indices(20, 4, 3, 10), cli.batch_indices(20, 4, 3, 11))

    def test_end_to_end_one_update_each(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'manifest.json').write_text('{}')
            common = ['--manifest', str(root / 'manifest.json'), '--limit', '2']
            run_command('train', *common, '--output', root / 'b0', '--steps', '1',
                        '--batch-size', '2', '--no-pretrained', '--sample-steps', '2')
            checkpoint = root / 'b0/last.pt'
            run_command('candidates', *common, '--checkpoint', checkpoint,
                        '--output', root / 'pool', '--split', 'train', '--group-size', '2',
                        '--sample-steps', '2', '--reference-samples', '1')
            cache = json.loads((root / 'pool/candidates.json').read_text())
            # Force a known empty qualified pool to exercise identical GT fallback.
            for row in cache['records']:
                row['reference'] = dict(dice=2., bf1=2.)
            (root / 'pool/candidates.json').write_text(json.dumps(cache))
            for selection in ('reward', 'random'):
                run_command('distill', *common, '--checkpoint', checkpoint,
                            '--candidate-cache', root / 'pool/candidates.json', '--selection', selection,
                            '--output', root / selection, '--steps', '1', '--batch-size', '2', '--sample-steps', '2')
                report = json.loads((root / selection / 'selection.json').read_text())
                self.assertEqual(report['gt_fallback_fraction'], 1)
            run_command('train', *common, '--checkpoint', checkpoint, '--output', root / 'b1',
                        '--steps', '1', '--batch-size', '2', '--sample-steps', '2')
            schedules = [json.loads((root / name / 'train.jsonl').read_text())['ids']
                         for name in ('b1', 'reward', 'random')]
            self.assertEqual(schedules[0], schedules[1])
            self.assertEqual(schedules[1], schedules[2])
            run_command('grpo', *common, '--checkpoint', checkpoint, '--output', root / 'grpo',
                        '--steps', '1', '--group-size', '4', '--sample-steps', '8', '--noise', '.2')
            stats = json.loads((root / 'grpo/train.jsonl').read_text())
            self.assertLess(stats['ratio_max_error'], 1e-5)
            with self.assertRaises(ValueError):
                run_command('evaluate', *common, '--checkpoint', checkpoint,
                            '--output', root / 'leak', '--split', 'holdout')
            # Frozen confirmation records three independent single-candidate runs.
            (root / 'manifest.json').write_text('{}')
            run_command('confirm', '--manifest', root / 'manifest.json', '--output', root / 'confirm', '--confirm-holdout',
                        '--sample-steps', '2', '--models', f'B0={checkpoint}', f'B1={checkpoint}')
            report = json.loads((root / 'confirm/confirmation.json').read_text())
            pairs = report['paired_region_differences']['B1']['B0']
            self.assertEqual(len(pairs), 6)
            self.assertTrue(all(row['dice'] == 0 for row in pairs))


if __name__ == '__main__':
    unittest.main()
