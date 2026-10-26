"""Candidate diagnostics and an explicit independent official-APLS adapter."""
import importlib
import numpy as np
from scipy.stats import spearmanr


def official_scorer(spec):
    """Load module:function(gt_mask, predicted_mask, metadata)->float.

    The adapter must run a separately pinned official implementation, including
    native-graph conversion. No local reward is substituted for missing APLS.
    """
    if not spec:
        return None
    module, function = spec.split(':', 1)
    return getattr(importlib.import_module(module), function)


def eligible_indices(candidates, reference, delta_dice=.005, delta_bf1=.01):
    return [i for i, candidate in enumerate(candidates)
            if candidate['dice'] >= reference['dice'] - delta_dice
            and candidate['bf1'] >= reference['bf1'] - delta_bf1]


def select_distillation_candidate(row, strategy, rng, delta_dice=.005, delta_bf1=.01):
    eligible = eligible_indices(row['candidates'], row['reference'], delta_dice, delta_bf1)
    if not eligible:
        return None
    if strategy == 'random':
        return int(rng.choice(eligible))
    if strategy == 'reward':
        return max(eligible, key=lambda i: row['candidates'][i]['reward'])
    raise ValueError('Selection must be reward or random')


def oracle_report(records, delta_dice=0.005, delta_bf1=0.01):
    reports = []
    for row in records:
        candidates = row['candidates']
        eligible = [candidates[i] for i in eligible_indices(candidates, row['reference'], delta_dice, delta_bf1)]
        report = dict(id=row['id'], stratum=row.get('stratum', 'unclassified'),
                      eligible=len(eligible), count=len(candidates))
        selected = max(eligible, key=lambda c: c['reward']) if eligible else None
        complete_apls = bool(eligible) and all(c.get('apls') is not None for c in eligible)
        report['reward_selected_apls'] = selected.get('apls') if selected else None
        report['reward_selection_apls_gain'] = (float(selected['apls'] - np.mean([c['apls'] for c in eligible]))
                                               if complete_apls and len(eligible) >= 2 else None)
        for metric in ('path', 'reward', 'apls'):
            values = [c[metric] for c in eligible if c.get(metric) is not None]
            report[metric + '_mean'] = float(np.mean(values)) if values else None
            report[metric + '_oracle'] = max(values) if values else None
            report[metric + '_gap'] = float(max(values) - np.mean(values)) if len(values) >= 2 else None
        paired = [c for c in eligible if c.get('apls') is not None]
        correlation = None
        if len(paired) >= 3:
            a, b = [c['path'] for c in paired], [c['apls'] for c in paired]
            if np.ptp(a) > 0 and np.ptp(b) > 0:
                correlation = float(spearmanr(a, b).statistic)
        report['path_apls_spearman'] = correlation
        reports.append(report)
    total = sum(r['count'] for r in reports)
    summary = dict(images=reports, eligible_fraction=sum(r['eligible'] for r in reports) / max(1, total),
                   fewer_than_two_fraction=float(np.mean([r['eligible'] < 2 for r in reports])) if reports else None)
    for key in ('path_gap', 'reward_gap', 'apls_gap', 'path_apls_spearman',
                'reward_selected_apls', 'reward_selection_apls_gain'):
        values = [r[key] for r in reports if r[key] is not None]
        summary[key + '_mean'] = float(np.mean(values)) if values else None
    summary['decision'] = ('needs_review_of_signal_and_selection_gain' if summary['apls_gap_mean'] is not None
                           else 'blocked_missing_official_apls' if not any(c.get('apls') is not None for row in records for c in row['candidates'])
                           else 'insufficient_eligible_candidates')
    summary['strata'] = {}
    for stratum in sorted({r['stratum'] for r in reports}):
        subset = [r for r in reports if r['stratum'] == stratum]
        summary['strata'][stratum] = dict(images=len(subset))
        for key in ('apls_gap', 'reward_selection_apls_gain'):
            values = [r[key] for r in subset if r[key] is not None]
            summary['strata'][stratum][key + '_mean'] = float(np.mean(values)) if values else None
    return summary
