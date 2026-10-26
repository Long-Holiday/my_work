"""Staged CLI. Training commands require an explicit positive step budget."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi
import torch
import hashlib

from .diagnostics import official_scorer, oracle_report, select_distillation_candidate
from .flow import group_advantages, policy_loss, sample
from .models import RoadModel, supervised_loss
from .verifier import Verifier, VerifierConfig


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def append_log(path, row):
    with Path(path).open('a') as file:
        file.write(json.dumps(row, allow_nan=False) + '\n')


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model = RoadModel(payload['discriminative'], encoder_kind=payload.get('encoder_kind', 'half_resnet18')).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    return model


def save_model(model, output, args, step):
    torch.save(dict(model=model.state_dict(), discriminative=model.discriminative,
                    encoder_kind=model.encoder_kind, step=step, config=vars(args)), Path(output) / 'last.pt')


def get_dataset(args, split):
    from .data import RoadDataset
    return RoadDataset(args.manifest, split)


def make_verifier(item, seed, count=48):
    from .verifier import sample_query_pool
    gt = item['mask'].squeeze().numpy()
    pool = item.get('query_pool')
    if pool is None:
        raise ValueError('Missing offline GT query pool; regenerate the prepared manifest')
    return Verifier(gt, VerifierConfig(query_count=count), seed=seed,
                    queries=sample_query_pool(pool, count, seed))


def batch_indices(length, batch_size, seed, step):
    # Identical image schedule for B1, B3 and B3-random, independent of torch RNG.
    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    return rng.choice(length, size=min(batch_size, length), replace=False).tolist()


def checkpoint_due(args, step):
    thirds = {max(1, int(np.ceil(args.steps * fraction / 3))) for fraction in (1, 2, 3)}
    return step % args.eval_every == 0 or step == args.steps or step in thirds


def training_batch(dataset, indices):
    items = [dataset[i] for i in indices]
    return items, torch.stack([i['image'] for i in items]), torch.stack([i['mask'] for i in items])


def stage0(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        dataset = get_dataset(args, 'dev')
        masks = [dataset[i]['mask'].squeeze().numpy() for i in range(min(args.limit, len(dataset)))]
    else:
        gt = np.zeros((96, 96), bool)
        gt[48, 8:88] = True
        gt[8:88, 48] = True
        masks = [ndi.binary_dilation(gt, iterations=2)]
    rows = []
    for index, mask in enumerate(masks):
        gt = mask.astype(bool)
        verifier = Verifier(gt, seed=args.seed)
        broken = gt.copy()
        # Delete around an actual road pixel, even for sparse/curved real masks.
        pixels = np.argwhere(verifier.skeleton)
        if len(pixels):
            y, x = pixels[len(pixels) // 2]
            broken[max(0, y - 5):y + 6, max(0, x - 5):x + 6] = False
        branch = gt.copy()
        if len(pixels):
            from skimage.draw import line
            distance = ndi.distance_transform_edt(~gt)
            target = np.unravel_index(np.argmax(distance), gt.shape)
            origin = pixels[np.argmin(np.square(pixels - np.array(target)).sum(axis=1))]
            extension = np.zeros_like(gt)
            rr, cc = line(int(origin[0]), int(origin[1]), int(target[0]), int(target[1]))
            extension[rr, cc] = True
            branch |= ndi.binary_dilation(extension, iterations=1)
        candidates = dict(identity=gt, deleted=broken, widened=ndi.binary_dilation(gt, iterations=5),
                          branch=branch)
        scores = {name: verifier.score(mask) for name, mask in candidates.items()}
        retest = {str(count): [Verifier(gt, VerifierConfig(query_count=count), seed=s).score(broken)['path']
                               for s in (args.seed, args.seed + 1)] for count in (24, 48, 96)}
        from .verifier import query_budget_report
        stability = query_budget_report(gt, list(candidates.values()), seeds=(args.seed, args.seed + 1))
        save_json(output / f'queries_{index}.json', verifier.cache())
        roundtrip = None
        if args.manifest:
            from .verifier import polyline_roundtrip
            with np.load(dataset[index]['npz'], allow_pickle=False) as saved:
                roundtrip = polyline_roundtrip(saved['polyline_points'], offsets=saved['polyline_offsets'],
                                               centerline=saved['centerline'])
        rows.append(dict(index=index, scores=scores, retest=retest, query_budget=stability,
                         native_roundtrip=roundtrip,
                         self_pass=(not verifier.queries or abs(scores['identity']['path'] - 1) < 1e-6),
                         has_path_evidence=bool(verifier.queries)))
    report = dict(stage=0, source='manifest' if args.manifest else 'synthetic', images=rows,
                  self_checks_pass=bool(rows) and all(r['self_pass'] for r in rows),
                  note='Inspect perturbation directions and retest gaps before training. Native graph roundtrip requires prepared-data tests.')
    save_json(output / 'stage0.json', report)
    print(json.dumps(report, ensure_ascii=False))


def train(args):
    if args.steps < 1:
        raise ValueError('Set --steps explicitly to a positive budget')
    dataset = get_dataset(args, 'train')
    if len(dataset) < 2:
        raise ValueError('Training requires at least two crops for BatchNorm')
    if args.checkpoint:
        model = load_model(args.checkpoint, args.device)
        if model.discriminative != args.discriminative:
            raise ValueError('Checkpoint model type differs from --discriminative')
    else:
        model = RoadModel(args.discriminative, pretrained=not args.no_pretrained).to(args.device)
    if args.batch_size < 2:
        raise ValueError('Use batch-size >=2 for supervised encoder training')
    if args.checkpoint:
        model.freeze_encoder()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.train()
    start = time.perf_counter()
    best = float('-inf')
    for step in range(1, args.steps + 1):
        items, image, mask = training_batch(dataset, batch_indices(len(dataset), args.batch_size, args.seed, step))
        image, mask = image.to(args.device), mask.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss(model, image, mask)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite supervised loss')
        loss.backward()
        optimizer.step()
        append_log(output / 'train.jsonl', dict(step=step, loss=loss.item(), ids=[i['id'] for i in items], elapsed=time.perf_counter() - start))
        if checkpoint_due(args, step):
            best = supervised_checkpoint(model, args, step, best)
            model.train()
    save_model(model, output, args, args.steps)
    print(f'Saved {output / "last.pt"}; {args.steps} updates')


@torch.no_grad()
def predict(model, image, args, seed, candidates=1):
    generator = torch.Generator(device=args.device).manual_seed(seed)
    if model.discriminative:
        return model(image).sigmoid().repeat(candidates, 1, 1, 1)
    result, _ = sample(model, image.repeat(candidates, 1, 1, 1), args.sample_steps,
                       args.noise, generator=generator)
    return result


def evaluate(args):
    model = load_model(args.checkpoint, args.device)
    split = getattr(args, 'split', 'dev')
    if split == 'holdout' and not getattr(args, 'confirm_holdout', False):
        raise ValueError('Holdout evaluation requires --confirm-holdout after freezing the protocol')
    dataset = get_dataset(args, split)
    if not len(dataset):
        raise ValueError(f'No {split} crops available')
    scorer = official_scorer(args.apls_adapter)
    rows = []
    if str(args.device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    for index in range(min(args.limit, len(dataset))):
        item = dataset[index]
        image = item['image'][None].to(args.device)
        pred = predict(model, image, args, args.seed + index)[0, 0].cpu().numpy()
        score = make_verifier(item, 2000000 + args.seed + index, args.query_count).score(pred)
        score['apls'] = float(scorer(item['mask'].squeeze().numpy(), pred,
                              {'id': item['id'], 'source_id': item.get('source_id'), 'manifest': args.manifest})) if scorer else None
        rows.append(dict(id=item['id'], source_id=item.get('source_id', item['id']), **score))
    if str(args.device).startswith('cuda'):
        torch.cuda.synchronize(args.device)
    keys = ('dice', 'bf1', 'centerline_precision', 'path', 'reward', 'endpoint_density',
            'short_branch_density', 'small_component_fraction')
    report = dict(stage=1, split=split, seed=args.seed, single_candidate=True, images=rows,
                  elapsed=time.perf_counter() - start,
                  peak_memory_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else None,
                  sampler=dict(steps=args.sample_steps, noise=args.noise),
                  means={key: float(np.mean([r[key] for r in rows])) for key in keys},
                  apls_status='provided_adapter' if scorer else 'missing_official_adapter')
    save_json(Path(args.output) / 'evaluation.json', report)
    print(json.dumps(report['means']))
    return report


def supervised_checkpoint(model, args, step, best):
    output = Path(args.output)
    save_model(model, output, args, step)
    torch.save(torch.load(output / 'last.pt', weights_only=True), output / f'step_{step:06d}.pt')
    evaluation_args = copy.copy(args)
    evaluation_args.checkpoint = str(output / 'last.pt')
    evaluation_args.output = str(output / f'dev_{step:06d}')
    evaluation_args.split = 'dev'
    if step % args.apls_every != 0 and step != args.steps:
        evaluation_args.apls_adapter = None
    report = evaluate(evaluation_args)
    apls = float(np.mean([r['apls'] for r in report['images']])) if evaluation_args.apls_adapter else None
    append_log(output / 'validation.jsonl', dict(step=step, **report['means'], apls=apls))
    # For B0 this is a development-only joint geometry/structure ranking, not convergence proof.
    quality = report['means']['dice'] + report['means']['bf1'] + report['means']['path']
    if quality > best:
        best = quality
        torch.save(torch.load(output / 'last.pt', weights_only=True), output / 'best.pt')
    return best


def candidates(args):
    model = load_model(args.checkpoint, args.device)
    if model.discriminative:
        raise ValueError('Candidate diagnosis requires a generative FM checkpoint')
    dataset = get_dataset(args, args.split)
    scorer = official_scorer(args.apls_adapter)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(min(args.limit, len(dataset))):
        item = dataset[index]
        verifier = make_verifier(item, args.seed + index, args.query_count)
        image = item['image'][None].to(args.device)
        # Independent fixed references, averaged before inspecting candidate selection.
        references = [verifier.score(predict(model, image, args, args.seed + 1000000 + index + j * 10000)[0, 0].cpu().numpy())
                      for j in range(args.reference_samples)]
        reference_score = {key: float(np.mean([r[key] for r in references])) for key in ('dice', 'bf1')}
        if str(args.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        predictions = predict(model, image, args, args.seed + index, args.group_size)[:, 0].cpu().numpy()
        generation_seconds = time.perf_counter() - start
        scores = []
        for pred in predictions:
            score = verifier.score(pred)
            score['apls'] = float(scorer(item['mask'].squeeze().numpy(), pred,
                                  {'id': item['id'], 'source_id': item.get('source_id'), 'manifest': args.manifest})) if scorer else None
            scores.append(score)
        filename = f'candidates_{index:05d}.npz'
        np.savez_compressed(output / filename, masks=predictions.astype(np.float32))
        save_json(output / f'queries_{index:05d}.json', verifier.cache())
        from .verifier import query_budget_report
        official_values = [s['apls'] for s in scores]
        stability = query_budget_report(item['mask'].squeeze().numpy(), predictions,
                     seeds=(args.seed + index, args.seed + index + 1),
                     apls=(lambda gt, pred, metadata: official_values[metadata['candidate']]) if scorer else None)
        save_json(output / f'query_stability_{index:05d}.json', stability)
        records.append(dict(id=item['id'], source_id=item.get('source_id', item['id']),
                            stratum=item.get('stratum', 'unclassified'), split=args.split, file=filename,
                            reference=reference_score, candidates=scores, generation_seconds=generation_seconds,
                            peak_memory_bytes=torch.cuda.max_memory_allocated(args.device)
                            if str(args.device).startswith('cuda') else None))
    save_json(output / 'candidates.json', dict(checkpoint=str(Path(args.checkpoint).resolve()),
                                             manifest=str(Path(args.manifest).resolve()),
                                             split=args.split, config=vars(args), records=records,
                                             checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                                             manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                                             geometry_tolerances=dict(delta_dice=args.delta_dice, delta_bf1=args.delta_bf1)))
    report = oracle_report(records, args.delta_dice, args.delta_bf1)
    save_json(output / 'oracle.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'images'}))


def distill(args):
    if args.steps < 1:
        raise ValueError('Set a positive --steps budget')
    model = load_model(args.checkpoint, args.device)
    if model.discriminative:
        raise ValueError('B3 requires an FM model')
    cache_path = Path(args.candidate_cache)
    cache = json.loads(cache_path.read_text())
    # Validation candidates diagnose H1, but must never become training targets.
    if cache['split'] != 'train':
        raise ValueError('B3 requires candidates generated with --split train; validation leakage forbidden')
    if Path(cache['checkpoint']).resolve() != Path(args.checkpoint).resolve():
        raise ValueError('B3 candidate cache and starting checkpoint differ')
    if Path(cache['manifest']).resolve() != Path(args.manifest).resolve():
        raise ValueError('B3 candidate cache and data manifest differ')
    for key, path in (('checkpoint_sha256', args.checkpoint), ('manifest_sha256', args.manifest)):
        if cache.get(key) != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError('Candidate cache source content changed; regenerate the cache')
    dataset = get_dataset(args, 'train')
    records = {row['id']: row for row in cache['records']}
    if set(records) != set(dataset.ids):
        raise ValueError('Matched B1/B3/random requires a candidate cache for every training crop')
    tolerances = cache.get('geometry_tolerances')
    if tolerances != dict(delta_dice=args.delta_dice, delta_bf1=args.delta_bf1):
        raise ValueError('Geometry tolerances differ from the frozen candidate cache protocol')
    model.freeze_encoder()
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best = float('-inf')
    rng = np.random.default_rng(args.seed)
    selections = {key: select_distillation_candidate(row, args.selection, rng, args.delta_dice, args.delta_bf1)
                  for key, row in sorted(records.items())}
    fallback_fraction = sum(i is None for i in selections.values()) / len(selections)
    save_json(output / 'selection.json', dict(selection=args.selection, indices=selections,
                                             gt_fallback_fraction=fallback_fraction))
    for step in range(1, args.steps + 1):
        items, image, masks = training_batch(dataset, batch_indices(len(dataset), args.batch_size, args.seed, step))
        for j, item in enumerate(items):
            selected = selections[item['id']]
            if selected is not None:
                with np.load(cache_path.parent / records[item['id']]['file'], allow_pickle=False) as saved:
                    masks[j, 0] = torch.from_numpy((saved['masks'][selected] >= .5).astype(np.float32))
        image, masks = image.to(args.device), masks.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss(model, image, masks)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite distillation loss')
        loss.backward()
        optimizer.step()
        append_log(output / 'train.jsonl', dict(step=step, loss=loss.item(), ids=[i['id'] for i in items],
                                               gt_fallback_fraction=fallback_fraction))
        if checkpoint_due(args, step):
            best = supervised_checkpoint(model, args, step, best)
            model.train()
    save_model(model, output, args, args.steps)
    print(f'Saved distilled model to {output / "last.pt"}')


def grpo(args):
    if args.steps < 1 or args.group_size not in (4, 8) or args.noise <= 0 or args.sample_steps not in (8, 16):
        raise ValueError('GRPO protocol requires positive steps/noise, G=4 or 8 and 8 or 16 sampling steps')
    model = load_model(args.checkpoint, args.device)
    if model.discriminative:
        raise ValueError('GRPO requires an FM checkpoint')
    model.freeze_encoder()
    # GroupNorm in the velocity field and frozen/eval BatchNorm ensure exact replay.
    model.eval()
    reference = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    dataset = get_dataset(args, 'train')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    reference_geometry = {}
    baseline_args = copy.copy(args)
    baseline_args.output = str(output / 'val_reference')
    baseline_report = evaluate(baseline_args)
    best_apls = float('-inf')
    for step in range(1, args.steps + 1):
        item = dataset[(step - 1) % len(dataset)]
        image = item['image'][None].repeat(args.group_size, 1, 1, 1).to(args.device)
        start = time.perf_counter()
        predictions, trajectory = sample(model, image, args.sample_steps, args.noise, generator, record=True)
        rollout_seconds = time.perf_counter() - start
        start = time.perf_counter()
        verifier = make_verifier(item, args.seed + step, args.query_count)
        if item['id'] not in reference_geometry:
            reference_prediction = predict(reference, image[:1], args, args.seed + 1000000 + (step - 1) % len(dataset))[0, 0].cpu().numpy()
            reference_geometry[item['id']] = verifier.score(reference_prediction)
        geometry = reference_geometry[item['id']]
        scores = [verifier.score(p[0].cpu().numpy()) for p in predictions]
        rewards = torch.tensor([s['reward'] for s in scores], device=args.device)
        advantages = group_advantages(rewards[None]).flatten()
        reward_seconds = time.perf_counter() - start
        with torch.no_grad():
            features = model.encode(image)
            reference_features = reference.encode(image)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = policy_loss(model, reference, features, reference_features, trajectory,
                                  advantages, args.noise, beta=args.beta)
        # This guard runs BEFORE every update; old is exactly the rollout policy.
        if stats['ratio_max_error'] > 1e-5:
            raise RuntimeError(f'Pre-update ratio failed: {stats}')
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite policy loss')
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize(args.device)
        update_start = time.perf_counter()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # No reward signal and no KL displacement: avoid AdamW weight decay drift.
        skipped_update = not bool(advantages.any()) and stats['kl'] == 0
        if not skipped_update:
            optimizer.step()
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize(args.device)
        update_seconds = time.perf_counter() - update_start
        keys = ('path', 'centerline_precision', 'dice', 'bf1')
        matrix = np.array([[s[k] for k in keys] for s in scores])
        std = matrix.std(axis=0)
        centered = matrix - matrix.mean(axis=0)
        covariance = centered.T @ centered / len(matrix)
        denominator = std[:, None] * std[None, :]
        correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
        row = dict(step=step, reward=rewards.mean().item(), reward_std=rewards.std(unbiased=False).item(),
                   loss=loss.item(), rollout_seconds=rollout_seconds, reward_seconds=reward_seconds,
                   update_seconds=update_seconds, skipped_update=skipped_update,
                   component_mean=dict(zip(keys, matrix.mean(axis=0).tolist())),
                   component_std=dict(zip(keys, std.tolist())),
                   weighted_std=(std * np.array(verifier.config.weights)).tolist(),
                   component_correlation=correlation.tolist(), low_variance=bool(rewards.std(unbiased=False) < 1e-4),
                   geometry_pass_fraction=float(np.mean([s['dice'] >= geometry['dice'] - args.delta_dice
                       and s['bf1'] >= geometry['bf1'] - args.delta_bf1 for s in scores])),
                   low_variance_type=('all_high' if rewards.min() >= .8 else
                                      'all_low' if rewards.max() <= .2 else 'other')
                       if rewards.std(unbiased=False) < 1e-4 else None,
                   **stats)
        append_log(output / 'train.jsonl', row)
        if checkpoint_due(args, step):
            save_model(model, output, args, step)
            torch.save(torch.load(output / 'last.pt', weights_only=True), output / f'step_{step:06d}.pt')
            evaluation_args = copy.copy(args)
            evaluation_args.checkpoint = str(output / 'last.pt')
            evaluation_args.output = str(output / f'dev_{step:06d}')
            evaluation_args.split = 'dev'
            if step % args.apls_every != 0 and step != args.steps:
                evaluation_args.apls_adapter = None
            report = evaluate(evaluation_args)
            apls = float(np.mean([r['apls'] for r in report['images']])) if evaluation_args.apls_adapter and report['images'] else None
            geometry_pass = bool(report['images']) and (
                report['means']['dice'] >= baseline_report['means']['dice'] - args.delta_dice
                and report['means']['bf1'] >= baseline_report['means']['bf1'] - args.delta_bf1)
            append_log(output / 'validation.jsonl', dict(step=step, **report['means'],
                       apls=apls, geometry_pass=geometry_pass))
            if geometry_pass and apls is not None and apls > best_apls:
                best_apls = apls
                torch.save(dict(model=model.state_dict(), discriminative=False, encoder_kind=model.encoder_kind,
                                step=step, config=vars(args)), output / 'best.pt')
    print(f'Saved GRPO model to {output / "last.pt"}')


def confirm(args):
    if not args.confirm_holdout:
        raise ValueError('Freeze all choices before passing --confirm-holdout')
    if args.limit != 1000000:
        raise ValueError('Final confirmation must evaluate the entire holdout; --limit is not allowed')
    if len(set(args.inference_seeds)) != 3:
        raise ValueError('Final confirmation requires exactly three distinct fixed inference seeds')
    model_paths = dict(entry.split('=', 1) for entry in args.models)
    if not {'B0', 'B1'}.issubset(model_paths):
        raise ValueError('Final confirmation requires named B0 and B1 checkpoints')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    adapter_hash = None
    if args.apls_adapter:
        import inspect
        adapter_file = inspect.getsourcefile(official_scorer(args.apls_adapter))
        adapter_hash = hashlib.sha256(Path(adapter_file).read_bytes()).hexdigest() if adapter_file else None
    protocol = dict(models={k: dict(path=str(Path(p).resolve()), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
                            for k, p in model_paths.items()}, seeds=args.inference_seeds,
                    manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                    sample_steps=args.sample_steps, noise=args.noise, query_count=args.query_count,
                    verifier=vars(VerifierConfig()), delta_dice=args.delta_dice, delta_bf1=args.delta_bf1,
                    apls_adapter=args.apls_adapter, apls_adapter_sha256=adapter_hash)
    protocol_path = output / 'frozen_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError('This holdout run already has a different frozen protocol; do not retune on holdout')
    else:
        save_json(protocol_path, protocol)
    reports = {}
    for name, checkpoint in model_paths.items():
        reports[name] = []
        for seed in args.inference_seeds:
            evaluation_args = copy.copy(args)
            evaluation_args.checkpoint = checkpoint
            evaluation_args.seed = seed
            evaluation_args.split = 'holdout'
            evaluation_args.output = str(output / name / f'seed_{seed}')
            reports[name].append(evaluate(evaluation_args))
    comparisons = {}
    for name, runs in reports.items():
        comparisons[name] = {}
        for baseline in ('B0', 'B1'):
            paired = []
            for run, base in zip(runs, reports[baseline]):
                base_rows = {row['id']: row for row in base['images']}
                regions = {}
                for row in run['images']:
                    region = regions.setdefault(row['source_id'], [])
                    other = base_rows[row['id']]
                    region.append({key: row[key] - other[key] if row[key] is not None and other[key] is not None else None
                                   for key in ('apls', 'dice', 'bf1')})
                for region, rows in regions.items():
                    paired.append(dict(seed=run['seed'], source_id=region,
                        **{key: float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None
                           for key in ('apls', 'dice', 'bf1')}))
            comparisons[name][baseline] = paired
    summaries = {}
    for name, runs in reports.items():
        summaries[name] = {}
        for key in ('dice', 'bf1', 'centerline_precision', 'path', 'reward', 'apls'):
            seed_means = []
            for run in runs:
                regions = {}
                for row in run['images']:
                    if row[key] is not None:
                        regions.setdefault(row['source_id'], []).append(row[key])
                if regions:
                    seed_means.append(float(np.mean([np.mean(values) for values in regions.values()])))
            summaries[name][key] = dict(seed_means=seed_means,
                mean=float(np.mean(seed_means)) if seed_means else None,
                std=float(np.std(seed_means)) if seed_means else None)
    save_json(output / 'confirmation.json', dict(region_weighted_seed_summary=summaries,
              paired_region_differences=comparisons,
              status='review_consistency_geometry_and_cost' if args.apls_adapter else 'blocked_missing_official_apls'))


def scan(args):
    configurations = [(steps, 0.) for steps in (8, 16, 32)] + [(args.sample_steps, n) for n in args.noise_levels]
    rows = []
    for steps, noise in configurations:
        current = copy.copy(args)
        current.sample_steps, current.noise = steps, noise
        current.output = str(Path(args.output) / f'steps{steps}_noise{noise}')
        current.split = 'dev'
        report = evaluate(current)
        # Candidate timing is measured separately at exactly the same sampler.
        model = load_model(args.checkpoint, args.device)
        item = get_dataset(args, 'dev')[0]
        image = item['image'][None].to(args.device)
        if str(args.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        predict(model, image, current, args.seed, 16)
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize(args.device)
        rows.append(dict(steps=steps, noise=noise, single_candidate=report,
                         sixteen_candidate_seconds=time.perf_counter() - start,
                         sixteen_candidate_peak_bytes=torch.cuda.max_memory_allocated(args.device)
                         if str(args.device).startswith('cuda') else None))
    save_json(Path(args.output) / 'sampling_scan.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prepare = subs.add_parser('prepare')
    prepare.add_argument('--root', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--train-limit', type=int, default=400)
    prepare.add_argument('--dev-limit', type=int, default=60)
    prepare.add_argument('--holdout-limit', type=int, default=40)
    prepare.add_argument('--spatial-buffer-tiles', type=int, default=1)
    prepare.add_argument('--crop-size', type=int, default=256)
    prepare.add_argument('--seed', type=int, default=0)
    for name in ('stage0', 'train', 'evaluate', 'candidates', 'distill', 'grpo', 'confirm', 'scan'):
        p = subs.add_parser(name)
        p.add_argument('--manifest', required=(name != 'stage0'))
        p.add_argument('--output', required=True)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--device', default='cpu')
        p.add_argument('--threads', type=int, default=2)
        p.add_argument('--limit', type=int, default=1000000)
        p.add_argument('--query-count', type=int, choices=(24, 48, 96), default=48)
        if name == 'stage0':
            continue
        p.add_argument('--checkpoint', required=(name not in ('train', 'confirm')))
        if name in ('train', 'distill', 'grpo'):
            p.add_argument('--steps', type=int, required=True)
            p.add_argument('--lr', type=float, default=1e-5 if name == 'grpo' else 1e-4)
            p.add_argument('--eval-every', type=int, default=50)
            p.add_argument('--apls-every', type=int, default=150)
            p.add_argument('--batch-size', type=int, default=4)
        if name == 'train':
            p.add_argument('--discriminative', action='store_true')
            p.add_argument('--no-pretrained', action='store_true', help='Offline smoke tests only; not the default research protocol')
        if name == 'distill':
            p.add_argument('--candidate-cache', required=True)
            p.add_argument('--selection', choices=('reward', 'random'), default='reward')
        if name != 'stage0':
            p.add_argument('--sample-steps', type=int, default=16)
            p.add_argument('--noise', type=float, default=0.0 if name != 'grpo' else 0.2)
            p.add_argument('--apls-adapter', help='module:function; independent official scorer')
        if name in ('candidates', 'grpo'):
            p.add_argument('--group-size', type=int, default=16 if name == 'candidates' else 4)
        p.add_argument('--delta-dice', type=float, default=.005)
        p.add_argument('--delta-bf1', type=float, default=.01)
        if name == 'candidates':
            p.add_argument('--split', choices=('train', 'dev'), default='dev')
            p.add_argument('--reference-samples', type=int, default=3)
        if name == 'evaluate':
            p.add_argument('--split', choices=('dev', 'holdout'), default='dev')
            p.add_argument('--confirm-holdout', action='store_true')
        if name == 'confirm':
            p.add_argument('--models', nargs='+', required=True, help='B0=path B1=path B3=path GRPO=path')
            p.add_argument('--inference-seeds', nargs=3, type=int, default=[0, 1, 2])
            p.add_argument('--confirm-holdout', action='store_true')
        if name == 'scan':
            p.add_argument('--noise-levels', nargs=3, type=float, default=[.1, .2, .3])
        if name == 'grpo':
            p.add_argument('--beta', type=float, default=.01)
    args = parser.parse_args()
    if args.command == 'prepare':
        from .data import prepare as prepare_data
        prepare_data(args.root, args.output, crop_size=args.crop_size, train_limit=args.train_limit,
                     dev_limit=args.dev_limit, holdout_limit=args.holdout_limit,
                     spatial_buffer_tiles=args.spatial_buffer_tiles, seed=args.seed)
        return
    if args.limit < 1 or args.threads < 1:
        parser.error('--limit and --threads must be positive')
    if args.command in ('train', 'distill', 'grpo') and min(args.eval_every, args.apls_every, args.batch_size) < 1:
        parser.error('--eval-every, --apls-every and --batch-size must be positive')
    if args.command == 'candidates' and args.reference_samples < 1:
        parser.error('--reference-samples must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
