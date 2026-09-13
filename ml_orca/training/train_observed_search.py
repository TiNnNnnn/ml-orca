#!/usr/bin/env python3
"""Train from existing traces only: retrospective search-work prediction, NOT policy value."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
import zlib

import torch

from ml_orca.common.artifacts import read_snapshot, relocate_artifacts
from ml_orca.common.paths import package_sources
from ml_orca.common.artifacts import artifact_snapshot


from ml_orca.objectives.observed_search import TARGETS, observed_target, validate_frozen_target
from ml_orca.encoding.observed_search import observed_features, ObservedFeatureBuilder
from ml_orca.models.rule_tree_model import ObservedSearchPredictor
from ml_orca.training.runtime import training_device, synchronize, runtime_metadata
from ml_orca.training.inputs import ObservedInputDataset, input_loader


def internal_split(family, seed):
    # New internal development split within the old TRAIN population only.
    # Existing validation/test SQL is neither loaded nor reassigned.
    return 'validation' if zlib.crc32(f'{seed}:{family}'.encode()) % 5 == 0 else 'train'


def training_contract(width, mode):
    return {'network': {'family': 'ordered_tree_rooted_gnn', 'width': width,
                        'message_rounds': 3, 'graph_mode': 'static', 'context_mode': mode},
            'objective': {'name': 'observed_search_work', 'targets': list(TARGETS),
                          'input_scope': 'retrospective_current_trace',
                          'transform': 'log1p_ms', 'loss': 'smooth_l1'}}


def validate_resume_contract(prior, width, mode):
    # Legacy checkpoints still have scope/target/mode/input checks in main and
    # strict tensor-shape checks in restore_checkpoint. New runs add an explicit
    # network/objective identity without rewriting historical manifests.
    contract = prior.get('training_contract')
    if contract is not None and contract != training_contract(width, mode):
        raise ValueError('resume network or objective changed')
    if prior.get('width', width) != width:
        raise ValueError('resume network width changed')


def restore_checkpoint(path, model, optimizer, train_count):
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    steps = checkpoint['steps']
    if (type(steps) is not int or steps < 1 or train_count < 1
            or checkpoint['epoch'] != (steps - 1) // train_count + 1):
        raise ValueError('invalid checkpoint training position')
    # A boundary checkpoint precedes validation. Re-enter that epoch with an
    # empty training suffix so validation is not silently skipped on resume.
    epoch, offset = divmod(steps - 1, train_count)
    offset += 1
    loss_sum, loss_count = checkpoint.get('epoch_loss_sum', 0.), checkpoint.get('epoch_loss_count', 0)
    if (type(loss_count) is not int or not 0 <= loss_count <= offset
            or not math.isfinite(loss_sum) or loss_sum < 0):
        raise ValueError('invalid checkpoint loss denominator')
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    device = next(model.parameters()).device
    if device.type == 'cuda':
        if 'cuda_rng_state' in checkpoint:
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'], device)
        for group in optimizer.param_groups:
            if group.get('foreach') is None:
                group['foreach'] = False  # Same per-parameter Adam path as the CPU run.
    return steps, epoch, offset, loss_sum, loss_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=929)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu', help='cpu or cuda[:index]; no implicit fallback')
    parser.add_argument('--input-workers', type=int, default=0, help='CPU preparation processes; ordered, one query per Adam step')
    parser.add_argument('--input-cache-mb', type=int, default=0, help='total retained Python feature budget, divided among workers')
    parser.add_argument('--input-prefetch', type=int, default=2, help='in-flight queries per worker, outside retained-cache budget')
    parser.add_argument('--artifact-root', nargs=2, metavar=('SOURCE', 'DESTINATION'),
                        help='absolute repository roots; retain original content checks and identities')
    parser.add_argument('--resume', type=Path, help='checkpoint beside its frozen manifest and vocabulary')
    parser.add_argument('--mode', choices=('graph', 'context', 'none'), default='graph')
    args = parser.parse_args()
    if args.epochs < 1 or args.width < 1 or args.threads < 1:
        parser.error('positive epochs, width and threads required')
    if args.input_workers < 0 or args.input_cache_mb < 0 or args.input_prefetch < 1:
        parser.error('nonnegative input workers/cache and positive prefetch required')
    with relocate_artifacts(args.artifact_root):
        return train(args)


def train(args):
    device = training_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    snapshots = artifact_snapshot({'manifest': args.corpus / 'manifest.json', 'admission': args.admission,
        'graph': args.corpus / 'audit/rule_graph.json', 'trainer': Path(__file__),
        **package_sources()})
    manifest = json.loads(read_snapshot(snapshots['manifest']))
    admission = json.loads(read_snapshot(snapshots['admission']))
    if manifest.get('history_split') != 'train':
        raise ValueError('original external holdout must not enter this pilot')
    cases = {c['case_id']: c for d in manifest['datasets'] for c in d['cases']}
    if (admission.get('scope') != 'query_graph_admission_not_execution_labels_or_model_benefit'
            or {r['case_id'] for r in admission['queries']} != cases.keys()
            or len(admission['queries']) != len(cases)):
        raise ValueError('audit population mismatch')
    native = json.loads(read_snapshot(snapshots['graph']))
    indices = {n['rule_hash']: i for i, n in enumerate(native['nodes'])}
    items, excluded, tokens = [], [], set()
    feature = ObservedFeatureBuilder(snapshots, indices)

    prior = None
    if args.resume:
        snapshots.update(artifact_snapshot({'resume_checkpoint': args.resume,
            'resume_manifest': args.resume.parent / 'manifest.json',
            'resume_vocabulary': args.resume.parent / 'vocabulary.json'}))
        prior = json.loads(read_snapshot(snapshots['resume_manifest']))
        validate_resume_contract(prior, args.width, args.mode)
        read_snapshot(snapshots['resume_checkpoint'])
        if (prior.get('scope') != 'retrospective_observed_search_work_not_policy_value_or_untraced_latency'
                or prior['targets'] != list(TARGETS) or prior['seed'] != args.seed or prior['mode'] != args.mode
                or any(prior['input_snapshots'][k] != snapshots[k] for k in ('manifest', 'admission', 'graph'))):
            raise ValueError('resume must preserve frozen population, objective, vocabulary and rule graph')
        for key, snapshot in prior['input_snapshots'].items():
            if key.startswith('catalog:'):
                read_snapshot(snapshot)
                snapshots[key] = snapshot
        items, excluded = prior['queries'], prior['excluded']
        identities = [i['case']['case_id'] for i in items] + [e['case_id'] for e in excluded]
        if len(identities) != len(cases) or set(identities) != cases.keys():
            raise ValueError('resume population denominator mismatch')
        for item in items:
            if (item['case'] != cases[item['case']['case_id']]
                    or item['split'] != internal_split(item['case']['family'], args.seed)):
                raise ValueError('resume query or split changed')
            validate_frozen_target(json.loads(read_snapshot(item['summary_snapshot'])), item['target_log1p_ms'])
        vocabulary = json.loads(read_snapshot(snapshots['resume_vocabulary']))
        expected_vocabulary = {t: i for i, t in enumerate(['<pad>', '<unknown>',
            *sorted(t for t in vocabulary if t not in ('<pad>', '<unknown>'))])}
        if vocabulary != expected_vocabulary:
            raise ValueError('resume vocabulary token identities changed')
        print(json.dumps({'phase': 'resumed_existing_index', 'queries': len(items)}), flush=True)
    else:
        for row in admission['queries']:
            case = cases[row['case_id']]
            if not row['eligible']:
                excluded.append({'case_id': row['case_id'], 'reasons': row['exclusions']})
                continue
            app, stem = case['case_id'].split(':')
            summary_path = args.corpus / app / (stem + '.summary.json')
            summary_snapshot = artifact_snapshot({'summary': summary_path})['summary']
            summary = json.loads(read_snapshot(summary_snapshot))
            try:
                target = observed_target(summary)
            except ValueError as error:
                excluded.append({'case_id': row['case_id'], 'reasons': [str(error)]})
                continue
            item = {'case': case, 'split': internal_split(case['family'], args.seed),
                    'target_log1p_ms': target, 'summary_snapshot': summary_snapshot,
                    'graph_snapshot': row['graph_snapshot'], 'attempts': summary['attempts']}
            current = feature(item)
            if item['split'] == 'train':
                tokens.update(token for seqs in current['sequences'].values() for seq in seqs for token, _ in seq)
            items.append(item)
            if len(items) % 100 == 0:
                print(json.dumps({'phase': 'encoding_existing_files', 'queries': len(items),
                                  'seconds': time.perf_counter() - started}), flush=True)
        vocabulary = {t: i for i, t in enumerate(['<pad>', '<unknown>', *sorted(tokens)])}
    counts = Counter(i['split'] for i in items)
    if counts['train'] < 2000 or not counts['validation']:
        raise ValueError(f'insufficient existing labelled graphs: {dict(counts)}')
    if '<pad>' in tokens or '<unknown>' in tokens:
        raise ValueError('reserved input token')
    with (args.output / 'vocabulary.json').open('x') as stream:
        json.dump(vocabulary, stream)
    receipt = {'scope': 'retrospective_observed_search_work_not_policy_value_or_untraced_latency',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'targets': TARGETS,
        'epochs': args.epochs, 'mode': args.mode, 'seed': args.seed, 'counts': dict(counts),
        'threads': args.threads, 'width': args.width,
        'runtime': runtime_metadata(device), 'artifact_relocation': args.artifact_root,
        'input_pipeline': {'workers': args.input_workers, 'cache_mb_total': args.input_cache_mb,
                           'prefetch_per_worker': args.input_prefetch, 'ordered': True,
                           'learned_cache': False, 'graph_content_check_on_every_access': True},
        'training_contract': training_contract(args.width, args.mode),
        'queries': items, 'excluded': excluded, 'input_snapshots': snapshots,
        'split': 'family_disjoint_internal_80_20_within_original_train_only',
        'new_sql_executions': 0, 'original_external_holdouts_used': False}
    with (args.output / 'manifest.json').open('x') as stream:
        json.dump(receipt, stream, allow_nan=False)
    model = ObservedSearchPredictor(len(vocabulary), args.width, 'static', 3, history=True,
                                    history_mode=args.mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, foreach=False)
    train = [i for i in items if i['split'] == 'train']
    validation = [i for i in items if i['split'] == 'validation']
    constant = torch.tensor([sum(i['target_log1p_ms'][j] for i in train) / len(train) for j in range(2)], device=device)
    best, best_state, steps = math.inf, None, 0
    start_epoch, resume_offset, resume_loss_sum, resume_loss_count = 0, 0, 0., 0
    if args.resume:
        steps, start_epoch, resume_offset, resume_loss_sum, resume_loss_count = restore_checkpoint(
            args.resume, model, optimizer, len(train))
        if start_epoch >= args.epochs:
            raise ValueError('checkpoint already completed requested epochs')
        saved = torch.load(args.resume, weights_only=True, map_location='cpu')
        best, best_state = saved.get('best_validation_loss', math.inf), saved.get('best_validation_model')
        if (start_epoch > 0 and 'best_validation_loss' not in saved
                or best_state is None and best != math.inf
                or best_state is not None and not math.isfinite(best)):
            raise ValueError('checkpoint missing consistent validation selection state')
        if best_state is not None:
            torch.save(best_state, args.output / 'best-validation.pt')
        del saved

    ordered_items = train + validation
    dataset = ObservedInputDataset(ordered_items, feature, vocabulary,
        args.input_cache_mb * 1024 * 1024 // max(1, args.input_workers), args.artifact_root)
    input_order = []
    loader = input_loader(dataset, input_order, args.input_workers, args.input_prefetch)

    print(json.dumps({'phase': 'training', 'counts': dict(counts), 'targets': TARGETS}), flush=True)
    def log_progress(event):
        event['seconds'] = time.perf_counter() - started
        with (args.output / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        offset = resume_offset if epoch == start_epoch else 0
        loss_sum = resume_loss_sum if epoch == start_epoch else 0.
        loss_count = resume_loss_count if epoch == start_epoch else 0
        input_order[:] = order[offset:]
        prepared = iter(loader)
        for expected_index in input_order:
            item = ordered_items[expected_index]
            begin = time.perf_counter()
            event = {'epoch': epoch + 1, 'steps': steps, 'case_id': item['case']['case_id']}
            log_progress(event | {'phase': 'input'})
            index, data, preparation = next(prepared)
            if index != expected_index:
                raise ValueError('input prefetch changed query order')
            loaded = time.perf_counter()
            log_progress(event | {'phase': 'forward', 'input_seconds': loaded - begin,
                'trees': len(data['history_trees']), 'contexts': len(data['history_contexts']),
                'edges': len(data['history_edges'])})
            optimizer.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(model(data), torch.tensor(item['target_log1p_ms'], device=device))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            forwarded = time.perf_counter()
            log_progress(event | {'phase': 'backward', 'forward_seconds': forwarded - loaded})
            loss.backward()
            synchronize(device)
            backwarded = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            steps += 1
            loss_sum += loss.item()
            loss_count += 1
            log_progress(event | {'phase': 'parameter_update', 'steps': steps, 'loss': loss.item(),
                'input_seconds': loaded - begin, 'forward_seconds': forwarded - loaded,
                'backward_seconds': backwarded - forwarded,
                'update_seconds': time.perf_counter() - backwarded,
                'input_preparation': preparation})
            del data, loss
            if steps == 1 or steps % 10 == 0 or steps % len(train) == 0:
                temporary = args.output / 'latest.pt.tmp'
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'epoch': epoch + 1, 'steps': steps, 'epoch_loss_sum': loss_sum,
                            'epoch_loss_count': loss_count, 'rng_state': torch.get_rng_state(),
                            **({'cuda_rng_state': torch.cuda.get_rng_state(device)} if device.type == 'cuda' else {}),
                            'best_validation_loss': best, 'best_validation_model': best_state}, temporary)
                temporary.replace(args.output / 'latest.pt')
        del prepared
        model.eval()
        input_order[:] = range(len(train), len(ordered_items))
        prepared = iter(loader)
        errors, baseline, val_loss = torch.zeros(2, device=device), torch.zeros(2, device=device), 0.
        with torch.inference_mode():
            for expected_index in input_order:
                item = ordered_items[expected_index]
                log_progress({'phase': 'validation', 'epoch': epoch + 1, 'steps': steps,
                              'case_id': item['case']['case_id']})
                expected = torch.tensor(item['target_log1p_ms'], device=device)
                index, data, _ = next(prepared)
                if index != expected_index:
                    raise ValueError('validation prefetch changed query order')
                predicted = model(data)
                val_loss += torch.nn.functional.smooth_l1_loss(predicted, expected).item()
                errors += (predicted - expected).abs()
                baseline += (constant - expected).abs()
        del prepared, data
        val_loss /= len(validation)
        if not math.isfinite(val_loss):
            raise ValueError('nonfinite validation loss')
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, args.output / 'best-validation.pt')
        row = {'epoch': epoch + 1, 'steps': steps,
               'training_loss': loss_sum / len(train) if loss_count == len(train) else None,
               'observed_training_loss': loss_sum / loss_count if loss_count else None,
               'observed_training_count': loss_count,
               'validation_loss': val_loss, 'validation_log1p_mae': (errors / len(validation)).tolist(),
               'constant_log1p_mae': (baseline / len(validation)).tolist(),
               'seconds': time.perf_counter() - started}
        with (args.output / 'learning-curve.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    del loader, dataset
    for snapshot in snapshots.values():
        read_snapshot(snapshot)
    torch.save(model.state_dict(), args.output / 'model.pt')
    print(json.dumps({'phase': 'complete', 'epochs': args.epochs, 'steps': steps}), flush=True)


if __name__ == '__main__':
    main()
