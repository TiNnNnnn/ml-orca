"""Fixed-checkpoint CPU/CUDA profiling on complete existing graphs; never executes SQL."""
import argparse
import cProfile
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pstats
from statistics import median
import time

from ml_orca.common.artifacts import read_snapshot, artifact_snapshot, relocate_artifacts
from ml_orca.encoding.observed_search import ObservedFeatureBuilder


def load_sample(manifest, vocabulary, case_id):
    item = next(i for i in manifest['queries'] if i['case']['case_id'] == case_id)
    snapshots = dict(manifest['input_snapshots'])
    native = json.loads(read_snapshot(snapshots['graph']))
    builder = ObservedFeatureBuilder(snapshots, {n['rule_hash']: i for i, n in enumerate(native['nodes'])})
    features = builder(item)
    from ml_orca.encoding.rule_policy_encoding import encode_sequence
    features['sequences'] = {k: [encode_sequence(s, vocabulary) for s in rows]
                            for k, rows in features['sequences'].items()}
    return features, item['target_log1p_ms']


def verify_cpu(model, checkpoint, features, labels):
    """Two sequential Adam steps, checking every parameter gradient and weight."""
    import torch
    if next(model.parameters()).device.type != 'cuda':
        raise ValueError('--verify-cpu requires a CUDA benchmark')
    cpu = deepcopy(model).cpu()
    models = (cpu, model)
    optimizers = [torch.optim.Adam(m.parameters(), lr=.003, foreach=False) for m in models]
    for m, opt in zip(models, optimizers):
        m.load_state_dict(checkpoint['model'], strict=True)
        opt.load_state_dict(deepcopy(checkpoint['optimizer']))
        for group in opt.param_groups:
            group['foreach'] = False
    tolerance = {'rtol': 2e-4, 'atol': 2e-6}
    report = {'tolerance': tolerance, 'steps': []}
    for _ in range(2):
        predictions = []
        for m, opt in zip(models, optimizers):
            opt.zero_grad()
            predicted = m(features)
            torch.nn.functional.smooth_l1_loss(predicted, predicted.new_tensor(labels)).backward()
            predictions.append(predicted.detach().cpu())
        torch.testing.assert_close(*predictions, **tolerance)
        grad_error = weight_error = 0.
        for (name, a), (other, b) in zip(cpu.named_parameters(), model.named_parameters()):
            if name != other or (a.grad is None) != (b.grad is None):
                raise AssertionError('gradient ownership changed: ' + name)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad.cpu(), **tolerance, msg=name)
                grad_error = max(grad_error, (a.grad - b.grad.cpu()).abs().max().item())
        for m, opt in zip(models, optimizers):
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5., error_if_nonfinite=True)
            opt.step()
        for (name, a), (_, b) in zip(cpu.named_parameters(), model.named_parameters()):
            torch.testing.assert_close(a, b.cpu(), **tolerance, msg=name)
            weight_error = max(weight_error, (a - b.cpu()).abs().max().item())
        report['steps'].append({'prediction_max_abs': (predictions[0] - predictions[1]).abs().max().item(),
                                'gradient_max_abs': grad_error, 'adam_weight_max_abs': weight_error})
    return report


def profile_inputs(args):
    """Measure fixed-input preparation/IPC, not GPU overlap or model quality."""
    from ml_orca.training.inputs import ObservedInputDataset, input_loader
    artifacts = artifact_snapshot({'manifest': args.manifest,
                                   'vocabulary': args.manifest.parent / 'vocabulary.json'})
    manifest = json.loads(read_snapshot(artifacts['manifest']))
    vocabulary = json.loads(read_snapshot(artifacts['vocabulary']))
    by_id = {i['case']['case_id']: i for i in manifest['queries']}
    items = [by_id[case] for case in args.case]
    snapshots = dict(manifest['input_snapshots'])
    native = json.loads(read_snapshot(snapshots['graph']))
    builder = ObservedFeatureBuilder(snapshots, {n['rule_hash']: i for i, n in enumerate(native['nodes'])})
    dataset = ObservedInputDataset(items, builder, vocabulary,
        args.input_cache_mb * 1024 * 1024 // max(1, args.input_workers), args.artifact_root)
    order = list(range(len(items)))
    loader = input_loader(dataset, order, args.input_workers, args.input_prefetch)
    report = {'scope': 'input_preparation_and_ipc_not_gpu_overlap_or_training_speedup',
              'workers': args.input_workers, 'cache_mb_total': args.input_cache_mb,
              'prefetch': args.input_prefetch, 'artifacts': artifacts, 'passes': []}
    for repeat in range(args.repeats + 1):
        start = time.perf_counter()
        prepared = iter(loader)
        rows = []
        for expected in order:
            before = time.perf_counter()
            index, features, stats = next(prepared)
            elapsed = time.perf_counter() - before
            if index != expected:
                raise ValueError('input order changed')
            rows.append({'case_id': items[index]['case']['case_id'], 'wait_seconds': elapsed,
                         'trees': len(features['history_trees']), 'contexts': len(features['history_contexts']),
                         'edges': len(features['history_edges']), **stats})
            del features
        result = {'pass': repeat, 'cold': repeat == 0,
                  'seconds': time.perf_counter() - start, 'cases': rows}
        report['passes'].append(result)
        print(json.dumps(result), flush=True)
        del prepared
    del loader, dataset
    for snapshot in artifacts.values():
        read_snapshot(snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--artifact-root', nargs=2, metavar=('SOURCE', 'DESTINATION'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--inputs-only', action='store_true', help='profile preparation/cache/IPC only; no model updates')
    parser.add_argument('--input-workers', type=int, default=0)
    parser.add_argument('--input-cache-mb', type=int, default=0)
    parser.add_argument('--input-prefetch', type=int, default=2)
    parser.add_argument('--profile', action='store_true', help='additional instrumented step, excluded from timings')
    parser.add_argument('--verify-cpu', action='store_true', help='additional CPU/CUDA gradient and Adam checks, excluded from timings')
    parser.add_argument('--reference-source', type=Path, help='trusted archived rule_tree_model.py to compare')
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error('positive threads and repeats required')
    if args.output.exists():
        parser.error('output already exists')
    if args.input_workers < 0 or args.input_cache_mb < 0 or args.input_prefetch < 1:
        parser.error('nonnegative workers/cache and positive prefetch required')
    if not args.inputs_only and args.checkpoint is None:
        parser.error('--checkpoint required unless --inputs-only')
    if args.inputs_only and (args.profile or args.verify_cpu or args.reference_source):
        parser.error('model diagnostics cannot be combined with --inputs-only')
    with relocate_artifacts(args.artifact_root):
        if args.inputs_only:
            return profile_inputs(args)
        return profile(args)


def profile(args):
    import torch
    from ml_orca.training.runtime import training_device, synchronize, runtime_metadata
    from ml_orca.models.rule_tree_model import ObservedSearchPredictor
    device = training_device(args.device)
    if args.reference_source:
        spec = importlib.util.spec_from_file_location('reference_tree_model', args.reference_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ObservedSearchPredictor = module.ObservedSearchPredictor
    torch.set_num_threads(args.threads)
    artifacts = artifact_snapshot({'manifest': args.manifest, 'checkpoint': args.checkpoint,
        'vocabulary': args.manifest.parent / 'vocabulary.json'})
    manifest = json.loads(read_snapshot(artifacts['manifest']))
    vocabulary = json.loads(read_snapshot(artifacts['vocabulary']))
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    model = ObservedSearchPredictor(len(vocabulary), manifest.get('width', 32), 'static', 3,
                                   history=True, history_mode=manifest['mode']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, foreach=False)
    report = {'scope': 'fixed_checkpoint_complete_graph_benchmark_not_training', 'threads': args.threads,
              'runtime': runtime_metadata(device), 'artifact_relocation': args.artifact_root,
              'checkpoint_step': checkpoint['steps'], 'artifacts': artifacts, 'cases': []}
    for case_id in args.case:
        start = time.perf_counter()
        features, labels = load_sample(manifest, vocabulary, case_id)
        target = torch.tensor(labels, device=device)
        row = {'case_id': case_id, 'load_seconds': time.perf_counter() - start,
               'rules': sum(features['loaded_rules']), 'trees': len(features['history_trees']),
               'nodes': sum(len(t['nodes']) for t in features['history_trees']),
               'contexts': len(features['history_contexts']), 'edges': len(features['history_edges']), 'steps': []}
        for repeat in range(-1, args.repeats + int(args.profile)):
            model.load_state_dict(checkpoint['model'], strict=True)
            # Adam's load_state_dict may share CPU tensors; clone before a benchmark update.
            optimizer.load_state_dict(deepcopy(checkpoint['optimizer']))
            for group in optimizer.param_groups:
                if group.get('foreach') is None:
                    group['foreach'] = False
            optimizer.zero_grad()
            profile = cProfile.Profile() if args.profile and repeat == args.repeats else None
            if profile:
                profile.enable()
            synchronize(device)
            start = time.perf_counter()
            prediction = model(features)
            loss = torch.nn.functional.smooth_l1_loss(prediction, target)
            synchronize(device)
            forward = time.perf_counter()
            loss.backward()
            synchronize(device)
            backward = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            end = time.perf_counter()
            if profile:
                profile.disable()
                stats = pstats.Stats(profile).stats
                row['profile'] = [{'file': file, 'line': line, 'function': name, 'calls': s[1],
                                   'self_seconds': s[2], 'inclusive_seconds': s[3]}
                                  for (file, line, name), s in sorted(stats.items(), key=lambda p: p[1][3], reverse=True)[:60]]
            elif repeat >= 0:
                row['steps'].append({'forward_seconds': forward - start, 'backward_seconds': backward - forward,
                                     'update_seconds': end - backward, 'seconds': end - start,
                                     'loss': loss.item(), 'prediction': prediction.detach().tolist()})
            del loss, prediction
        row['median_seconds'] = median(s['seconds'] for s in row['steps'])
        if args.verify_cpu:
            row['cpu_cuda_verification'] = verify_cpu(model, checkpoint, features, labels)
        report['cases'].append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ('steps', 'profile')}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
