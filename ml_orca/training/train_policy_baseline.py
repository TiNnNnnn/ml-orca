#!/usr/bin/env python3
"""Shared CBO graph ablations across catalogs under one audited execution configuration."""

from ml_orca.objectives.policy import objective_channels, selection_metrics, policy_target
from ml_orca.data.admission import history_training_gate, assigned_history_queries
from ml_orca.encoding.rule_history_encoding import reuse_history

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
from statistics import median
import time

from ml_orca.data.export_policy_learning_samples import export_comparison, policy_runtime_settings
from ml_orca.common.artifacts import read_snapshot
from ml_orca.common.paths import package_sources
from ml_orca.data.export_history_corpus import iter_history_runs
from ml_orca.encoding.rule_policy_encoding import input_sequences, encode_sequence, fit_vocabulary
from ml_orca.common.artifacts import artifact_snapshot


def baseline_exclusions(record):
    errors = list(record['admission']['feature_exclusions'])
    if not record['admission']['feature_integrity_verified']:
        errors.append('input_integrity_unverified')
    if record['inputs']['stats_experiment_document'] is not None:
        errors.append('intervention_channel_not_encoded')
    if any(r.get('placement') != 'cbo' for r in record['inputs']['candidate_policy'] or []):
        errors.append('outside_fixed_cbo_action_domain')
    if record['response']['status'] != 'complete':
        errors.extend(record['response']['exclusions'])
        errors.append('incomplete_response')
    for key in ('planning_ms_median', 'execution_ms_median'):
        value = record['response'][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            errors.append('invalid_response_' + key)
    return sorted(set(errors))


def measurement_environment(context, comparison, arm):
    """Catalog contents are encoded per query; unencoded execution settings must agree.

    Names, OIDs, capture timestamps and schema sizes are not environment IDs.
    Policy paths are replaced only because their native resolved contents are
    already model inputs; every other generated runtime statement is retained.
    """
    artifacts = comparison['artifact_provenance']['before_server_start']
    binaries = {}
    for key in ('postgres', 'pg_orca', 'rule_audit', 'rules', 'runner'):
        snapshot = artifacts[key]
        if (type(snapshot.get('size')) is not int or snapshot['size'] < 1
                or not re.fullmatch(r'[0-9a-f]{8}', snapshot.get('crc32', ''))):
            raise ValueError('invalid measurement environment artifact: ' + key)
        binaries[key] = {k: snapshot[k] for k in ('size', 'crc32')}
    settings = context['catalog']['settings']
    runtime = context['settings_sql'][arm]
    if not isinstance(settings, dict) or not settings or not isinstance(runtime, str) or not runtime:
        raise ValueError('missing captured measurement settings')
    runtime = policy_runtime_settings(runtime)
    return {'artifacts': binaries, 'catalog_settings': settings, 'runtime_settings_sql': runtime,
            'scope': 'captured_software_and_settings_only_not_hardware_or_load_equivalence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='pre-collection family assignments')
    parser.add_argument('--run', type=Path, required=True, help='comparison collection with a common audited execution configuration')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--objective', choices=('planning', 'plan_execution'), default='plan_execution',
                        help='planning excludes execution time from loss, checkpoint and policy selection')
    parser.add_argument('--seed', type=int, default=929)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--graph-mode', choices=('none', 'static', 'self'), default='static')
    parser.add_argument('--rule-encoding', choices=('tree', 'sequence'), default='tree',
                        help='tree is the primary model; sequence reproduces the earlier ablation only')
    parser.add_argument('--message-rounds', type=int, help='tree default 3; legacy sequence supports 1')
    parser.add_argument('--static-graph', type=Path, help='native template-only graph export; also aligns vocabulary in K=0')
    parser.add_argument('--history', type=Path, help='audited history frozen before collection; training query population only')
    parser.add_argument('--history-mode', choices=('none', 'context', 'graph'), default='graph',
                        help='with --history, same vocabulary/initialization; none masks history, context masks dynamic edges')
    parser.add_argument('--history-pooling', choices=('mean', 'mean_variance'), default='mean',
                        help='attempt-weighted encoded context moments; mean preserves the original baseline')
    parser.add_argument('--evaluate-test', action='store_true', help='only after freezing the model; default evaluates development splits only')
    args = parser.parse_args()
    targets = objective_channels(args.objective)
    if args.epochs < 1 or args.width < 1:
        parser.error('positive fixed epoch budget and hidden width required')
    if args.graph_mode != 'none' and args.static_graph is None:
        parser.error('graph ablations require --static-graph')
    if args.rule_encoding == 'tree' and args.static_graph is None:
        parser.error('tree encoding requires a verified --static-graph for endpoint binding')
    if args.history is not None and args.rule_encoding != 'tree':
        parser.error('historical actual trees require --rule-encoding tree')
    if args.history_pooling != 'mean' and (args.history is None or args.history_mode == 'none'):
        parser.error('context pooling requires enabled --history')
    if args.message_rounds is None:
        args.message_rounds = 3 if args.rule_encoding == 'tree' else 1
    if args.message_rounds < 1 or (args.rule_encoding == 'sequence' and args.message_rounds != 1):
        parser.error('positive message rounds required; legacy sequence ablation supports one round')
    manifest = json.loads(args.manifest.read_bytes())
    families, units, records, examples = {}, set(), [], []
    catalogs, environments, assets = set(), set(), {'manifest': args.manifest, 'trainer': Path(__file__)}
    history = prepared_history = None
    history_rule_orders = {}
    if args.history is not None:
        from ml_orca.encoding.rule_history_encoding import attach_history
        assets['history'] = args.history
        history_snapshot = artifact_snapshot({'history': args.history})['history']
        history = json.loads(read_snapshot(history_snapshot))
        for i, run in enumerate(history['runs']):
            assets['history_source:' + str(i)] = Path(run['source']['path'])
            for key in ('catalog_snapshot', 'graph_snapshot'):
                assets['history_' + key + ':' + str(i)] = Path(run['inputs'][key]['path'])
            snapshot = run['inputs']['graph_snapshot']
            if snapshot['path'] not in history_rule_orders:
                history_rule_orders[snapshot['path']] = [n['rule_hash'] for n in
                    json.loads(read_snapshot(snapshot))['nodes']]
    allowed_history = assigned_history_queries(manifest)
    if 'history_population' in manifest:
        assets['history_population'] = Path(manifest['history_population']['path'])
    history_gate = history_training_gate(history, allowed_history,
                                        manifest.get('minimum_training_query_graphs', 0))
    static_snapshot = None
    if args.static_graph is not None:
        assets['static_graph'] = args.static_graph
        static_snapshot = artifact_snapshot({'static_graph': args.static_graph})['static_graph']
    for assignment in manifest['queries']:
        family, split = assignment['family'], assignment['split']
        if split not in ('train', 'validation', 'test') or families.get(family, split) != split:
            raise ValueError('invalid or leaking family partition')
        families[family] = split
        workload, query = assignment['case_id'].split(':')
        if (workload, query) in units:
            raise ValueError('duplicate assigned query')
        units.add((workload, query))
        path = args.run / workload / query / 'comparison.json'
        assets['comparison:' + assignment['case_id']] = path
        comparison = json.loads(path.read_bytes())
        for record in export_comparison(path):
            if record['inputs']['query_sql'] != assignment['query']:
                raise ValueError('query changed since split assignment; require a new revision')
            record['pilot'] = {'split': split, 'family': family, 'exclusions': baseline_exclusions(record)}
            records.append(record)
            snapshot = record['inputs']['catalog_snapshot'] or {}
            catalogs.add((snapshot.get('size'), snapshot.get('crc32')))
            if record['pilot']['exclusions']:
                continue
            context = json.loads(read_snapshot(record['inputs']['catalog_snapshot']))
            environment = measurement_environment(context, comparison, record['unit']['policy'])
            environments.add(json.dumps(environment, sort_keys=True))
            feature = input_sequences(record['inputs'], static_snapshot, tree_rules=args.rule_encoding == 'tree')
            if not feature['query_binding']['complete']:
                record['pilot']['exclusions'].append('query_binding_unavailable')
                continue
            if history is not None:
                # Identity is index-validation metadata, never an embedding token.
                feature['rule_hashes'] = [n['rule_hash'] for n in
                    json.loads(read_snapshot(record['inputs']['graph_snapshot']))['nodes']]
                catalog = json.loads(read_snapshot(record['inputs']['catalog_snapshot']))
                timestamp = catalog['catalog']['captured_at']
                if prepared_history is None:
                    if any(order != feature['rule_hashes'] for order in history_rule_orders.values()):
                        raise ValueError('historical rule identity order differs from prediction graph')
                    attach_history(feature, {**history, 'runs': iter_history_runs(history)},
                                   static_snapshot, timestamp, allowed_history)
                    prepared_history = feature
                else:
                    reuse_history(feature, prepared_history, timestamp)
            # Legacy/static controls share native vocabulary. Runtime evidence
            # enters only through the explicit time/population-audited history.
            rule_channels = ('rule_node', 'rule_symbol', 'rule_constraint') if args.rule_encoding == 'tree' else ('rule',)
            feature['sequences'] = {key: value for key, value in feature['sequences'].items()
                                    if key in (*rule_channels, 'policy', 'query', 'relation')
                                    or (key == 'edge' and static_snapshot is not None)
                                    or (history is not None and key.startswith('history_'))}
            examples.append((record, feature))
    if len(environments) != 1:
        raise ValueError('unencoded software/runtime settings differ; cannot pool these measurements')
    required_splits = ('train', 'validation', 'test') if args.evaluate_test else ('train', 'validation')
    if not all(any(r['pilot']['split'] == split for r, _ in examples) for split in required_splits):
        raise ValueError('not enough usable split coverage; do not reassign failed units')
    history_sequences = {} if prepared_history is None else {
        k: v for k, v in prepared_history['sequences'].items() if k.startswith('history_')}
    vocabulary = fit_vocabulary([
        {k: v for k, v in f['sequences'].items() if not k.startswith('history_')}
        for r, f in examples if r['pilot']['split'] == 'train'] + [history_sequences])
    encoded_history = {key: [encode_sequence(s, vocabulary) for s in value]
                       for key, value in history_sequences.items()}
    for record, feature in examples:
        feature['sequences'] = {key: encoded_history[key] if key in encoded_history else
                               [encode_sequence(s, vocabulary) for s in value]
                                for key, value in feature['sequences'].items()}
    if history is not None:
        # One admission pass prepares shared data; verify every source again at the
        # fit boundary so sharing cannot hide an intervening source-file change.
        for run in history['runs']:
            for snapshot in (run['source'], run['inputs']['catalog_snapshot'],
                             run['inputs']['graph_snapshot']):
                read_snapshot(snapshot)
    if 'history_population' in manifest:
        read_snapshot(manifest['history_population'])
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'samples.jsonl').open('x') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    with (args.output / 'vocabulary.json').open('x') as stream:
        json.dump(vocabulary, stream)
    assets.update(package_sources())
    before_fit = artifact_snapshot(assets)
    if static_snapshot is not None and before_fit['static_graph'] != static_snapshot:
        raise ValueError('static graph changed during input preparation')
    if history is not None and before_fit['history'] != history_snapshot:
        raise ValueError('history changed during input preparation')
    import torch
    from ml_orca.models.rule_policy_model import WholePolicyPredictor
    from ml_orca.models.rule_tree_model import TreePolicyPredictor
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = (TreePolicyPredictor(len(vocabulary), args.width, args.graph_mode, args.message_rounds,
                                history=history is not None, history_mode=args.history_mode,
                                history_pooling=args.history_pooling)
             if args.rule_encoding == 'tree' else WholePolicyPredictor(len(vocabulary), args.width, args.graph_mode))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    train = [(r, f) for r, f in examples if r['pilot']['split'] == 'train']
    validation = [(r, f) for r, f in examples if r['pilot']['split'] == 'validation']

    target = policy_target

    losses, curve = [], []
    best_validation, best_epoch = math.inf, None
    fit_wall, fit_cpu = time.perf_counter(), time.process_time()
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        total = 0.
        for index in order:
            record, feature = train[index]
            optimizer.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(model(feature)[:len(targets)],
                                                     torch.tensor(target(record)[:len(targets)]))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()
        losses.append(total / len(train))
        model.eval()
        with torch.inference_mode():
            validation_loss = sum(torch.nn.functional.smooth_l1_loss(
                model(feature)[:len(targets)], torch.tensor(target(record)[:len(targets)])).item()
                for record, feature in validation) / len(validation)
        if not math.isfinite(validation_loss):
            raise ValueError('nonfinite validation loss')
        improved = validation_loss < best_validation
        if improved:
            best_validation, best_epoch = validation_loss, epoch + 1
            torch.save(model.state_dict(), args.output / 'best-validation.pt')
        row = {'epoch': epoch + 1, 'training_online_smooth_l1': losses[-1],
               'validation_smooth_l1': validation_loss, 'checkpoint_improved': improved,
               'fit_wall_seconds': time.perf_counter() - fit_wall}
        curve.append(row)
        with (args.output / 'learning-curve.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)
    # Test responses never participate in epoch selection. Preserve the final
    # epoch too so continued fitting is not confused with the chosen checkpoint.
    torch.save(model.state_dict(), args.output / 'last-epoch.pt')
    model.load_state_dict(torch.load(args.output / 'best-validation.pt', weights_only=True))
    fit_seconds = {'wall': time.perf_counter() - fit_wall, 'cpu': time.process_time() - fit_cpu}
    constants = {}
    for policy in {r['unit']['policy'] for r, _ in train}:
        labels = [target(r) for r, _ in train if r['unit']['policy'] == policy]
        constants[policy] = [sum(y[j] for y in labels) / len(labels) for j in range(2)]
    predictions, metrics, inference_seconds = [], {}, []
    model.eval()
    with torch.inference_mode():
        for record, feature in examples:
            if record['pilot']['split'] == 'test' and not args.evaluate_test:
                continue
            start = time.perf_counter()
            predicted = model(feature).tolist()
            inference_seconds.append(time.perf_counter() - start)
            if not all(math.isfinite(v) for v in predicted):
                raise ValueError('nonfinite prediction')
            predictions.append({'unit': record['unit'], 'split': record['pilot']['split'],
                'observed_log1p_ms': target(record), 'predicted_log1p_ms': predicted,
                'training_policy_constant': constants[record['unit']['policy']],
                'unknown_tokens': sum(s['unknown_tokens'] for seqs in feature['sequences'].values() for s in seqs)})
    for split in ('train', 'validation', 'test'):
        subset = [p for p in predictions if p['split'] == split]
        if not subset:
            metrics[split] = {'evaluated': False}
            continue
        metrics[split] = {'records': len(subset), 'unknown_tokens': sum(p['unknown_tokens'] for p in subset)}
        for key in ('predicted_log1p_ms', 'training_policy_constant'):
            metrics[split][key + '_mae'] = [sum(abs(p[key][j] - p['observed_log1p_ms'][j])
                                                 for p in subset) / len(subset) for j in range(len(targets))]
    after_fit = artifact_snapshot(assets)
    if after_fit != before_fit:
        raise ValueError('input or code changed during model fitting')
    report = {'scope': 'fixed_cbo_graph_ablation_completed_response_pilot_not_generalization_or_dro_certificate',
        'model_trained': True, 'epochs': args.epochs, 'width': args.width, 'seed': args.seed,
        'objective': args.objective, 'trained_response_channels': targets,
        'input_scope': 'prospective_candidate_policy_and_admitted_prior_history',
        'target_transform': 'log1p_ms', 'loss': 'smooth_l1',
        'catalog_snapshots': len(catalogs), 'measurement_environment': json.loads(next(iter(environments))),
        'architecture': ('rooted_tree_v1' if args.rule_encoding == 'tree' else 'mean_count_sequence_v3'),
        'graph_mode': args.graph_mode, 'rule_encoding': args.rule_encoding,
        'message_rounds': args.message_rounds if args.graph_mode != 'none' else 0,
        'history_enabled': history is not None,
        'history_mode': args.history_mode if history is not None else 'none',
        'history_input_storage': 'shared_read_only' if history is not None else 'none',
        'learned_history_representation_cached': False,
        'history_training_gate': history_gate,
        'history_pooling': args.history_pooling if history is not None and args.history_mode != 'none' else 'none',
        'history_admission': None if history is None else {'frozen_at_utc': history['frozen_at_utc'],
            'scope': 'training_queries_only_before_collection',
            'runs': [r['denominators'] for r in history['runs']]},
        'parameter_count': sum(p.numel() for p in model.parameters()), 'fit_seconds': fit_seconds,
        'inference_wall_seconds': inference_seconds, 'inference_includes_input_preparation': False,
        'test_evaluated': args.evaluate_test,
        'validation_used_for_model_selection': True, 'selected_epoch': best_epoch,
        'checkpoint_objective': 'validation_mean_smooth_l1_log1p_' + args.objective,
        'learning_curve': curve, 'records': len(records),
        'exclusions': dict(Counter(e for r in records for e in r['pilot']['exclusions'])),
        'response_statuses': dict(Counter(r['response']['status'] for r in records)),
        'global_training_admission_overridden': False, 'training_losses': losses,
        'metrics': metrics, 'predictions': predictions, 'input_snapshots': before_fit,
        'selection': {split: selection_metrics(predictions, records, split, args.objective)
                      for split in ('train', 'validation', 'test')
                      if split != 'test' or args.evaluate_test},
        'fit_artifact_endpoints_equal': after_fit == before_fit,
        'not_claimed': ['template_or_application_independence', 'rbo_or_intervention_generalization',
                        'policy_recommendation_safety', 'gnn_incremental_value', 'causal_rule_effect']}
    with (args.output / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    torch.save(model.state_dict(), args.output / 'model.pt')
    print(json.dumps({'records': len(records), 'metrics': metrics, 'exclusions': report['exclusions']}))


if __name__ == '__main__':
    main()
