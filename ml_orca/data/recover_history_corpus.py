#!/usr/bin/env python3
"""Finish offline decoding of a stopped collection; never execute SQL or train on timings."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import resource
import subprocess
import sys

from ml_orca.collect.profile_corpus_attempts import summarize_trace, trace_run
from ml_orca.encoding.rule_history_encoding import encode_observations
from ml_orca.collect.run_trace_corpus import orca_fallback_reason
from ml_orca.common.artifacts import artifact_snapshot
from ml_orca.common.paths import ML_ORCA_ROOT


def publish(path, document, compressed=False):
    # Exclusive temporary creation and hard-link publication never overwrite evidence.
    import os
    temporary = path.with_name(path.name + '.recovering')
    opener = gzip.open if compressed else Path.open
    with opener(temporary, 'xt') as stream:
        json.dump(document, stream, allow_nan=False)
    os.link(temporary, path)
    temporary.unlink()


def check_artifacts(manifest):
    expected = manifest['artifact_start']
    actual = artifact_snapshot({k: Path(v['path']) for k, v in expected.items()})
    if actual != expected:
        raise ValueError('collection artifacts changed: ' + str(
            [k for k in expected if expected[k] != actual[k]]))
    return actual


def decode(root, case, defer_failed=False):
    app, stem = case['case_id'].split(':')
    folder = root / app
    target = folder / f'{stem}.summary.json'
    if target.exists():
        raise ValueError('existing summary must not be overwritten')
    trace = folder / 'logs' / f'{stem}.log.gz'
    sources = {'trace': trace, 'context': folder / 'pre-workload-context.json',
               'manifest': root / 'manifest.json', 'status': folder / 'status.tsv'}
    before = artifact_snapshot(sources)
    if any('error' in v for v in before.values()):
        raise ValueError('missing captured inputs')
    statuses = dict(line.split('\t') for line in sources['status'].read_text().splitlines())
    rc = int(statuses[stem])
    if rc != 0 and defer_failed:
        if artifact_snapshot(sources) != before:
            raise ValueError('captured inputs changed during recovery')
        publish(target, {**{k: v for k, v in case.items() if k != 'query'},
                         'complete': False, 'returncode': rc,
                         'exclusions': ['captured_query_failed', 'partial_trace_not_decoded'],
                         'partial_trace_decode_pending': True,
                         'trace': str(trace.resolve()), 'recovery_source_files': before})
        return
    fallback = orca_fallback_reason(trace)
    with gzip.open(trace, 'rt', errors='replace') as stream:
        text = stream.read()
    run = trace_run(text, rc, fallback)
    version = json.loads(sources['manifest'].read_text()).get('required_binding_edge_trace_version', 3)
    if type(version) is not int or version not in (3, 4):
        raise ValueError('unsupported frozen binding edge trace version')
    result = summarize_trace(text, rc, fallback, required_binding_version=version,
                             require_stats=True, run=run)
    # Do not duplicate per-event state in batch reports; original lossless trace stays intact.
    keys = ('complete', 'exclusions', 'stats_timeline_complete', 'candidate_complete',
            'returncode', 'fallback', 'attempts', 'evaluations', 'statuses',
            'attempted_rules', 'ready_rules', 'memo_outcomes')
    result = {k: result[k] for k in keys}
    result.update({k: v for k, v in case.items() if k != 'query'})
    result.update(trace=str(trace.resolve()), recovery_source_files=before)
    if result['complete']:
        graph = encode_observations(run)
        graph.update(schema_version=1, case=case,
                     scope='audited_planning_history_not_execution_labels',
                     frozen_at_utc=datetime.now(timezone.utc).isoformat(), source_files=before)
        graph_path = folder / f'{stem}.recovered.graph.json.gz'
        publish(graph_path, graph, compressed=True)
        result.update(graph=str(graph_path.resolve()), graph_denominators=graph['denominators'])
    if artifact_snapshot(sources) != before:
        raise ValueError('captured inputs changed during recovery')
    publish(target, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--case', help='internal single-query worker')
    parser.add_argument('--memory-gb', type=int, default=6)
    parser.add_argument('--jobs', type=int, default=1, help='isolated decoder processes; memory cap is per job')
    parser.add_argument('--defer-failed-traces', action='store_true',
                        help='preserve failed SQL/logs as excluded, defer costly prefix decoding')
    parser.add_argument('--timeout', type=int, default=300, help='offline decoder seconds, not SQL timeout')
    args = parser.parse_args()
    if args.memory_gb < 1 or args.timeout < 1 or args.jobs < 1:
        parser.error('positive decoder limits required')
    root = args.corpus.resolve()
    manifest = json.loads((root / 'manifest.json').read_bytes())
    cases = [case for item in manifest['datasets'] for case in item['cases']]
    if args.case:
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gb * 1024 ** 3,) * 2)
        decode(root, next(c for c in cases if c['case_id'] == args.case), args.defer_failed_traces)
        return
    if (root / 'summary.json').exists():
        raise ValueError('collection already finalized')
    check_artifacts(manifest)
    def recover_case(case):
        app, stem = case['case_id'].split(':')
        if (root / app / f'{stem}.summary.json').exists():
            return None
        print('decode ' + case['case_id'], flush=True)
        try:
            result = subprocess.run([sys.executable, '-u', '-B', '-m', 'ml_orca.data.recover_history_corpus',
                '--corpus', str(root), '--case', case['case_id'], '--memory-gb', str(args.memory_gb)]
                + (['--defer-failed-traces'] if args.defer_failed_traces else []),
                timeout=args.timeout, capture_output=True, text=True, cwd=ML_ORCA_ROOT.parent)
            status = result.returncode
            if status:
                print(case['case_id'] + '\n' + result.stdout + result.stderr, flush=True)
        except subprocess.TimeoutExpired:
            status = 'decoder_timeout'
        if status != 0:
            return {'case_id': case['case_id'], 'status': status}
        return None
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failures = [result for result in pool.map(recover_case, cases) if result is not None]
    endpoint = check_artifacts(manifest)
    report = {'scope': 'offline_recovery_not_execution_labels',
              'assigned_queries': len(cases), 'decoder_failures': failures,
              'collection_artifacts_unchanged': True, 'artifact_end': endpoint,
              'recovery_tool': artifact_snapshot({'tool': Path(__file__)}),
              'decoder_limits': {'jobs': args.jobs, 'memory_gb_per_job': args.memory_gb,
                                 'timeout_seconds_per_query': args.timeout},
              'failed_trace_prefix_decoding_deferred': args.defer_failed_traces,
              'per_query_reports': '<dataset>/<query>.summary.json'}
    # Failed decoders remain missing/excluded in the original assigned population, not successful runs.
    publish(root / ('recovery-incomplete.json' if failures else 'summary.json'), report)
    print(json.dumps({'assigned': len(cases), 'decoder_failures': len(failures)}), flush=True)


if __name__ == '__main__':
    main()
