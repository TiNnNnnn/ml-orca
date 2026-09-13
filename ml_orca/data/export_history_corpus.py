#!/usr/bin/env python3
"""Index admitted corpus graphs for the existing trainer without duplicating their payloads."""

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from ml_orca.common.artifacts import read_snapshot
from ml_orca.common.artifacts import artifact_snapshot


def iter_history_runs(bundle):
    """Load one compressed payload at a time; legacy inline history stays supported."""
    for run in bundle['runs']:
        if 'graph_payload' not in run:
            yield run
            continue
        payload = json.loads(gzip.decompress(read_snapshot(run['graph_payload'])))
        if (payload.get('schema_version') != 1
                or payload.get('scope') != 'audited_planning_history_not_execution_labels'
                or payload['case'] != run['case']
                or payload['denominators'] != run['denominators']
                or run['source'] != run['graph_payload']):
            raise ValueError('history payload differs from admitted query/denominators')
        if (run['unit']['workload'] + ':' + Path(run['unit']['query']).stem != run['case']['case_id']
                or run['inputs']['query_sql'] != run['case']['query']):
            raise ValueError('history payload query identity mismatch')
        for snapshot in payload['source_files'].values():
            read_snapshot(snapshot)
        yield {**run, **{k: payload[k] for k in ('trees', 'contexts', 'edges')}}


def export_bundle(root, audit_path, freeze_available=False):
    receipt = artifact_snapshot({'audit': audit_path})['audit']
    audit = json.loads(read_snapshot(receipt))
    final = (audit.get('progress_only') is False and audit.get('collection_finalized') is True
             and audit.get('minimum_training_graphs_met') is True)
    available = (freeze_available and audit.get('progress_only') is True
                 and type(audit.get('eligible_graphs_with_dynamic_edges')) is int
                 and audit['eligible_graphs_with_dynamic_edges'] >= 2000)
    if (audit.get('scope') != 'query_graph_admission_not_execution_labels_or_model_benefit'
            or not (final or available)):
        raise ValueError('require finalized >=2000-query training admission, not progress counts')
    manifest_snapshot = artifact_snapshot({'manifest': root / 'manifest.json'})['manifest']
    manifest = json.loads(read_snapshot(manifest_snapshot))
    for snapshot in manifest['artifact_start'].values():
        read_snapshot(snapshot)
    if manifest.get('history_split') != 'train':
        raise ValueError('only training history may enter the shared graph')
    cases = {c['case_id']: c for d in manifest['datasets'] for c in d['cases']}
    if (len(cases) != audit['assigned_queries'] or len(audit['queries']) != len(cases)
            or {r['case_id'] for r in audit['queries']} != cases.keys()):
        raise ValueError('admission differs from assigned corpus population')
    graph_snapshot = artifact_snapshot({'graph': root / 'audit/rule_graph.json'})['graph']
    contexts, runs = {}, []
    for row in audit['queries']:
        if not row['eligible']:
            continue  # Full failed/negative denominator stays in the referenced admission report.
        case = cases[row['case_id']]
        app, stem = case['case_id'].split(':')
        snapshot = row['graph_snapshot']
        payload = json.loads(gzip.decompress(read_snapshot(snapshot)))
        if payload['case'] != case or payload['source_files']['manifest'] != manifest_snapshot:
            raise ValueError('graph belongs to another corpus capture')
        context_snapshot = payload['source_files']['context']
        if Path(context_snapshot['path']).resolve() != (root / app / 'pre-workload-context.json').resolve():
            raise ValueError('graph belongs to another catalog')
        if app not in contexts:
            contexts[app] = json.loads(read_snapshot(context_snapshot))
        context = contexts[app]
        policy = context['resolved_policies']['behavior']
        if (context['status'] != 'ok' or not context['capture_input_endpoints_equal']
                or policy['status'] != 'ok'):
            raise ValueError('invalid captured behavior policy')
        runs.append({'source': snapshot, 'graph_payload': snapshot, 'case': case,
                     'unit': {'workload': app, 'query': stem, 'policy': 'behavior'},
                     'inputs': {'query_sql': case['query'], 'graph_snapshot': graph_snapshot,
                                'catalog_snapshot': context_snapshot,
                                'candidate_policy': policy['snapshot']['rules']},
                     'denominators': payload['denominators']})
    bundle = {'schema_version': 1, 'capture': 'audited_history_frozen',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'runs': runs,
              'population_scope': 'final_collection' if final else 'available_at_audit_cutoff',
              'collection_finalized': audit['collection_finalized'],
              'assigned_queries': audit['assigned_queries'],
              'admission_report': receipt, 'corpus_manifest': manifest_snapshot}
    # The report cannot bless modified/unresolved graph payloads: reuse the trainer gate.
    from ml_orca.data.admission import history_training_gate
    history_training_gate(bundle,
                          {k: c['query'] for k, c in cases.items()}, 2000)
    read_snapshot(receipt)
    for snapshot in manifest['artifact_start'].values():
        read_snapshot(snapshot)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze-available', action='store_true',
                        help='explicit pilot cutoff; still requires >=2000 complete dynamic query graphs')
    args = parser.parse_args()
    bundle = export_bundle(args.corpus.resolve(), args.admission, args.freeze_available)
    with args.output.open('x') as stream:
        json.dump(bundle, stream, allow_nan=False)
    print(json.dumps({'indexed_queries': len(bundle['runs']), 'frozen_at_utc': bundle['frozen_at_utc']}))


if __name__ == '__main__':
    main()
