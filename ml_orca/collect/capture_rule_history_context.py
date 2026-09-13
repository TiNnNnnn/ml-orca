#!/usr/bin/env python3
"""Capture existing catalog and resolved behavior policy before a corpus is planned."""

import argparse
import json
from pathlib import Path

from ml_orca.common.artifacts import artifact_snapshot
from ml_orca.collect.run_workload_comparison import collect_catalog_context, collect_policy_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--psql', type=Path, required=True)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--audit-bin', type=Path, required=True)
    parser.add_argument('--rules', type=Path, required=True)
    parser.add_argument('--schema', type=Path, required=True)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {'schema': args.schema, 'rules': args.rules, 'audit': args.audit_bin,
             'capture': Path(__file__), 'catalog_capture': Path(__file__).with_name('run_workload_comparison.py')}
    if args.policy:
        paths['policy'] = args.policy
    before = artifact_snapshot(paths)
    context = collect_catalog_context(args.psql, args.socket, args.port, 'postgres', 60)
    context.update(resolved_policies=collect_policy_context(args.audit_bin, args.rules,
                   {'behavior': args.policy}, 60),
                   policy_document=None if args.policy is None else args.policy.read_text(),
                   input_files=before, capture_input_endpoints_equal=artifact_snapshot(paths) == before,
                   fixture={'scope': 'migrated_schema_without_loaded_rows', 'schema': before['schema']},
                   label_scope='planning_history_only_no_execution_time_label')
    with args.output.open('x') as stream:
        json.dump(context, stream, allow_nan=False)
    if (context['status'] != 'ok' or context['resolved_policies']['behavior']['status'] != 'ok'
            or not context['capture_input_endpoints_equal']):
        raise SystemExit('history context capture failed; no queries dispatched')


if __name__ == '__main__':
    main()
