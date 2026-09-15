#!/usr/bin/env python3
"""Separate prospective inputs from whole-policy responses; retain incomplete cells."""

from ml_orca.common.artifacts import artifact_snapshot, read_snapshot
from ml_orca.common.stats_requests import validate_request_snapshot

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
from statistics import median
import zlib

from ml_orca.collect.run_workload_comparison import experiment_run_status, timing_schedule
from ml_orca.trace.audit_rule_utility import audit_utility_run


def policy_runtime_settings(runtime):
    """Remove only the resolved policy path, preserving every other runtime setting."""
    if not isinstance(runtime, str) or not runtime:
        raise ValueError('missing captured measurement settings')
    runtime, replaced = re.subn(r"(?m)^SET pg_orca\.dsl_rule_policy_path='(?:''|[^'])*';$",
                               "SET pg_orca.dsl_rule_policy_path='<encoded_policy>';", runtime)
    if replaced != 1:
        raise ValueError('expected one explicitly resolved policy setting')
    return runtime


def policy_samples(result, context, query_sql, graph, input_errors=(), *, include_search=False):
    profile = result['policy_comparison']
    if profile.get('scope') != 'complete_policy_not_individual_rule_effect':
        raise ValueError('require a complete-policy comparison, not a single-rule effect')
    receipt = result.get('feature_graph') or {}
    errors = list(input_errors)
    if (receipt.get('capture') != 'before_server_start'
            or context.get('feature_graph') != receipt):
        errors.append('missing_or_inconsistent_frozen_graph')
    try:
        frozen = datetime.fromisoformat(receipt['frozen_at_utc'])
        captured = datetime.fromisoformat(context['catalog']['captured_at'])
        if frozen.utcoffset() is None or captured.utcoffset() is None or frozen > captured:
            raise ValueError('invalid capture order')
    except (KeyError, TypeError, ValueError):
        errors.append('unverified_capture_order')
    if (context.get('status') != 'ok' or not context.get('capture_input_endpoints_equal')
            or not result.get('pre_workload_context', {}).get('input_endpoints_equal')
            or not result.get('artifact_provenance', {}).get('endpoints_equal')):
        errors.append('input_integrity_not_verified')
    if query_sql is None or f'{zlib.crc32(query_sql.encode()):08x}' != result['query_crc32']:
        errors.append('query_content_not_verified')
    known = {n['rule_hash'] for n in graph.get('nodes', [])}
    timing = profile.get('timing') or {}
    samples = timing.get('samples', [])
    expected = list(timing_schedule(len(profile['scenarios']), timing.get('repeats', 0),
                                   timing.get('warmups', 0), timing.get('seed', 0), profile['arms']))
    schedule_ok = (timing.get('repeats', 0) > 0 and
                   [(s['phase'], s['block'], s['scenario'], s['arm']) for s in samples] == expected)
    records = []
    for index, scenario in enumerate(profile['scenarios']):
        intervention = None
        requests = None
        scenario_errors = list(errors)
        if scenario['stats_experiment'] is not None:
            matches = [key.split(':', 1)[1] for key, value in context.get('input_files', {}).items()
                       if key.startswith('stats:') and value['path'] == scenario['stats_experiment']]
            if len(matches) == 1:
                intervention = context['stats_experiment_documents'].get(matches[0])
            if intervention is None:
                scenario_errors.append('intervention_document_missing')
            elif 'stats_experiment_requests' in context:
                stats_receipt = context['stats_experiment_requests'].get(matches[0])
                try:
                    if not isinstance(stats_receipt, dict) or stats_receipt.get('status') != 'ok':
                        raise ValueError('native request snapshot missing or failed')
                    validate_request_snapshot(stats_receipt.get('snapshot'))
                    requests = stats_receipt['snapshot']
                except ValueError:
                    scenario_errors.append('native_stats_requests_not_verified')
        for arm in profile['arms']:
            run = scenario['arms'][arm]
            baseline = profile['arms'][0]
            expect_outcome = scenario['stats_experiment'] is not None
            baseline_status = experiment_run_status(scenario['arms'][baseline], expect_outcome)
            # Samples retain pairwise exclusions. A failed reference invalidates
            # its contrasts, not another policy's independently validated label.
            reference_only = ({baseline + ':' + baseline_status}
                              if arm != baseline and baseline_status != 'ok'
                              and experiment_run_status(run, expect_outcome) == 'ok' else set())
            exclusions = list(scenario_errors)
            policy = context.get('resolved_policies', {}).get(arm, {})
            rules = (policy.get('snapshot') or {}).get('rules')
            if policy.get('status') != 'ok' or rules is None:
                exclusions.append('native_policy_not_resolved')
            elif any(r['rule_hash'] not in known for r in rules):
                exclusions.append('loaded_rule_missing_from_graph')
            cell = [s for s in samples if s['scenario'] == index and s['arm'] == arm]
            measured = [s for s in cell if s['phase'] == 'measurement']
            response_errors = []
            if not schedule_ok:
                response_errors.append('incomplete_timing_schedule')
            if run['plan_rc'] or run['rows_rc'] or run.get('optimizer') != 'pg_orca':
                response_errors.append('diagnostic_execution_failed')
            if result.get('postgres_oracle', {}).get('mode_result_equal', {}).get(f'policy:{index}:{arm}') is not True:
                response_errors.append('independent_postgres_check_failed')
            if any(s['status'] != 'ok' or
                   (set(s['comparison_exclusions']) -
                    (reference_only if s.get('diagnostic_plan_matches') is True else set()))
                   or s.get('optimizer') != 'pg_orca'
                   or any(type(s.get(k)) not in (int, float) or not math.isfinite(s[k]) or s[k] < 0
                          for k in ('planning_ms', 'execution_ms')) for s in cell):
                response_errors.append('invalid_timing_sample_including_warmups')
            complete = not response_errors
            records.append({
                'schema_version': 1,
                'unit': {'workload': result['workload'], 'query': result['query'],
                         'query_crc32': result['query_crc32'], 'fixture': context.get('fixture'),
                         'scenario': index, 'policy': arm},
                'inputs': {'query_sql': query_sql, 'graph_snapshot': receipt.get('snapshot'),
                           'catalog_snapshot': result.get('pre_workload_context', {}).get('snapshot'),
                           'candidate_policy': rules, 'stats_experiment_document': intervention,
                           'stats_experiment_requests': requests},
                'response': {'status': 'complete' if complete else 'incomplete',
                             'validation_scope': 'individual_policy_not_reference_contrast',
                             'planning_ms_median': median(s['planning_ms'] for s in measured) if complete else None,
                             'execution_ms_median': median(s['execution_ms'] for s in measured) if complete else None,
                             'timing_samples': cell, 'exclusions': response_errors},
                'admission': {'feature_integrity_verified': not exclusions,
                              'feature_exclusions': exclusions,
                              'model_training_eligible': False,
                              'training_exclusions': ['independent_split_not_assigned',
                                                      'historical_graph_population_not_audited',
                                                      'measurement_environment_not_admitted']}})
            if include_search:
                audited = audit_utility_run(run)
                search_errors = []
                status = experiment_run_status(run, expect_outcome)
                if status != 'ok':
                    search_errors.append(status)
                if result.get('postgres_oracle', {}).get('mode_result_equal', {}).get(f'policy:{index}:{arm}') is not True:
                    search_errors.append('independent_postgres_check_failed')
                runtime = None
                try:
                    runtime = policy_runtime_settings(context.get('settings_sql', {}).get(arm))
                except ValueError as error:
                    search_errors.append(str(error))
                # Keep separate masks and units. First-feasible quality is not
                # an incumbent decrease; none of these are per-rule D/F labels.
                records[-1]['response']['search'] = {
                    'scope': 'whole_policy_cost_and_observed_work_not_per_rule_credit',
                    'validation': {'complete': not search_errors, 'exclusions': search_errors,
                                   'runtime_settings': runtime,
                                   'scope': 'diagnostic_plan_and_independent_postgres_not_timing'},
                    'trace_audits': {**audited['audits'], **{
                        name: {key: audited[name][key] for key in ('complete', 'exclusions')}
                        for name in ('cost_origin_audit', 'physical_plan_source_audit')}},
                    'terminal_quality': audited['terminal_plan_quality'],
                    'work': audited['search_work_audit'],
                    'progress': audited['root_search_progress'],
                }
    return records


def export_comparison(path, *, include_search=False):
    result = json.loads(path.read_bytes())
    errors, context, graph, query_sql = [], {}, {}, None
    try:
        context = json.loads(read_snapshot(result['pre_workload_context']['snapshot']))
        graph = json.loads(read_snapshot(result['feature_graph']['snapshot']))
        query_sql = read_snapshot(context['input_files']['query:' + result['query']]).decode()
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
    records = policy_samples(result, context, query_sql, graph, errors, include_search=include_search)
    for record in records:
        record['response']['comparison_path'] = str(path.resolve())
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True, help='comparison.json (repeatable)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-search', action='store_true',
                        help='also audit/export terminal cost, work vector and root progress; not D/F credit')
    args = parser.parse_args()
    paths = [p.resolve() for p in args.input]
    if len(set(paths)) != len(paths):
        parser.error('duplicate comparison input')
    records = [record for path in paths for record in export_comparison(path, include_search=args.include_search)]
    with args.output.open('x') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps({'records': len(records), 'responses': dict(Counter(r['response']['status'] for r in records)),
                      'verified_inputs': sum(r['admission']['feature_integrity_verified'] for r in records),
                      'training_eligible': 0}))


if __name__ == '__main__':
    main()
