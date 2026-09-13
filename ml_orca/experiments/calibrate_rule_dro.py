#!/usr/bin/env python3
"""Frozen finite-policy scalar DRO calibration, conditional on declared sampling.

This checks bookkeeping, NOT independence, support truth or SQL equivalence.
One row is one independent unit/policy aggregate, never one timing repetition.
Version 1 refuses any incomplete unit; it does not estimate censored runtimes.
No scheduler changes or deployment authorization are produced.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
import zlib

from ml_orca.objectives.rule_dro import _number, _support, dkw_w1_radius, metric_bounds, cdf_metric_bounds, sample_budget


def contract_identity(contract):
    """Content provenance only, consistent with existing experiment CRCs."""
    raw = json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return f'{zlib.crc32(raw):08x}'


def validate_contract(contract):
    if contract.get('ambiguity_set', 'wasserstein') not in ('wasserstein', 'dkw_cdf'):
        raise ValueError('unsupported frozen ambiguity set')
    if (type(contract.get('version')) is not int or contract['version'] != 1 or contract.get('placement') != 'cbo'
            or contract.get('incomplete_units') != 'reject_epoch'):
        raise ValueError('require version 1, fixed CBO placement and reject_epoch failures')
    for field in ('epoch', 'environment_identity', 'population', 'independent_unit',
                  'sampling_justification', 'aggregation', 'support_justification'):
        if not isinstance(contract.get(field), str) or not contract[field].strip():
            raise ValueError(f'missing contract declaration: {field}')
    if contract.get('sampling') not in ('iid_units_fixed_n', 'acquisition_smoke_test'):
        raise ValueError('require fixed-size iid calibration or explicit acquisition-only smoke test')
    if not 0 < _number(contract['beta']) < 1:
        raise ValueError('invalid total confidence budget')
    if 'holdout_beta' in contract and not 0 < _number(contract['holdout_beta']) < 1:
        raise ValueError('invalid independent holdout confidence budget')
    policies, strata, metrics = (contract[name] for name in ('policies', 'strata', 'metrics'))
    for mapping in (policies, strata, metrics):
        if not isinstance(mapping, dict) or not mapping or any(
                not isinstance(key, str) or not key for key in mapping):
            raise ValueError('need named nonempty policies, strata and metrics')
    if any(not isinstance(snapshot, str) or not snapshot for snapshot in policies.values()):
        raise ValueError('each complete policy needs a frozen snapshot identity')
    if contract.get('baseline') not in policies or contract.get('objective') != 'gain' or 'gain' not in metrics:
        raise ValueError('need a frozen baseline and a gain maximization objective')
    units = set()
    for split in strata.values():
        for phase in ('calibration', 'holdout'):
            ids = split[phase]
            if not isinstance(ids, list) or not ids or any(not isinstance(v, str) or not v for v in ids):
                raise ValueError('need predeclared nonempty calibration and holdout unit IDs')
            if len(set(ids)) != len(ids) or units.intersection(ids):
                raise ValueError('duplicate unit or overlap between strata/calibration/holdout')
            units.update(ids)
    if 'collection_units' in contract:
        bindings = contract['collection_units']
        if (not isinstance(bindings, dict) or set(bindings) != units
                or any(not isinstance(value, str) or not value for value in bindings.values())):
            raise ValueError('collection descriptors must cover every calibration and holdout unit')
    for name, metric in metrics.items():
        low, high = _support(metric['support'])
        if (metric['tail'] not in ('below', 'above') or not 0 <= _number(metric['risk_limit']) < 1
                or not isinstance(metric.get('definition'), str) or not metric['definition'].strip()):
            raise ValueError('invalid metric definition, tail or risk limit')
        threshold = _number(metric['threshold'])
        if name == 'gain' and (not low <= 0 <= high or metric['tail'] != 'below' or threshold > 0):
            raise ValueError('gain must include baseline zero and use a nonpositive regression threshold')
    if 'measurement' in contract:
        spec = contract['measurement']
        if (spec.get('kind') != 'paired_capped_timing_mean_v1'
                or type(spec.get('scenario')) is not int or spec['scenario'] < 0
                or spec.get('gain_time', 'execution') not in ('execution', 'planning_execution')
                or set(metrics) != {'gain', 'planning_ms'}):
            raise ValueError('unsupported timing metric definition or scenario')
        floor, cap, clip, planning_cap = [_number(spec[k]) for k in
            ('execution_floor_ms', 'execution_cap_ms', 'gain_clip', 'planning_cap_ms')]
        if (not 0 < floor <= cap or clip <= 0 or planning_cap <= 0
                or float(floor) == 0 or float(clip) == 0 or float(planning_cap) == 0
                or _support(metrics['gain']['support']) != (-clip, clip)
                or _support(metrics['planning_ms']['support']) != (0, planning_cap)):
            raise ValueError('timing supports must match the prespecified caps, not empirical extrema')
    return contract_identity(contract)


def bind_collection(contract, unit, acquisition):
    """Bind the actual runner descriptor BEFORE initdb; not an iid assertion."""
    identity = validate_contract(contract)
    assignments = [(h, phase) for h, split in contract['strata'].items()
                   for phase in ('calibration', 'holdout') if unit in split[phase]]
    if len(assignments) != 1:
        raise ValueError('unit is not assigned exactly once in the frozen contract')
    if acquisition['environment_identity'] != contract['environment_identity']:
        raise ValueError('collector environment differs from frozen contract')
    if acquisition['policies'] != contract['policies']:
        raise ValueError('collector policies differ from frozen contract')
    expected = contract.get('collection_units', {}).get(unit)
    if expected != contract_identity(acquisition):
        raise ValueError('collector input/protocol differs from frozen unit descriptor')
    if 'measurement' in contract and contract['measurement']['scenario'] > sum(
            name.startswith('stats_') for name in acquisition['inputs']):
        raise ValueError('requested metric scenario is not in the acquisition protocol')
    h, phase = assignments[0]
    return {'contract_identity': identity, 'unit': unit, 'stratum': h, 'phase': phase,
            'acquisition_identity': expected, 'acquisition': acquisition,
            'scope': 'bound_before_server_start_not_independence_or_calibration_certificate'}


def calibrate(contract, observations, *, frozen_selection=None):
    """Check the complete frozen Cartesian product, then compute simultaneous bounds.

    Independence/support/snapshot authenticity are external assumptions. CRC
    agreement detects accidental mismatches, not retrospective relabeling.
    Rejection applies to the epoch; failed policies never shrink HKJ or n.
    """
    identity = validate_contract(contract)
    if contract['sampling'] != 'iid_units_fixed_n':
        raise ValueError('acquisition smoke tests cannot produce population calibration certificates')
    phase, beta = 'calibration', contract['beta']
    if frozen_selection is not None:
        if (frozen_selection.get('contract_identity') != identity
                or frozen_selection.get('evaluation_phase') != 'calibration'
                or 'holdout_beta' not in contract):
            raise ValueError('holdout requires the frozen calibration selection and its own confidence budget')
        phase, beta = 'holdout', contract['holdout_beta']
    if observations.get('contract_identity') != identity:
        raise ValueError('observations do not reference the frozen contract')
    policies, strata, metrics = (contract[name] for name in ('policies', 'strata', 'metrics'))
    expected = {(h, unit, policy) for h, split in strata.items()
                for unit in split[phase] for policy in policies}
    records = {}
    for row in observations['records']:
        key = row['stratum'], row['unit'], row['policy']
        if key not in expected or key in records:
            raise ValueError('unexpected/duplicate unit-policy row (holdout and repeats are not calibration units)')
        if 'collection_units' in contract and (
                row.get('acquisition_identity') != contract['collection_units'][row['unit']]
                or row.get('input_endpoints_equal') is not True):
            raise ValueError('missing or changed prospective acquisition binding')
        if (row.get('status') != 'ok' or row.get('semantic_valid') is not True
                or row.get('environment_identity') != contract['environment_identity']
                or row.get('policy_identity') != policies[row['policy']]):
            raise ValueError('incomplete, unvalidated or mismatched unit; refusing entire epoch')
        if set(row['metrics']) != set(metrics):
            raise ValueError('missing or extra metric; cannot shrink the frozen family')
        for name, value in row['metrics'].items():
            low, high = _support(metrics[name]['support'])
            if not low <= _number(value) <= high:
                raise ValueError('observation outside known support; cannot enlarge support after sampling')
        if row['policy'] == contract['baseline'] and _number(row['metrics']['gain']) != 0:
            raise ValueError('baseline relative gain must be identically zero')
        records[key] = row
    if records.keys() != expected:
        raise ValueError('missing unit-policy rows; cannot drop failures or stop sampling early')

    comparisons = len(policies) * len(strata) * len(metrics)
    reports = {}
    for h, split in strata.items():
        candidates = {}
        for policy in policies:
            bounds = {}
            for name, metric in metrics.items():
                values = [records[h, unit, policy]['metrics'][name] for unit in split[phase]]
                # Baseline relative gain is structurally zero, but its absolute
                # planning/space risks must still pass the ordinary calibration.
                support = (0, 0) if policy == contract['baseline'] and name == 'gain' else metric['support']
                if contract.get('ambiguity_set', 'wasserstein') == 'dkw_cdf':
                    # Unit-width DKW radius is the dimensionless CDF bandwidth.
                    eta = dkw_w1_radius(len(values), (0, 1), beta=beta, comparisons=comparisons)
                    bounds[name] = cdf_metric_bounds(values, support, bandwidth=eta,
                                                     threshold=metric['threshold'], tail=metric['tail'])
                else:
                    radius = dkw_w1_radius(len(values), support, beta=beta, comparisons=comparisons)
                    bounds[name] = metric_bounds(values, support, radius=radius,
                                                 threshold=metric['threshold'], tail=metric['tail'])
            feasible = all(_number(bounds[name]['violation_upper']) <= _number(metric['risk_limit'])
                           for name, metric in metrics.items())
            candidates[policy] = {'bounds': bounds, 'risk_feasible': feasible}
        if frozen_selection is not None:
            selected = frozen_selection['strata'][h]['selected_policy']
            if selected is not None and selected not in policies:
                raise ValueError('frozen selection contains an undeclared policy')
            reports[h] = {'policies': candidates, 'selected_policy': selected, 'decision': 'evaluate_frozen_selection',
                          'selected_risk_feasible': None if selected is None else candidates[selected]['risk_feasible']}
            continue  # Never select again from holdout outcomes.
        improving = [p for p in policies if candidates[p]['risk_feasible']
                     and candidates[p]['bounds']['gain']['mean_lower'] > 0]
        selected = (max(improving, key=lambda p: candidates[p]['bounds']['gain']['mean_lower'])
                    if improving else contract['baseline']
                    if candidates[contract['baseline']]['risk_feasible'] else None)
        reports[h] = {'policies': candidates, 'selected_policy': selected,
                      'decision': 'positive_certified_lower_bound' if improving else
                      'retain_baseline' if selected else 'no_risk_feasible_fallback'}
    return {'scope': 'conditional_iid_calibration_not_deployment_authorization',
            'contract_identity': identity, 'comparisons': comparisons,
            'ambiguity_set': contract.get('ambiguity_set', 'wasserstein'),
            'beta': beta, 'evaluation_phase': phase, 'strata': reports,
            'external_assumptions': ['frozen_before_sampling', 'iid_units_from_declared_population',
                                     'known_support_and_metric_semantics', 'provenance_and_semantic_validation']}


def audit_timing_input(comparison):
    """Admit old runner data for measurement checks, NOT retrospective calibration."""
    from ml_orca.collect.run_workload_comparison import profile_comparability, results_equal, timing_schedule

    profile = comparison['rule_profile']
    timing = profile['timing']
    scenarios, samples = profile['scenarios'], timing['samples']
    reasons = []
    if (timing['source'] != 'untraced_explain_analyze'
            or timing['design'] != 'randomized_complete_blocks'
            or tuple(timing['arms']) != ('off', 'cbo')):
        reasons.append('unsupported_measurement_protocol')
    if any(type(timing[k]) is not int or timing[k] < minimum
           for k, minimum in (('repeats', 1), ('warmups', 0), ('seed', 0))):
        raise ValueError('invalid timing schedule parameters')
    expected = list(timing_schedule(len(scenarios), timing['repeats'], timing['warmups'],
                                    timing['seed'], timing['arms']))
    if [(s['phase'], s['block'], s['scenario'], s['arm']) for s in samples] != expected:
        reasons.append('incomplete_or_modified_randomized_schedule')
    if [s['sequence'] for s in samples] != list(range(len(samples))):
        reasons.append('invalid_sequence')
    provenance = comparison.get('artifact_provenance', {})
    before = provenance.get('before_server_start', {})
    if (not before or before != provenance.get('after_query') or any(
            'crc32' not in item or 'size' not in item or 'error' in item for item in before.values())):
        reasons.append('missing_or_changed_artifacts')
    oracle = comparison.get('postgres_oracle', {})
    if oracle.get('rows_rc') != 0 or oracle.get('valid') is not True:
        reasons.append('missing_or_invalid_postgres_oracle')
    for scenario in scenarios:
        arms = scenario['arms']
        checked = profile_comparability(arms['off'], arms['cbo'], scenario['stats_experiment'] is not None)
        reasons.extend(checked['exclusions'])
        if any(not results_equal(oracle, arms[arm]) for arm in ('off', 'cbo')):
            reasons.append('scenario_postgres_result_mismatch')
    for sample in samples:
        if (sample['status'] != 'ok' or sample['returncode'] != 0
                or sample['optimizer'] != 'pg_orca' or sample['diagnostic_plan_matches'] is not True
                or sample['comparison_exclusions']):
            reasons.append('failed_or_incomparable_timing_sample')
        for field in ('execution_ms', 'planning_ms'):
            try:
                if _number(sample[field]) < 0:
                    raise ValueError('negative time')
            except (ValueError, TypeError):
                reasons.append('invalid_timing')
    return {'query': comparison['query'], 'measurement_exclusions': sorted(set(reasons)),
            'scheduled_runs': len(expected), 'observed_runs': len(samples),
            'repeats_per_scenario': timing['repeats'], 'scenarios': len(scenarios),
            'population_calibration_eligible': False,
            'calibration_exclusions': ['no_prospective_dro_contract', 'no_declared_iid_population_units'],
            'scope': 'historical_measurement_audit_not_population_certificate'}


def timing_records(contract, comparison):
    """One paired aggregate per unit/policy; failures remain rows, never get imputed.

    Caps define a NEW measured utility, not original uncensored runtime gain.
    gain_time defaults to execution; planning_execution instead measures the
    sum of server planning and execution (not connection or client latency).
    The historical execution_floor/cap fields bound the selected time metric.
    Warmups validate acquisition but do not enter the aggregate; their failures
    still invalidate the entire unit. Quantized timing/float arithmetic is part
    of the declared measurement, not a claim of exact physical execution time.
    """
    validate_contract(contract)
    spec = contract['measurement']
    receipt = comparison['dro_collection']
    expected = bind_collection(contract, receipt['unit'], receipt['acquisition'])
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('comparison receipt does not match the frozen acquisition')
    descriptor = expected['acquisition']
    profile, timing = comparison['rule_profile'], comparison['rule_profile']['timing']
    actual_policies = {arm: contract_identity({'policy': text, 'background': profile['background_policy']})
                       for arm, text in profile['policy_snapshots'].items()}
    artifacts = comparison['artifact_provenance']['before_server_start']
    actual_environment = contract_identity({name: {key: item[key] for key in ('crc32', 'size')}
                                            for name, item in artifacts.items()})
    if (actual_policies != descriptor['policies'] or actual_environment != descriptor['environment_identity']
            or comparison['workload'] != descriptor['workload']
            or comparison['query_crc32'] != descriptor['inputs']['query']['crc32']
            or any(timing[key] != descriptor['protocol']['timing_' + key] for key in ('repeats', 'warmups', 'seed'))
            or any(s['timeout_ms'] != descriptor['protocol']['timeout'] * 1000 for s in timing['samples'])):
        raise ValueError('actual query, policies, environment or timing protocol differs from receipt')
    audit = audit_timing_input(comparison)
    reasons = audit['measurement_exclusions'][:]
    if receipt.get('input_endpoints_equal') is not True:
        reasons.append('input_files_changed')
    index = spec['scenario']
    if index >= len(profile['scenarios']):
        raise ValueError('missing prespecified measurement scenario')
    rows = []
    lookup = {(sample['block'], sample['arm']): sample for sample in timing['samples']
              if sample['phase'] == 'measurement' and sample['scenario'] == index}
    for policy, identity in contract['policies'].items():
        metrics = {}
        if not reasons:
            gains, planning = [], []
            floor, cap, clip, planning_cap = (float(spec[k]) for k in
                ('execution_floor_ms', 'execution_cap_ms', 'gain_clip', 'planning_cap_ms'))
            for block in range(timing['repeats']):
                sample, baseline = lookup[block, policy], lookup[block, contract['baseline']]
                total = spec.get('gain_time', 'execution') == 'planning_execution'
                execution = min(cap, max(floor, sample['execution_ms'] + (sample['planning_ms'] if total else 0)))
                reference = min(cap, max(floor, baseline['execution_ms'] + (baseline['planning_ms'] if total else 0)))
                gains.append(max(-clip, min(clip, math.log(reference) - math.log(execution))))
                planning.append(min(planning_cap, sample['planning_ms']))
            metrics = {'gain': mean(gains), 'planning_ms': mean(planning)}
        rows.append({'stratum': receipt['stratum'], 'unit': receipt['unit'], 'policy': policy,
                     'phase': receipt['phase'], 'status': 'invalid' if reasons else 'ok',
                     'semantic_valid': not reasons, 'environment_identity': actual_environment,
                     'policy_identity': identity, 'acquisition_identity': receipt['acquisition_identity'],
                     'input_endpoints_equal': receipt.get('input_endpoints_equal') is True,
                     'metrics': metrics, 'exclusions': reasons,
                     'measurement_repeats': timing['repeats'], 'measurement_scenario': index})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze', help='validate then write a new, non-overwritten contract')
    freeze.add_argument('input', type=Path)
    evaluate = commands.add_parser('calibrate')
    evaluate.add_argument('contract', type=Path)
    evaluate.add_argument('observations', type=Path)
    evaluate.add_argument('--frozen-selection', type=Path,
                          help='evaluate holdout against a previous calibration result, without reselection')
    audit = commands.add_parser('audit-timing')
    audit.add_argument('comparisons', type=Path, nargs='+')
    export = commands.add_parser('export-timing', help='convert bound comparisons to one row per unit/policy')
    export.add_argument('contract', type=Path)
    export.add_argument('comparisons', type=Path, nargs='+')
    budget = commands.add_parser('budget', help='plan radius precision before acquiring any outcomes')
    budget.add_argument('--support', nargs=2, type=float, required=True)
    budget.add_argument('--target-radius', type=float, required=True)
    budget.add_argument('--beta', type=float, required=True)
    budget.add_argument('--comparisons', type=int, required=True)
    budget.add_argument('--max-units', type=int, required=True)
    for command in (freeze, evaluate, audit, budget, export):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        result = json.loads(args.input.read_text())
        validate_contract(result)
    elif args.command == 'calibrate':
        result = calibrate(json.loads(args.contract.read_text()), json.loads(args.observations.read_text()),
                           frozen_selection=json.loads(args.frozen_selection.read_text()) if args.frozen_selection else None)
    elif args.command == 'budget':
        result = {**sample_budget(args.support, target_radius=args.target_radius, beta=args.beta,
                                 comparisons=args.comparisons, max_units=args.max_units),
                  'support': args.support, 'target_radius': args.target_radius,
                  'beta': args.beta, 'comparisons': args.comparisons}
    elif args.command == 'export-timing':
        contract = json.loads(args.contract.read_text())
        records = [row for path in args.comparisons for row in timing_records(contract, json.loads(path.read_text()))]
        keys = [(row['unit'], row['policy']) for row in records]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate acquisition unit/policy; cannot count repeated input files twice')
        result = {'contract_identity': contract_identity(contract), 'records': records,
                  'scope': 'unit_measurements_not_population_certificate'}
    else:
        result = [audit_timing_input(json.loads(path.read_text())) for path in args.comparisons]
    # No quiet overwrite of contracts, certificates or audits on rerun.
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
