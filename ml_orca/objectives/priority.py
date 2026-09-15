"""Paired priority-policy supervision, not per-rule credit or a scalar time proxy."""
import math

from ml_orca.objectives.search_work import WORK_CONTRACT

PRIORITY_OBJECTIVE = 'paired-log1p-plan-cost-margin-v1'


def _work(audit):
    if audit.get('contract') != WORK_CONTRACT or audit.get('complete') is not True or audit.get('exclusions') != []:
        raise ValueError('incomplete_observed_work')
    values = audit.get('values')
    fields = {'dsl_dispatched_attempts', 'dsl_evaluated_attempts', 'dsl_budget_skips',
              'cost_entries', 'cost_entries_by_status', 'search_checks_by_stage_and_status'}
    if not isinstance(values, dict) or set(values) != fields:
        raise ValueError('invalid_work_channels')
    flat = {}

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str) or not key or '/' in key:
                    raise ValueError('invalid_work_channel')
                visit(child, path + '/' + key)
        elif type(value) is int and value >= 0:
            flat[path] = value
        else:
            raise ValueError('invalid_work_count')

    for key, value in values.items():
        if key.endswith('_status') != isinstance(value, dict):
            raise ValueError('invalid_work_channel_shape')
        visit(value, key)
    if (values['dsl_dispatched_attempts'] != values['dsl_evaluated_attempts'] + values['dsl_budget_skips']
            or values['cost_entries'] != sum(v for k, v in flat.items() if k.startswith('cost_entries_by_status/'))):
        raise ValueError('inconsistent_work_totals')
    return flat


def priority_pair(left, right):
    """Contrast two exported cells of ONE frozen query/scenario/comparison.

    Lower cost wins; log1p(cost_left)-log1p(cost_right) is a signed ranking
    margin, not an absolute improvement shared across queries. Work retains
    separate event units. Missing labels stay missing, including failed arms.
    This does not grant training admission or use current traces as features.
    """
    errors = []
    if left is None or right is None:
        return {'contract': 'priority-pair-v1', 'exclusions': ['missing_policy_cell'],
                'cost': None, 'work_delta': None, 'cost_exclusions': ['missing_policy_cell'],
                'work_exclusions': ['missing_policy_cell'], 'trace_complete': False}
    units = [{k: v for k, v in row['unit'].items() if k != 'policy'} for row in (left, right)]
    if units[0] != units[1] or left['unit']['policy'] == right['unit']['policy']:
        errors.append('different_query_scenario_or_same_policy')
    paths = [r['response'].get('comparison_path') for r in (left, right)]
    if not paths[0] or paths[0] != paths[1]:
        errors.append('different_or_missing_frozen_comparison')
    inputs = [row['inputs'] for row in (left, right)]
    for key in ('query_sql', 'graph_snapshot', 'catalog_snapshot', 'stats_experiment_document', 'stats_experiment_requests'):
        if (inputs[0].get(key) != inputs[1].get(key)
                or (key in ('query_sql', 'graph_snapshot', 'catalog_snapshot') and not inputs[0].get(key))):
            errors.append('different_or_missing_input:' + key)
    normalized = []
    for side, row in zip(('left', 'right'), (left, right)):
        admission = row['admission']
        if admission.get('feature_integrity_verified') is not True or admission.get('feature_exclusions') != []:
            errors.append(side + ':unverified_inputs')
        policy = row['inputs'].get('candidate_policy')
        if (not isinstance(policy, list) or not policy
                or any(r.get('placement') != 'cbo' or type(r.get('priority')) is not int for r in policy)
                or len({r['rule_hash'] for r in policy}) != len(policy)):
            errors.append(side + ':invalid_cbo_policy')
            normalized.append(None)
        else:
            # Preserve native order and every other field, including all budgets.
            normalized.append([{k: v for k, v in r.items() if k != 'priority'} for r in policy])
    if normalized[0] != normalized[1]:
        errors.append('changes_beyond_priority')
    searches = [r['response'].get('search', {}) for r in (left, right)]
    for side, search in zip(('left', 'right'), searches):
        validation = search.get('validation', {})
        if validation.get('complete') is not True or validation.get('exclusions') != []:
            errors.append(side + ':invalid_diagnostic_or_postgres_validation')
            errors.extend(side + ':' + e for e in validation.get('exclusions', []))
    runtimes = [s.get('validation', {}).get('runtime_settings') for s in searches]
    if not runtimes[0] or runtimes[0] != runtimes[1]:
        errors.append('different_or_missing_runtime_settings')
    cost_errors, work_errors = list(errors), list(errors)
    costs, work = [], []
    for side, search in zip(('left', 'right'), searches):
        quality = search.get('terminal_quality', {})
        value = quality.get('optimizer_cost')
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if quality.get('available') is not True or quality.get('exclusions') != [] or not valid:
            cost_errors.append(side + ':unavailable_terminal_cost')
        costs.append(value)
        try:
            work.append(_work(search.get('work', {})))
        except ValueError as error:
            work_errors.append(side + ':' + str(error))
    cost = None
    if not cost_errors:
        cost = {'left': costs[0], 'right': costs[1],
                'log1p_margin': math.log1p(costs[0]) - math.log1p(costs[1]),
                'observed_winner': 'left' if costs[0] < costs[1] else 'right' if costs[0] > costs[1] else 'tie'}
    work_delta = None
    if not work_errors:
        # Absent statuses are zero ONLY in complete event streams.
        work_delta = {k: work[0].get(k, 0) - work[1].get(k, 0) for k in sorted(work[0].keys() | work[1].keys())}
    audits = ('attempts', 'origins', 'cost_lifecycle', 'cost_origin_audit', 'physical_plan_source_audit')
    return {'contract': 'priority-pair-v1', 'exclusions': sorted(set(errors)),
            'cost': cost, 'work_delta': work_delta, 'cost_exclusions': sorted(set(cost_errors)),
            'work_exclusions': sorted(set(work_errors)),
            'trace_complete': all(s.get('trace_audits', {}).get(k, {}).get('complete') is True
                                  and s['trace_audits'][k].get('exclusions') == [] for s in searches for k in audits)}
