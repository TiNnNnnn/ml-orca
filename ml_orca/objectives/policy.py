"""Prospective complete-policy targets and workload-level evaluation."""
import json
import math
from statistics import median

def policy_target(record):
    values = [record['response'][key] for key in ('planning_ms_median', 'execution_ms_median')]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('invalid measured policy latency')
    return [math.log1p(value) for value in values]

def objective_channels(objective):
    if objective == 'planning':
        return ('planning_ms',)
    if objective == 'plan_execution':
        return ('planning_ms', 'execution_ms')
    raise ValueError('unknown prediction objective')

def selection_metrics(predictions, records, split, objective='plan_execution'):
    """Score one fixed policy for the whole split, never choose using response labels.

    Descriptive equal-query pilot only: failed/missing arms exclude the whole
    query from paired scoring, not just the inconvenient policy.
    """
    def key(unit):
        return json.dumps(unit, sort_keys=True)

    targets = objective_channels(objective)
    predicted = {}
    for item in predictions:
        if item['split'] == split:
            identity = key(item['unit'])
            if identity in predicted:
                raise ValueError('duplicate policy prediction')
            predicted[identity] = item
    groups, seen = {}, set()
    for record in records:
        if record['pilot']['split'] != split:
            continue
        unit = record['unit']
        identity = key(unit)
        if identity in seen:
            raise ValueError('duplicate policy response')
        seen.add(identity)
        group = key({k: v for k, v in unit.items() if k != 'policy'})
        groups.setdefault(group, []).append(record)
    if predicted.keys() - seen:
        raise ValueError('prediction without assigned response')
    totals, rows, excluded = {}, [], []
    channels = {'model': 'predicted_log1p_ms', 'training_policy_constant': 'training_policy_constant'}
    for group, cells in groups.items():
        if (len(cells) < 2 or any(r['pilot']['exclusions'] or key(r['unit']) not in predicted
                                or r['response']['status'] != 'complete' for r in cells)):
            excluded.append({'unit': json.loads(group), 'reason': 'incomplete_policy_comparison'})
            continue
        arms = {r['unit']['policy'] for r in cells}
        if totals and arms != set(totals):
            raise ValueError('workload selection requires identical candidate policy sets')
        costs = {}
        for record in cells:
            arm = record['unit']['policy']
            prediction = predicted[key(record['unit'])]
            samples = [s for s in record['response']['timing_samples'] if s['phase'] == 'measurement']
            if not samples or any(s['status'] != 'ok' or s['comparison_exclusions']
                    or s.get('optimizer') != 'pg_orca'
                    or any(type(s.get(k)) not in (int, float) or not math.isfinite(s[k]) or s[k] < 0
                           for k in ('planning_ms', 'execution_ms')) for s in samples):
                raise ValueError('complete response has invalid measured timings')
            costs[arm] = {'observed_ms': median(sum(s[k] for k in targets) for s in samples)}
            for channel, field in channels.items():
                values = prediction[field]
                if len(values) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                    raise ValueError('invalid time prediction')
                # Nonnegative time domain; a sum of predicted marginal medians
                # is only a proxy for the median of joint planning+execution.
                costs[arm][channel + '_ms'] = sum(math.expm1(max(0., v)) for v in values[:len(targets)])
            total = totals.setdefault(arm, {k: 0. for k in costs[arm]})
            for channel, value in costs[arm].items():
                total[channel] += value
                if not math.isfinite(total[channel]):
                    raise ValueError('nonfinite policy total')
        rows.append({'unit': json.loads(group), 'policies': costs})
    result = {'assigned_queries': len(groups), 'paired_queries': len(rows), 'excluded': excluded,
              'objective': ('equal_query_sum_of_median_planning_ms' if objective == 'planning'
                            else 'equal_query_sum_of_median_plan_plus_execution_ms'),
              'scope': 'descriptive_complete_pairs_not_population_or_safety_certificate',
              'includes_inference_or_maintenance_cost': False, 'queries': rows, 'policy_totals': totals}
    if totals:
        oracle = min(totals, key=lambda arm: (totals[arm]['observed_ms'], arm))
        result['best_measured_fixed_policy'] = oracle
        result['selectors'] = {}
        for channel in channels:
            chosen = min(totals, key=lambda arm: (totals[arm][channel + '_ms'], arm))
            result['selectors'][channel] = {'policy': chosen,
                'observed_ms': totals[chosen]['observed_ms'],
                'empirical_regret_ms': totals[chosen]['observed_ms'] - totals[oracle]['observed_ms']}
    return result
