"""Versioned D/F/C utility and complete-workload ranking, not time regression.

Inputs are predictions for COMPLETE configurations, not additive rule scores.
Scales and preferences belong to a frozen experiment contract. Hard optimizer
budgets, semantic admission and independent DRO calibration remain external.
"""
import math

OBJECTIVE = 'rule-net-utility-v1'


def _finite(value, name, minimum=None):
    try:
        valid = (type(value) in (int, float) and math.isfinite(value)
                 and (minimum is None or value >= minimum))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('invalid ' + name)
    return value


def net_utility(direct_gain, future_gain, search_work, *, plan_cost_scale,
                search_work_scale, future_weight=1., search_weight=1.):
    """Return explicit dimensionless contributions; missing/negative C is invalid.

    D/F may be signed for configuration-change predictions. Neither share an
    implicit time unit with C. The caller must name C's work metric, deduplicate
    shared descendants and use identical scales across compared configurations.
    """
    for name, value in (('plan_cost_scale', plan_cost_scale), ('search_work_scale', search_work_scale)):
        if _finite(value, name, 0) == 0:
            raise ValueError(name + ' must be positive')
    _finite(future_weight, 'future_weight', 0)
    _finite(search_weight, 'search_weight', 0)
    d = _finite(direct_gain, 'direct_gain') / plan_cost_scale
    f = _finite(future_gain, 'future_gain') / plan_cost_scale
    c = _finite(search_work, 'search_work', 0) / search_work_scale
    result = {'direct': d, 'future': future_weight * f, 'search_penalty': search_weight * c}
    result['utility'] = result['direct'] + result['future'] - result['search_penalty']
    for name, value in result.items():
        _finite(value, name)
    return result


def rank_policy_predictions(predictions, *, query_weights, policies, **utility_contract):
    """Rank a frozen candidate family on exactly the same weighted workload.

    A policy identity includes subset AND order/priority. Predictions must cover
    the complete query/policy product; failure/missing arms are never dropped.
    This is the scoring boundary for a selector, not a beam search or an oracle.
    """
    if (not query_weights or not policies or len(set(policies)) != len(policies)
            or any(not isinstance(x, str) or not x for x in (*query_weights, *policies))):
        raise ValueError('need distinct policies and nonempty query identities')
    for weight in query_weights.values():
        if _finite(weight, 'query weight', 0) == 0:
            raise ValueError('query weights must be positive')
    weight_sum = _finite(sum(query_weights.values()), 'weight sum')
    expected = {(q, p) for q in query_weights for p in policies}
    seen, terms = set(), {p: [] for p in policies}
    for row in predictions:
        key = (row.get('query'), row.get('policy'))
        if key not in expected or key in seen:
            raise ValueError('unknown or duplicate query/policy prediction')
        if row.get('status') != 'complete':
            raise ValueError('incomplete policy prediction; cannot rank by dropping an arm')
        seen.add(key)
        score = net_utility(row.get('direct_gain'), row.get('future_gain'), row.get('search_work'),
                            **utility_contract)['utility']
        terms[key[1]].append(query_weights[key[0]] / weight_sum * score)
    if seen != expected:
        raise ValueError('missing query/policy prediction')
    try:
        values = {p: _finite(math.fsum(xs), 'workload utility') for p, xs in terms.items()}
    except OverflowError as error:
        raise ValueError('nonfinite workload utility') from error
    return [{'policy': p, 'utility': values[p]} for p in sorted(policies, key=lambda p: (-values[p], p))]
