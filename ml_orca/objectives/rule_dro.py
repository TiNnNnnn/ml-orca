#!/usr/bin/env python3
"""Offline scalar W1-DRO bounds; NOT an experiment-validity or deployment certificate.

The caller must freeze policies/strata/metrics, known support and sample counts,
and justify independent calibration units. Timing repeats and trace events do
not become independent population samples by passing their count here.
No empirical min/max support, censoring imputation or online policy changes.
Theory: DKW--Massart (doi:10.1214/aop/1176990746); sorted transport follows
Chen--Kuhn--Wiesemann (arxiv:1809.00210, Theorem 2.1), with explicit handling
of open scalar bad sets, restricted support and zero radius below.
"""

from ml_orca.common.paths import PGORCA_ROOT

import math
from decimal import Decimal, ROUND_CEILING, localcontext
from fractions import Fraction


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, Fraction)):
        raise ValueError('expected a finite numeric value')
    try:
        if not math.isfinite(value):
            raise ValueError('expected a finite numeric value')
        return Fraction(value)
    except (OverflowError, TypeError) as error:
        raise ValueError('numeric value is not representable') from error


def _support(support):
    if len(support) != 2:
        raise ValueError('support must contain two known population endpoints')
    low, high = map(_number, support)
    if low > high:
        raise ValueError('reversed support')
    return low, high


def _observations(samples, support):
    low, high = _support(support)
    values = [_number(value) for value in samples]
    if not values or any(not low <= value <= high for value in values):
        raise ValueError('need nonempty observations within the declared support')
    return low, high, values


def _outward(value, upper):
    """Convert an exact rational outward, never make a safety bound optimistic."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('bound is not representable')
    rounded = Fraction(result)
    if (upper and rounded < value) or (not upper and rounded > value):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    if not math.isfinite(result):
        raise ValueError('outward bound is not representable')
    return result


def dkw_w1_radius(sample_count, support, *, beta, comparisons):
    """Upper-round (b-a)*min(1,sqrt(log(2*comparisons/beta)/(2*n))).

    comparisons is the entire PRE-FROZEN family H*K*J, not the number that
    happened to succeed. beta is total calibration failure probability, not
    allowable per-execution risk. Marginal simultaneous coverage follows from
    DKW + integral W1 identity + union bound, conditional on i.i.d. bounded data.
    """
    if any(type(v) is not int or v < 1 for v in (sample_count, comparisons)):
        raise ValueError('sample_count and comparisons must be positive integers')
    low, high = _support(support)
    beta = _number(beta)
    if not 0 < beta < 1:
        raise ValueError('beta must be between zero and one')
    with localcontext() as context:
        context.prec = 60
        context.rounding = ROUND_CEILING
        ratio = Decimal(2 * comparisons * beta.denominator) / Decimal(beta.numerator)
        # Decimal ln/sqrt are correctly rounded to nearest, irrespective of
        # context rounding; next_plus explicitly encloses them from above.
        eta = (ratio.ln().next_plus() / Decimal(2 * sample_count)).sqrt().next_plus()
        return _outward((high - low) * min(Fraction(1), Fraction(eta)), True)


def sample_budget(support, *, target_radius, beta, comparisons, max_units):
    """Smallest n within a hard acquisition cap passing the implemented DKW bound.

    This controls statistical radius, NOT positive gain or risk feasibility.
    Search the outward-rounded bound itself rather than rounding its inverse
    formula optimistically. Units are independent aggregates, not repetitions.
    """
    target = _number(target_radius)
    if target <= 0 or type(max_units) is not int or max_units < 1:
        raise ValueError('need positive target radius and a positive integer unit cap')
    at_cap = dkw_w1_radius(max_units, support, beta=beta, comparisons=comparisons)
    count = None
    if _number(at_cap) <= target:
        low, high = 1, max_units
        while low < high:
            middle = (low + high) // 2
            if _number(dkw_w1_radius(middle, support, beta=beta, comparisons=comparisons)) <= target:
                high = middle
            else:
                low = middle + 1
        count = low
    return {'scope': 'a_priori_radius_budget_not_gain_guarantee', 'required_units': count,
            'max_units': max_units, 'radius_at_cap': at_cap,
            'radius_at_required_units': None if count is None else dkw_w1_radius(
                count, support, beta=beta, comparisons=comparisons)}


def metric_bounds(samples, support, *, radius, threshold, tail):
    """Exact scalar-ball mean extrema and strict-event probability supremum.

    tail='below' means Y < threshold; 'above' means Y > threshold. For positive
    radius, transporting nonbad mass to the bad set is a fractional knapsack:
    fill cheapest distances first. An open bad set may only approach the bound.
    At radius zero the empirical strict probability must be used instead.
    Fraction arithmetic preserves input values; JSON-compatible outputs round
    lower/upper bounds outward. Runtime O(n log n); no optimization dependency.
    """
    low, high, values = _observations(samples, support)
    radius, threshold = _number(radius), _number(threshold)
    if radius < 0 or tail not in ('below', 'above'):
        raise ValueError('need nonnegative radius and below/above tail')
    mean = sum(values) / len(values)
    lower_mean, upper_mean = max(low, mean - radius), min(high, mean + radius)
    if tail == 'above':
        values, low, high, threshold = [-v for v in values], -high, -low, -threshold
    mass = Fraction(1, len(values))
    empirical = sum(value < threshold for value in values) * mass
    worst = empirical
    budget = radius
    if radius > 0 and low < threshold <= high:
        for distance in sorted(value - threshold for value in values if value >= threshold):
            moved = mass if distance == 0 else min(mass, budget / distance)
            worst += moved
            budget -= moved * distance
    return {'scope': 'ambiguity_set_bounds_only', 'sample_count': len(values),
            'radius': _outward(radius, True),
            'mean_lower': _outward(lower_mean, False),
            'mean_upper': _outward(upper_mean, True),
            'empirical_violation': float(empirical),
            'violation_upper': _outward(worst, True)}


def cdf_metric_bounds(samples, support, *, bandwidth, threshold, tail):
    """Exact extrema over {Q: ||F_Q-F_emp||_infinity <= eta, support [a,b]}.

    This is a DIFFERENT ambiguity set, not a tighter solver for a W1 ball.
    DKW covers it directly. Moving the largest eta mass to a maximizes the CDF
    and minimizes the mean; moving the smallest eta mass to b is symmetric.
    Strict threshold tails use F(t-) or 1-F(t), including boundary atoms.
    No W1 drift robustness follows from this stationary CDF confidence set.
    """
    low, high, values = _observations(samples, support)
    eta, threshold = _number(bandwidth), _number(threshold)
    if not 0 <= eta <= 1 or tail not in ('below', 'above'):
        raise ValueError('need CDF bandwidth in [0,1] and below/above tail')
    lower_mean = upper_mean = sum(values) / len(values)
    mass, remaining = Fraction(1, len(values)), eta
    ordered = sorted(values)
    for small, large in zip(ordered, reversed(ordered)):
        moved = min(mass, remaining)
        lower_mean -= moved * (large - low)
        upper_mean += moved * (high - small)
        remaining -= moved
        if not remaining:
            break
    empirical = mass * sum(value < threshold if tail == 'below' else value > threshold for value in values)
    partial = low < threshold <= high if tail == 'below' else low <= threshold < high
    worst = min(Fraction(1), empirical + eta) if partial else empirical
    return {'scope': 'cdf_band_bounds_only', 'sample_count': len(values),
            'cdf_bandwidth': _outward(eta, True),
            'mean_lower': _outward(lower_mean, False), 'mean_upper': _outward(upper_mean, True),
            'empirical_violation': float(empirical), 'violation_upper': _outward(worst, True)}


def tv_shift_bounds(bounds, support, *, distance, threshold, tail):
    """Transfer supplied scalar bounds across a KNOWN total-variation budget.

    If TV(P,Q)<=rho, bounded means move by at most (b-a)*rho and
    event probabilities by rho. A mixture-weight budget implies this only
    when conditional outcome kernels remain unchanged. This function neither
    estimates drift nor authenticates the source confidence statement.
    These are conservative transfers, not exact extrema of a W1/CDF ball.
    """
    low, high = _support(support)
    rho, threshold = _number(distance), _number(threshold)
    lower, risk = (_number(bounds[name]) for name in ('mean_lower', 'violation_upper'))
    if (not 0 <= rho <= 1 or not low <= lower <= high or not 0 <= risk <= 1
            or tail not in ('below', 'above')):
        raise ValueError('invalid TV budget, source bounds or event direction')
    partial = low < threshold <= high if tail == 'below' else low <= threshold < high
    known_bad = low < threshold if tail == 'below' else low > threshold
    if not partial and risk < int(known_bad):
        raise ValueError('source risk bound contradicts the known support')
    return {'scope': 'conditional_tv_transfer_of_supplied_bounds',
            'tv_radius': _outward(rho, True),
            'mean_lower': _outward(max(low, lower - (high - low) * rho), False),
            'violation_upper': _outward(min(Fraction(1), risk + rho) if partial else Fraction(known_bad), True)}


def demo(output, font):
    """Reproducible, explicitly synthetic sample-size diagnostic (not SQL evidence)."""
    import json
    from ml_orca.experiments.plot_stats_sweep import chinese_plotting

    counts = [8, 32, 128, 512, 2048, 8192]
    rows = []
    for count in counts:
        radius = dkw_w1_radius(count, (-1, 1), beta=0.05, comparisons=6)
        for gain in (0.1, 0.4):
            bounds = metric_bounds([gain] * count, (-1, 1), radius=radius,
                                   threshold=-0.1, tail='below')
            rows.append({'constant_gain': gain, **bounds})
    plt = chinese_plotting(font)
    fig, panels = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for gain in (0.1, 0.4):
        selected = [row for row in rows if row['constant_gain'] == gain]
        for panel, field in zip(panels, ('radius', 'mean_lower', 'violation_upper')):
            panel.plot(counts, [row[field] for row in selected], marker='o',
                       label=f'合成恒定收益 {gain}')
    for panel, title in zip(panels, ('分布不确定半径', '最坏期望收益下界', '回退概率上界')):
        panel.set(xscale='log', xlabel='假设的独立校准样本数', ylabel=title, title=title)
        panel.grid(alpha=0.3)
        panel.legend(fontsize=9)
    panels[1].axhline(0, color='black', linestyle=':')
    panels[2].axhline(0.1, color='black', linestyle=':', label='示例风险限额 0.1')
    panels[2].legend(fontsize=9)
    fig.suptitle('分布鲁棒优化数值校验：合成数据，不代表规则实测收益\n'
                 '已知支持 [-1, 1]；总置信失败概率 0.05；同时覆盖 6 项；回退阈值 -0.1')
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / '分布鲁棒界校验.png', dpi=160)
    plt.close(fig)
    (output / 'bounds.json').write_text(json.dumps({
        'scope': 'synthetic_numerical_diagnostic_not_workload_certificate',
        'beta': 0.05, 'comparisons': 6, 'support': [-1, 1],
        'threshold': -0.1, 'tail': 'below', 'rows': rows}, indent=2) + '\n')
    targets = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    budgets = []
    fig, panel = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for comparisons in (4, 200):
        values = [{'target_radius': target, 'comparisons': comparisons,
                   **sample_budget((-1, 1), target_radius=target, beta=0.05,
                                   comparisons=comparisons, max_units=1000000)} for target in targets]
        budgets.extend(values)
        panel.plot(targets, [value['required_units'] for value in values], marker='o',
                   label=f'同时覆盖 {comparisons} 项')
    panel.set(xscale='log', yscale='log', xlabel='目标半径（越小要求越严格）', ylabel='每层每策略独立单位数',
              title='事前样本预算：已知支持 [-1, 1]，总置信失败概率 0.05\n'
                    '只保证半径精度，不保证真实收益；重复计时不增加独立单位数')
    panel.invert_xaxis()
    panel.grid(alpha=0.3)
    panel.legend()
    fig.savefig(output / '事前样本预算.png', dpi=160)
    plt.close(fig)
    (output / 'budgets.json').write_text(json.dumps({
        'scope': 'theoretical_precision_budget_not_acquisition_authorization',
        'support': [-1, 1], 'beta': 0.05, 'rows': budgets}, indent=2) + '\n')


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='synthetic diagnostic directory')
    parser.add_argument('--font', type=Path, default=PGORCA_ROOT
                        / 'output/fonts/NotoSansCJKsc-Regular.otf')
    args = parser.parse_args()
    demo(args.output, args.font)
