#!/usr/bin/env python3
"""Compare audited scale responses without assigning predetermined curve families."""

import argparse
import json
import math
from pathlib import Path
from statistics import linear_regression, mean, median

from ml_orca.experiments.plot_stats_sweep import chinese_plotting


def model_check(xs, ys):
    """Leave one scale out, not one timing repeat or one query out."""
    if len(xs) < 3 or len(xs) != len(ys) or len(set(xs)) != len(xs):
        raise ValueError('need at least three distinct paired scales')
    if not all(math.isfinite(v) for v in [*xs, *ys]):
        raise ValueError('nonfinite curve observation')
    errors = {'constant': [], 'affine_rows': []}
    for i, (x, y) in enumerate(zip(xs, ys)):
        tx, ty = xs[:i] + xs[i+1:], ys[:i] + ys[i+1:]
        slope, intercept = linear_regression(tx, ty)
        errors['constant'].append(abs(y - mean(ty)))
        errors['affine_rows'].append(abs(y - (slope*x + intercept)))
    slope, intercept = linear_regression(xs, ys)
    return {'leave_one_scale_out_mae': {k: mean(v) for k, v in errors.items()},
            'heldout_absolute_errors': errors, 'affine_slope': slope, 'affine_intercept': intercept,
            'scope': 'within_query_descriptive_model_check_not_cross_query_prediction'}


def transport_check(train_x, train_y, test_x, test_y):
    """Freeze the same two simple models on source observations; never refit on the target."""
    fitted = model_check(train_x, train_y)
    if (not test_x or len(test_x) != len(test_y)
            or not all(math.isfinite(v) for v in [*test_x, *test_y])):
        raise ValueError('need finite paired transfer observations')
    predictions = {
        'constant': [mean(train_y)] * len(test_x),
        'affine_rows': [fitted['affine_slope'] * x + fitted['affine_intercept'] for x in test_x]}
    errors = {key: [abs(y-p) for y, p in zip(test_y, values)] for key, values in predictions.items()}
    return {'source_fit': fitted, 'source_constant': mean(train_y), 'predictions': predictions,
            'absolute_errors': errors, 'mae': {key: mean(values) for key, values in errors.items()},
            'extrapolated': [not min(train_x) <= x <= max(train_x) for x in test_x],
            'scope': 'frozen_source_curve_transfer_not_target_fit_or_population_guarantee'}


def additive_response_check(xs, ys, control):
    """Predict joint interventions from coordinate axes only; never fit on joint responses."""
    if (len(control) < 2 or not xs or len(xs) != len(ys)
            or any(len(x) != len(control) for x in xs)
            or not all(math.isfinite(v) for v in [*control, *ys, *(v for x in xs for v in x)])):
        raise ValueError('need finite paired multi-input observations and control')
    observed = {tuple(x): y for x, y in zip(xs, ys)}
    control = tuple(control)
    if len(observed) != len(xs) or control not in observed:
        raise ValueError('duplicate coordinates or missing control')
    baseline = observed[control]
    predictions = []
    for coordinates, response in sorted(observed.items()):
        changed = [i for i, value in enumerate(coordinates) if value != control[i]]
        if len(changed) < 2:
            continue
        axes = [tuple(value if j == i else control[j] for j, value in enumerate(coordinates))
                for i in changed]
        missing = [point for point in axes if point not in observed]
        prediction = None if missing else baseline + sum(observed[point] - baseline for point in axes)
        predictions.append({'coordinates': list(coordinates), 'observed': response,
                            'predicted': prediction, 'missing_axes': [list(x) for x in missing],
                            'interaction_residual': None if prediction is None else response - prediction})
    errors = [abs(p['interaction_residual']) for p in predictions if p['predicted'] is not None]
    return {'control': list(control), 'control_response': baseline, 'joint_points': predictions,
            'predicted_points': len(errors), 'mae': mean(errors) if errors else None,
            'scope': 'within_fixture_axis_composition_test_not_query_holdout_or_full_interaction_coverage'}


def clipped_affine_check(train_x, train_y, test_x, test_y):
    """Fit an active affine cost-gap branch, then compare min(0, gap) out of fit."""
    if (not train_x or not test_x or len(train_x) != len(train_y) or len(test_x) != len(test_y)
            or not all(math.isfinite(v) for v in [*train_x, *train_y, *test_x, *test_y])):
        raise ValueError('need finite paired training and evaluation observations')
    if any(v > 0 for v in train_y):
        raise ValueError('positive training responses violate the clipped cost-gap hypothesis')
    active = [(x, y) for x, y in zip(train_x, train_y) if y < 0]
    if len({x for x, _ in active}) < 2:
        return {'identified': False, 'reason': 'need_two_distinct_active_training_coordinates'}
    slope, intercept = linear_regression([x for x, _ in active], [y for _, y in active])
    predictions = [min(0, slope*x + intercept) for x in test_x]
    errors = [abs(y-p) for y, p in zip(test_y, predictions)]
    return {'identified': True, 'slope': slope, 'intercept': intercept,
            'threshold': -intercept/slope if slope else None,
            'active_training_points': len(active),
            'training_mae': mean(abs(y-min(0, slope*x+intercept)) for x, y in zip(train_x, train_y)),
            'predictions': predictions, 'absolute_errors': errors, 'mae': mean(errors),
            'extrapolated': [not min(train_x) <= x <= max(train_x) for x in test_x],
            'scope': 'chosen_clipped_affine_hypothesis_not_automatic_curve_family_or_query_validation'}


def summarize(report):
    points = []
    for source in sorted(report['points'], key=lambda p: p['rows']):
        runs = source.get('evidence', {}).get('runs', [])
        r = runs[0] if len(runs) == 1 else {}
        valid = (source['runner_returncode'] == 0 and not source.get('analysis_error')
                 and r.get('delta') is not None and not r.get('exclusions')
                 and r.get('search', {}).get('complete'))
        point = {'rows': source['rows'], 'audited': bool(valid), 'samples': {},
                 'search': None, 'plan_change': r.get('plan_change'),
                 'target_ready': r.get('target_ready'),
                 'target_root_inserted': r.get('target_root_inserted'),
                 'target_attempts': r.get('target_attempts') if valid else None,
                 'target_stages': (r.get('rule_deltas') or {}).get(report['manifest']['rule_hash'], {}) if valid else None,
                 'physical_search_delta': r['search']['delta'] if valid else None,
                 'analysis_error': source.get('analysis_error'), 'exclusions': r.get('exclusions', [])}
        for field, sign in [('delta_planning_ms', 1), ('delta_execution_ms', -1),
                            ('delta_plan_plus_execution_ms', -1)]:
            point['samples'][field] = [sign*p[field] for p in r.get('pairs', [])
                                      if valid and not p.get('exclusions') and p[field] is not None]
        if valid:
            point['search'] = {**r['delta'], 'costed': r['search']['delta'].get('cost:costed', 0)}
        points.append(point)
    declared = sorted(report['manifest']['scales'])
    complete = ([p['rows'] for p in points] == declared and all(p['audited'] for p in points))
    result = {'rule_hash': report['manifest']['rule_hash'], 'points': points,
              'complete_design': complete, 'execution_model': None, 'search_constant': None}
    if complete:
        result['search_constant'] = {k: len({p['search'][k] for p in points}) == 1
                                     for k in ('attempts', 'memo_expressions', 'costed')}
    if complete and len(points) >= 3 and all(len(p['samples']['delta_execution_ms']) == report['manifest']['timing_repeats'] for p in points):
        xs = [p['rows'] for p in points]
        ys = [median(p['samples']['delta_execution_ms']) for p in points]
        result['execution_model'] = model_check(xs, ys)
        result['execution_model'].update(endpoint_change=ys[-1]-ys[0],
                                         adjacent_changes=[b-a for a, b in zip(ys, ys[1:])])
    return result


def render(curves, output, font):
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    metrics = [('delta_execution_ms', '执行时间节省（毫秒）'),
               ('delta_plan_plus_execution_ms', '单次规划＋执行净收益（毫秒）'),
               ('delta_planning_ms', '规划时间净增（毫秒）')]
    for label, curve in curves.items():
        xs = [p['rows'] for p in curve['points']]
        for ax, (field, title) in zip(axes.flat, metrics):
            values = [p['samples'][field] for p in curve['points']]
            line, = ax.plot(xs, [median(v) if v else math.nan for v in values], 'o-', label=label)
            for x, samples in zip(xs, values):
                ax.scatter([x]*len(samples), samples, alpha=.18, s=10, color=line.get_color())
            ax.set_title(title)
        axes[1, 1].plot(xs, [p['search']['memo_expressions'] if p['search'] else math.nan
                            for p in curve['points']], 'o-', label=label)
    axes[1, 1].set_title('Memo 表达式净增：开启 − 关闭')
    all_x = [p['rows'] for c in curves.values() for p in c['points']]
    for ax in axes.flat:
        ax.set_xscale('log', base=2)
        ax.set_xlim(min(all_x)/1.2, max(all_x)*1.2)
        ax.set_xticks(sorted(set(all_x)), [str(x) for x in sorted(set(all_x))], rotation=20)
        ax.set_xlabel('模板规模参数（对数轴；不是统一的局部输入基数）')
        ax.axhline(0, color='gray', linewidth=.7)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('少数曲线族假设：跨查询机制复核\n散点是重复测量，不是独立查询；连线不是拟合或总体规律；所有规则仍在成本搜索中')
    fig.savefig(output / '跨查询曲线复核.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    metrics = [('target_attempts', '目标规则尝试数'), ('target_ready', '目标规则生成候选数'),
               ('target_root_inserted', '目标规则新根插入数'), ('costed', '全部完成物理计价净增：开启 − 关闭')]
    for ax, (field, title) in zip(axes.flat, metrics):
        for label, curve in curves.items():
            values = [(p['search'].get(field) if field == 'costed' else p.get(field))
                      if p['audited'] else None for p in curve['points']]
            ax.plot([p['rows'] for p in curve['points']],
                    [v if v is not None else math.nan for v in values], 'o-', label=label)
        ax.set_xscale('log', base=2)
        ax.set_xlim(min(all_x)/1.2, max(all_x)*1.2)
        ax.set_xticks(sorted(set(all_x)), [str(x) for x in sorted(set(all_x))], rotation=20)
        ax.set_xlabel('模板规模参数（对数轴）')
        ax.set_title(title)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.axhline(0, color='gray', linewidth=.7)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('搜索阶段响应：尝试不等于新根，更不等于最终收益\n曲线重合表示计数一致；目标在关闭臂没有尝试\n物理计价是整个搜索的条件净差，不是单次规则独占成本')
    fig.savefig(output / '搜索阶段响应.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True, help='Chinese label=响应.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, default=Path('output/fonts/NotoSansCJKsc-Regular.otf'))
    args = parser.parse_args()
    cases = [c.split('=', 1) for c in args.case]
    if any(len(c) != 2 or not all(c) for c in cases) or len({c[0] for c in cases}) != len(cases):
        parser.error('require unique nonempty labels and paths')
    curves = {label: {**summarize(json.loads(Path(path).read_text())), 'source': str(Path(path).resolve())}
              for label, path in cases}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / '曲线复核.json').write_text(json.dumps(curves, ensure_ascii=False, indent=2) + '\n')
    render(curves, args.output, args.font)


if __name__ == '__main__':
    main()
