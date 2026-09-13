#!/usr/bin/env python3
"""Rule-blind, leave-application-out checks of generic pre-evaluation dimensions."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean
import zlib

from ml_orca.experiments.plot_stats_sweep import chinese_plotting
from ml_orca.experiments.profile_rule_conditions import collect


AXES = {'nodes': '总树节点数', 'relational_nodes': '关系节点数', 'scalar_nodes': '标量节点数',
        'depth': '树深度', 'root_rows': '已缓存根基数',
        'template_nodes': '源模板规模', 'template_constraints': '模板约束数量',
        'template_joint': '模板＋标量规模'}
TARGETS = {'match_log': ('match_log_sum', 'match_log_square_sum'),
           'ready': ('ready_cbo', 'ready_cbo')}


def template_cells(cells, graph):
    """Join by canonical identity, but predict using shared coarse RuleIR features only."""
    features = {}
    for node in graph['nodes']:
        key, values = node['rule_hash'], node.get('template_features', {})
        if key in features or any(type(values.get(k)) is not int or values[k] < minimum
                                  for k, minimum in [('source_nodes', 1), ('constraint_count', 0)]):
            raise ValueError('duplicate rule identity or missing/invalid RuleIR features')
        features[key] = values
    result = []
    bucket = lambda n: 1 << (n.bit_length()-1) if n else 0
    for c in cells:
        if c['axis'] not in {'nodes', 'scalar_nodes'} or not c['complete'] or not c.get('evaluated'):
            continue
        if c['rule_hash'] not in features:
            raise ValueError('observed rule missing from template graph')
        f = features[c['rule_hash']]
        size, constraints = bucket(f['source_nodes']), bucket(f['constraint_count'])
        dimensions = [('template_nodes', size), ('template_constraints', constraints)] if c['axis'] == 'nodes' else [
            ('template_joint', (size, constraints, c['band']) if c['band'] is not None else None)]
        result.extend({**c, 'axis': axis, 'band': band} for axis, band in dimensions)
    return result


def pooled_cells(cells):
    """Remove rule identity, keeping query/application only for weighting and splits."""
    groups = defaultdict(Counter)
    for c in cells:
        if not c['complete'] or not c.get('evaluated'):
            continue
        g = groups[c['case_id'], c['dataset'], c['axis'], c['band']]
        for field in ('evaluated', 'ready_cbo', 'match_log_sum', 'match_log_square_sum'):
            g[field] += c.get(field, 0)
    return [{**dict(v), 'case_id': q, 'dataset': app, 'axis': axis, 'band': band}
            for (q, app, axis, band), v in groups.items()]


def fit_dimensions(cells):
    """Freeze query-equal bin means; identities are split guards, never predictors."""
    models = []
    for axis in AXES:
        data = [c for c in cells if c['axis'] == axis]
        totals = Counter()
        for c in data:
            totals[c['case_id']] += c['evaluated']
        if not totals or any(n <= 0 for n in totals.values()):
            continue
        for target, (total, _) in TARGETS.items():
            baseline = sum(c[total] / totals[c['case_id']] for c in data) / len(totals)
            sums, weights = Counter(), Counter()
            for c in data:
                band = tuple(c['band']) if isinstance(c['band'], list) else c['band']
                if band is not None:
                    sums[band] += c[total] / totals[c['case_id']]
                    weights[band] += c['evaluated'] / totals[c['case_id']]
            models.append({'axis': axis, 'target': target, 'baseline': baseline,
                           'bins': [{'band': b, 'prediction': sums[b] / w} for b, w in weights.items()]})
    return {'training_queries': sorted({c['case_id'] for c in cells}),
            'training_applications': sorted({c['dataset'] for c in cells}), 'models': models}


def score_dimensions(frozen, cells):
    """Score without fitting; reject query leakage and retain unknown-bin fallback."""
    if set(frozen['training_queries']) & {c['case_id'] for c in cells}:
        raise ValueError('training and test query identities overlap')
    results = []
    for model in frozen['models']:
        axis, target = model['axis'], model['target']
        total, square = TARGETS[target]
        data = [c for c in cells if c['axis'] == axis]
        totals = Counter()
        for c in data:
            totals[c['case_id']] += c['evaluated']
        predictions = {(tuple(b['band']) if isinstance(b['band'], list) else b['band']): b['prediction']
                       for b in model['bins']}
        errors = defaultdict(Counter)
        for c in data:
            band = tuple(c['band']) if isinstance(c['band'], list) else c['band']
            known = band is not None and band in predictions
            prediction = predictions[band] if known else model['baseline']
            q = errors[c['case_id']]
            for name, p in [('baseline', model['baseline']), ('dimension', prediction)]:
                q[name + '_mse'] += max(0, c[square] - 2*p*c[total] + c['evaluated']*p*p) / totals[c['case_id']]
                q[name + '_prediction'] += c['evaluated'] * p / totals[c['case_id']]
                if target == 'ready':
                    # Keep rare positives visible: all-negative predictions can have tiny Brier loss.
                    q[name + '_positive_loss'] += c[total] * (1-p)**2 / totals[c['case_id']]
                    q[name + '_negative_loss'] += (c['evaluated']-c[total]) * p*p / totals[c['case_id']]
            q['observed_mean'] += c[total] / totals[c['case_id']]
            q['known_fraction'] += c['evaluated'] * known / totals[c['case_id']]
        queries = [{'case_id': q, **dict(v)} for q, v in sorted(errors.items())]
        if not queries:
            continue
        base, loss = (mean(q[k] for q in queries) for k in ('baseline_mse', 'dimension_mse'))
        result = {'axis': axis, 'target': target, 'queries': queries,
                  'baseline_mse': base, 'dimension_mse': loss,
                  'relative_mse_reduction': 1-loss/base if base > 0 else None,
                  'known_fraction': mean(q['known_fraction'] for q in queries)}
        if target == 'ready':
            rate = mean(q['observed_mean'] for q in queries)
            result['positive_rate'] = rate
            for name in ('baseline', 'dimension'):
                result[name + '_prediction'] = mean(q[name + '_prediction'] for q in queries)
                result[name + '_positive_brier'] = mean(q[name + '_positive_loss'] for q in queries) / rate if rate > 0 else None
                result[name + '_negative_brier'] = mean(q[name + '_negative_loss'] for q in queries) / (1-rate) if rate < 1 else None
        results.append(result)
    return results


def validate(cells):
    """Exact per-attempt squared error from sufficient statistics; queries weigh equally."""
    results = []
    for axis in AXES:
        data = [{**c, 'band': tuple(c['band']) if isinstance(c['band'], list) else c['band']}
                for c in cells if c['axis'] == axis]
        apps = sorted({c['dataset'] for c in data})
        if len(apps) < 2:
            continue
        for target, (total, square) in TARGETS.items():
            folds = []
            for app in apps:
                train = [c for c in data if c['dataset'] != app]
                test = [c for c in data if c['dataset'] == app]
                frozen = fit_dimensions(train)
                scored = next(r for r in score_dimensions(frozen, test) if r['target'] == target)
                folds.append({'heldout_application': app, 'training_queries': len(frozen['training_queries']),
                              'queries': scored['queries']})
            queries = [q for f in folds for q in f['queries']]
            base = mean(q['baseline_mse'] for q in queries)
            loss = mean(q['dimension_mse'] for q in queries)
            results.append({'axis': axis, 'target': target, 'queries': len(queries), 'applications': len(apps),
                            'baseline_mse': base, 'dimension_mse': loss,
                            'relative_mse_reduction': 1-loss/base if base > 0 else None,
                            'known_fraction': mean(q['known_fraction'] for q in queries), 'folds': folds})
    return results


def render(report, output, font):
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
    for ax, target, title in zip(axes, ('match_log', 'ready'), ('匹配耗时对数误差', '候选生成概率误差')):
        rows = [r for r in report['validation'] if r['target'] == target]
        values = [100*r['relative_mse_reduction'] if r['relative_mse_reduction'] is not None
                  and r['known_fraction'] > 0 else math.nan for r in rows]
        ax.bar([AXES[r['axis']] for r in rows], values, color='#4477aa')
        for i, value in enumerate(values):
            if math.isfinite(value):
                label = f'{value:.3f}%' if target == 'ready' else f'{value:.1f}%'
                ax.annotate(label, (i, value), ha='center', xytext=(0, 4), textcoords='offset points')
            else:
                ax.text(i, 0, '无可用\n观测', ha='center', va='bottom')
        ax.set_title(title)
        ax.set_ylabel('相对常数基线的均方误差下降（%；负值更差）')
        ax.axhline(0, color='gray', linewidth=.8)
    rows = [r for r in report['validation'] if r['target'] == 'ready']
    axes[2].bar([AXES[r['axis']] for r in rows], [100*r['known_fraction'] for r in rows], color='#228833')
    axes[2].set_title('测试输入落入已知训练分桶的比例')
    axes[2].set_ylabel('查询等权比例（%）；未知退回训练常数')
    axes[2].set_ylim(0, 110)
    for ax in axes:
        ax.tick_params(axis='x', labelrotation=40)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('通用维度检验：每次留出整个应用，模型不使用规则或查询标识\n全部完整查询中的已发出尝试；回顾性离线预测，不是因果效应或部署性能')
    fig.savefig(output / '通用维度检验.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rule-graph', type=Path, help='production audit graph with RuleIR template_features')
    parser.add_argument('--font', type=Path, default=Path('output/fonts/NotoSansCJKsc-Regular.otf'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    graph_bytes = args.rule_graph.read_bytes() if args.rule_graph else None
    graph = json.loads(graph_bytes) if graph_bytes is not None else None
    if graph is not None:
        template_cells([], graph)  # Validate metadata before the expensive trace audit.
    design = {'axes': {k: v for k, v in AXES.items() if graph is not None or not k.startswith('template_')},
              'rule_graph': {'path': str(args.rule_graph.resolve()), 'crc32': f'{zlib.crc32(graph_bytes):08x}'} if graph is not None else None,
              'targets': {'match_log': 'log1p(match_us)', 'ready': 'status == ready_cbo'},
              'model': 'dyadic_bin_training_mean_with_global_fallback; joint=(source_template_nodes,constraint_count,scalar_nodes)',
              'validation': 'leave_application_out_query_equal_per_attempt_squared_error',
              'excluded_features': ['rule_id', 'rule_hash', 'query_id', 'application_id', 'group_id',
                                    'fingerprints', 'post_match_bindings', 'failed_constraint', 'future_graph_edges'],
              'limitations': ['retrospective_previously_inspected_traces', 'empty_tables', 'traced_timing',
                              'dispatched_attempts_only', 'failed_prefixes_excluded_from_fitting',
                              'no_causal_or_unseen_rule_claim', 'single_dimensions_not_adjusted_for_rule_composition']}
    (args.output / 'design.json').write_text(json.dumps(design, ensure_ascii=False, indent=2) + '\n')
    evidence = collect(args.results)
    raw = evidence['query_cells']
    cells = pooled_cells(raw + template_cells(raw, graph) if graph is not None else raw)
    report = {'design': design, 'sources': evidence['sources'], 'rule_crc32': evidence['rule_crc32'],
              'queries': evidence['queries'], 'cells': cells, 'validation': validate(cells),
              'complete_rule_count': len({c['rule_hash'] for c in evidence['query_cells'] if c['complete']}),
              'analysis_errors': sum('analysis_error' in q for q in evidence['queries'])}
    if graph is not None:
        observed = {c['rule_hash'] for c in raw if c['complete'] and c.get('evaluated')}
        # Report signature collisions: coarse structural dimensions must not silently become rule IDs.
        signatures = Counter((n['template_features']['source_nodes'].bit_length(),
                              n['template_features']['constraint_count'].bit_length())
                             for n in graph['nodes'] if n['rule_hash'] in observed)
        report['template_support'] = {'observed_rules': len(observed), 'coarse_signatures': len(signatures),
                                      'single_rule_signatures': sum(v == 1 for v in signatures.values()),
                                      'rules_in_shared_signatures': sum(v for v in signatures.values() if v > 1)}
    (args.output / '通用维度检验.json').write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':')) + '\n')
    render(report, args.output, args.font)
    print(json.dumps({'complete_queries': sum(q['complete'] for q in report['queries']),
                      'rules': report['complete_rule_count'], 'analysis_errors': report['analysis_errors']}))


if __name__ == '__main__':
    main()
