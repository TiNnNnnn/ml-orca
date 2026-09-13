#!/usr/bin/env python3
"""Describe rule-local conditional responses from frozen corpus traces; no SQL replay."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re
from statistics import mean, median, quantiles

from ml_orca.experiments.plot_stats_sweep import chinese_plotting
from ml_orca.trace.profile_rule_candidates import (candidate_evidence, candidate_state, cost_lifecycle_evidence,
                                     target_root_costs, TIMES)
from ml_orca.collect.run_workload_comparison import trace_records, dsl_observability


TRACE_KINDS = re.compile(r'DSL_TRACE \{\s*"kind"\s*:\s*"(?:candidate_context|rule_candidate|'
                         r'rule_candidate_outcome|rule_summary|experiment_outcome)"')
PHYSICAL_KINDS = re.compile(r'DSL_TRACE \{\s*"kind"\s*:\s*"(?:cost_candidate|cost_lifecycle)"')
CONSTRUCTED = {'instantiate_rejected', 'duplicate', 'ready_cbo', 'budget_exhausted'}
METRICS = {'ready_rate': ('ready_cbo', 'evaluated'),
           'duplicate_rate': ('duplicate', 'evaluated'),
           'construction_us': ('construction_us', 'construction_attempts'),
           'memo_growth': ('direct_insertions', 'evaluated')}


def condition_bins(row):
    state = candidate_state(row)
    shape = state['features']['source_shape'] or {}
    known = (state['captured'] and shape.get('complete') is True and shape.get('pattern_nodes') == 0)
    values = {key: shape.get(key) if known else None for key in ('nodes', 'depth', 'scalar_nodes')}
    nodes, scalar = values['nodes'], values['scalar_nodes']
    values['relational_nodes'] = nodes - scalar if type(nodes) is int and type(scalar) is int and 0 <= scalar <= nodes else None
    values['root_rows'] = state['features']['root']['rows']
    result = {}
    for key, value in values.items():
        # Dyadic bins are display resolution, not learned thresholds or independent interventions.
        if type(value) is int and value > 0:
            result[key] = 1 << (value.bit_length() - 1)
        elif type(value) in (int, float) and math.isfinite(value) and value >= 0:
            result[key] = 0 if value == 0 else 2 ** (math.frexp(value)[1] - 1)
        else:
            result[key] = None
    return result, state


def query_cells(rows, case_id, dataset, complete):
    cells = defaultdict(Counter)
    availability = Counter()
    for row in rows:
        bands, state = condition_bins(row)
        availability['attempts'] += 1
        availability['captured'] += state['captured']
        availability['root_rows_available'] += state['features']['root']['rows_available']
        availability['any_child_rows_available'] += any(c['node']['rows_available'] for c in state['features']['children'])
        availability['source_shape_known'] += bands['nodes'] is not None
        for axis, band in bands.items():
            cell = cells[row['rule_hash'], axis, band]
            cell['attempts'] += 1
            cell['evaluated'] += row.get('evaluated') is True
            if row.get('evaluated') is True:
                cost = math.log1p(row['match_us'])
                cell['match_log_sum'] += cost
                cell['match_log_square_sum'] += cost * cost
            cell[row['status']] += 1
            if row['status'] in CONSTRUCTED:
                cell['construction_attempts'] += 1
                cell['construction_us'] += row['instantiate_us']
            if row.get('direct_insertions') is not None:
                cell['direct_insertions'] += row['direct_insertions']
            if row['status'] == 'ready_cbo' and row.get('direct_insertions') is None:
                cell['unresolved_insertions'] += 1
            for field in TIMES:
                cell[field] += row[field]
    return [{**dict(counts), 'case_id': case_id, 'dataset': dataset, 'complete': complete,
             'rule_hash': rule, 'axis': axis, 'band': band}
            for (rule, axis, band), counts in cells.items()], dict(availability)


def aggregate_cells(cells):
    groups = defaultdict(list)
    for cell in cells:
        groups[cell['complete'], cell['rule_hash'], cell['axis'], cell['band']].append(cell)
    points = []
    for (complete, rule, axis, band), group in sorted(groups.items(), key=lambda p: str(p[0])):
        point = {'complete': complete, 'rule_hash': rule, 'axis': axis, 'band': band,
                 'queries': len(group), 'applications': len({g['dataset'] for g in group}),
                 'attempts': sum(g['attempts'] for g in group), 'metrics': {}}
        for metric, (numerator, denominator) in METRICS.items():
            eligible = [g for g in group if g.get(denominator, 0) > 0
                        and (metric != 'memo_growth' or not g.get('unresolved_insertions', 0))]
            values = [g.get(numerator, 0) / g[denominator] for g in eligible]
            if not values:
                point['metrics'][metric] = None
                continue
            quartiles = quantiles(values, n=4, method='inclusive') if len(values) > 1 else values * 3
            point['metrics'][metric] = {
                'queries': len(eligible), 'applications': len({g['dataset'] for g in eligible}),
                'query_mean': mean(values), 'query_median': median(values),
                'query_q25': quartiles[0], 'query_q75': quartiles[2],
                'pooled': sum(g.get(numerator, 0) for g in eligible) / sum(g[denominator] for g in eligible),
                'denominator': sum(g[denominator] for g in eligible)}
        points.append(point)
    return points


def rule_effects(run, audit, edges):
    """Account for work and exact-root physical evidence, never net causal benefit."""
    lifecycle = cost_lifecycle_evidence(run)
    problems = sorted(set(audit['exclusions'] + lifecycle['exclusions']))
    result = {'complete': not problems, 'exclusions': problems, 'rules': [],
              'scope': 'direct_inserted_root_physical_origin_and_emitted_source_root_consumption',
              'not_measured': ['counterfactual_net_benefit', 'execution_speedup',
                               'all_descendant_or_child_rule_effects', 'physical_search_time'],
              'observed_cost_status_counts': dict(Counter(e['status'] for e in run['cost_events']))}
    if problems:
        return result  # Missing/truncated physical evidence is not zero benefit.
    grouped = defaultdict(list)
    for row in audit['rows']:
        grouped[row['rule_hash']].append(row)
    best = {e['candidate_sequence'] for e in lifecycle['events'] if e['status'] == 'best_updated'}
    for rule, rows in sorted(grouped.items()):
        roots = target_root_costs(run, rows)
        comparisons = []
        for comparison in roots['comparisons']:
            gap, incumbent = comparison['cost_gap'], comparison['context_best_cost_at_event']
            if gap is None:
                status = 'no_incumbent'
            elif math.isclose(comparison['cost'], incumbent, rel_tol=1e-8, abs_tol=1e-8):
                status = 'equal_at_trace_precision'
            else:
                status = 'improved' if gap < 0 else 'worse'
            comparisons.append({**comparison, 'status': status,
                                'became_best': comparison['sequence'] in best,
                                'relative_reduction': -gap / incumbent
                                if gap is not None and incumbent > 0 else None})
        result['rules'].append({
            'rule_hash': rule, 'attempts': len(rows),
            'evaluation_us': sum(r[t] for r in rows for t in TIMES),
            'nonready_evaluation_us': sum(r[t] for r in rows if r['status'] != 'ready_cbo' for t in TIMES),
            'stages': dict(Counter(r['status'] for r in rows)),
            'direct_insertions': sum(r['direct_insertions'] or 0 for r in rows),
            'memo_outcomes': dict(Counter(r['memo_outcome']['status'] for r in rows if r['memo_outcome'])),
            'outgoing_consumptions': dict(Counter(e['dst_rule'] for e in edges if e['src_rule'] == rule)),
            'ancestral': target_root_costs(run, rows, include_ancestors=True)
                if all('origin_chain' in e for e in run['cost_events']) else None,
            **roots, 'comparisons': comparisons})
    return result


def collect(roots, physical_effects=False):
    cells, queries, seen = [], [], set()
    rule_crc = None
    for root in roots:
        manifest = json.loads((root / 'manifest.json').read_text())
        if rule_crc is not None and manifest['rule_crc32'] != rule_crc:
            raise ValueError('cannot combine different rule libraries')
        rule_crc = manifest['rule_crc32']
        identities = {c['case_id']: c for d in manifest['datasets'] for c in d['cases']}
        report = json.loads((root / 'summary.json').read_text())
        for dataset in report['datasets']:
            for saved in dataset['runs']:
                case_id = saved['case_id']
                if case_id in seen or saved['query_crc32'] != identities[case_id]['query_crc32']:
                    raise ValueError(f'duplicate or mismatched query: {case_id}')
                seen.add(case_id)
                path = root / dataset['dataset'] / 'logs' / (case_id.split(':')[1] + '.log.gz')
                query = {'case_id': case_id, 'complete': False, 'artifact': str(path),
                         'original_exclusions': saved['exclusions']}
                if not path.exists():
                    query['analysis_error'] = 'missing_trace'
                    queries.append(query)
                    continue
                # Keep raw logs on disk; expanded contexts and physical event streams
                # are not copied into new artifacts.
                with gzip.open(path, 'rt', errors='replace') as stream:
                    records = trace_records(''.join(line for line in stream if TRACE_KINDS.search(line)
                                            or (physical_effects and PHYSICAL_KINDS.search(line))))
                run = {'plan_rc': saved['returncode'], 'rows_rc': 0,
                       'error': 'timeout' if 'plan_timeout' in saved['exclusions'] else '',
                       'fallback': bool(saved['fallback']),
                       'optimizer': 'postgres' if saved['fallback'] else 'pg_orca',
                       'candidate_events': [r for r in records if r['kind'] in {'rule_candidate', 'rule_candidate_outcome'}],
                       'experiment_outcomes': [r for r in records if r['kind'] == 'experiment_outcome'],
                       'dsl_observability': dsl_observability(records)}
                try:
                    audit = candidate_evidence(run)
                    if audit['attempts'] != saved['attempts'] or audit['complete'] != saved['complete']:
                        raise ValueError('reanalysis differs from frozen candidate audit')
                    current, availability = query_cells(audit['rows'], case_id, dataset['dataset'], audit['complete'])
                    cells.extend(current)
                    query.update(complete=audit['complete'], availability=availability)
                    if physical_effects:
                        run.update(cost_events=[r for r in records if r['kind'] == 'cost_candidate'],
                                   cost_lifecycle_events=[r for r in records if r['kind'] == 'cost_lifecycle'])
                        query['effects'] = rule_effects(run, audit, saved['edges'])
                except ValueError as error:
                    query['analysis_error'] = str(error)
                queries.append(query)
                if len(queries) % 20 == 0:
                    print(f'{len(queries)} queries analyzed', flush=True)
    return {'scope': 'descriptive_pre_evaluation_shape_conditioned_cbo_attempts_not_causal',
            'sources': [str(r) for r in roots], 'rule_crc32': rule_crc,
            'weighting': 'query_equal_within_observed_bin_and_rule; pooled_attempt_rate_as_sensitivity',
            'intervals': 'query_interquartile_range_not_confidence_interval',
            'feature_axes': 'dyadic_nodes_depth_scalar_relational_nodes_and_cached_root_rows; missing_is_unknown',
            'selection': 'posthoc_discovery; no_heldout_generalization_test',
            'limitations': ['empty_tables', 'generic_parameters', 'traced_timing', 'one_factor_marginals_not_adjusted_effects',
                            'no_stats_backfill', 'no_final_plan_filter', 'memo_growth_excludes_future_rule_chains'],
            'queries': queries, 'query_cells': cells, 'points': aggregate_cells(cells)}


def render_effects(report, output, labels, font):
    plt = chinese_plotting(font)
    queries = [q for q in report['queries'] if q.get('effects', {}).get('complete')]
    fig, axes = plt.subplots(len(labels), 3, figsize=(16, 2.8 * len(labels)), squeeze=False,
                             layout='constrained')
    for row, (rule, label) in enumerate(labels.items()):
        entries = [r for q in queries for r in q['effects']['rules'] if r['rule_hash'] == rule]
        if not entries:
            for ax in axes[row]:
                ax.text(.5, .5, '无完整证据', ha='center', transform=ax.transAxes)
            continue
        # Descriptive totals, not query-independent population rates or additive savings.
        total = lambda field: sum(e[field] for e in entries)
        cost_counts, lifecycle, comparisons = Counter(), Counter(), Counter()
        for entry in entries:
            cost_counts.update(entry['cost_status_counts'])
            lifecycle.update(entry['lifecycle_status_counts'])
            comparisons.update(c['status'] for c in entry['comparisons'])
        panels = [
            (['未生成候选', '生成候选'],
             [total('nonready_evaluation_us') / 1000,
              (total('evaluation_us') - total('nonready_evaluation_us')) / 1000], '累计求值耗时（毫秒，带埋点）'),
            (['直接入库', '物理入口', '来源消费'],
             [total('direct_insertions'), sum(cost_counts.values()),
              sum(sum(e['outgoing_consumptions'].values()) for e in entries)], '搜索工作计数（各项不相加）'),
            (['无前任', '改善', '近似相等', '更贵', '最终采用'],
             [comparisons[k] for k in ('no_incumbent', 'improved', 'equal_at_trace_precision', 'worse')]
             + [lifecycle['selected_plan']], '直接物理代价比较；采用为另一个标签')]
        for col, (names, values, title) in enumerate(panels):
            ax = axes[row, col]
            ax.bar(names, values, color=['#cc6677', '#4477aa', '#999999', '#ee7733', '#228833'][:len(names)])
            for i, value in enumerate(values):
                ax.text(i, value, f'{value:.1f}' if col == 0 else str(value), ha='center', va='bottom', fontsize=9)
            ax.set_ylim(0, max(values) * 1.22 or 1)
            ax.set_title(f'{label}（{len(entries)}查）\n{title}', fontsize=10)
            ax.tick_params(axis='x', labelrotation=15)
            if col == 2 and not cost_counts:
                ax.text(.5, .5, '无直接物理来源关联\n不能推断无间接收益', ha='center',
                        transform=ax.transAxes, color='#666666')
    fig.suptitle(f'规则开销与局部正收益证据｜{len(queries)} 条完整物理审计\n'
                 '不是开关规则的净收益；不累计局部代价改善；零直接关联不代表无间接收益', fontsize=12)
    fig.savefig(output / '规则开销与局部收益.png', dpi=150)
    plt.close(fig)
    if not any(e.get('ancestral') is not None for q in queries for e in q['effects']['rules']):
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for index, (title, field, statuses) in enumerate([
            ('关联的物理成本入口事件', 'cost_status_counts', None),
            ('关联的完整物理计价事件', 'cost_status_counts', ['costed'])]):
        for offset, ancestry, label in [(-.18, False, '紧邻来源'), (.18, True, '含已记录祖先')]:
            values = []
            for rule in labels:
                entries = [e for q in queries for e in q['effects']['rules']
                           if e['rule_hash'] == rule and e.get('ancestral') is not None]
                counts = Counter()
                for entry in entries:
                    counts.update((entry['ancestral'] if ancestry else entry)[field])
                values.append(sum(counts.values()) if statuses is None else sum(counts[s] for s in statuses))
            positions = [i + offset for i in range(len(labels))]
            axes[index].bar(positions, values, width=.36, label=label)
            for x, value in zip(positions, values):
                axes[index].text(x, value, str(value), ha='center', va='bottom', fontsize=9)
        axes[index].set_xticks(range(len(labels)), list(labels.values()), rotation=25)
        axes[index].set_title(title)
        axes[index].margins(y=.15)
        axes[index].legend()
    fig.suptitle('补全原生逻辑转换之间的真实来源链\n祖先关联包含紧邻来源；不是互斥归因，不覆盖全部子输入消费，不表示净收益')
    fig.savefig(output / '来源链补全.png', dpi=150)
    plt.close(fig)


def render(report, output, labels, font):
    plt = chinese_plotting(font)
    metric_labels = ['每次求值生成候选比例', '每次求值构造重复比例', '每次构造平均耗时（微秒）', '每次求值的直接入库新增量']
    for axis_name, name in [('nodes', '树节点数'), ('depth', '树深度')]:
        fig, axes = plt.subplots(len(labels), 4, figsize=(19, 3.1 * len(labels)), squeeze=False,
                                 layout='constrained')
        for row, (rule, label) in enumerate(labels.items()):
            points = sorted([p for p in report['points'] if p['complete'] and p['rule_hash'] == rule
                             and p['axis'] == axis_name and p['band'] is not None], key=lambda p: p['band'])
            for col, metric in enumerate(METRICS):
                ax = axes[row, col]
                data = [p for p in points if p['metrics'][metric] is not None]
                if not data:
                    ax.text(.5, .5, '无可用样本', transform=ax.transAxes, ha='center')
                    continue
                x = [p['band'] for p in data]
                measures = [p['metrics'][metric] for p in data]
                ax.fill_between(x, [m['query_q25'] for m in measures], [m['query_q75'] for m in measures],
                                color='#4477aa', alpha=.18, label='查询间四分位范围')
                ax.plot(x, [m['query_mean'] for m in measures], 'o-', color='#4477aa', label='每查询等权')
                ax.plot(x, [m['pooled'] for m in measures], 'x--', color='#ee7733', label='合并次数加权')
                for xv, m in zip(x, measures):
                    ax.annotate(f'{m["queries"]}查/{m["applications"]}库', (xv, m['query_mean']),
                                xytext=(0, 6), textcoords='offset points', fontsize=7, ha='center')
                ax.set_xscale('log', base=2)
                ax.set_xticks(x, [f'{v}～{2*v-1}' for v in x], rotation=30)
                ax.set_title(f'{label}｜{metric_labels[col]}', fontsize=10)
                ax.set_xlabel(f'求值前{name}区间')
                ax.grid(alpha=.2)
                if col < 2:
                    ax.set_ylim(-.04, 1.15)
                else:
                    ax.set_ylim(0, max(max(m['query_mean'], m['pooled'], m['query_q75'])
                                       for m in measures) * 1.2 or 1)
                if row == 0 and col == 0:
                    ax.legend(fontsize=7)
        fig.suptitle('完整查询的规则条件响应｜节点含标量与关系算子\n'
                     '探索性分桶连线；非因果曲线；阴影不是置信区间；缺失形状不入横轴；耗时带埋点', fontsize=12)
        fig.savefig(output / f'{name}条件分布.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for index, complete in enumerate((True, False)):
        queries = [q for q in report['queries'] if q['complete'] == complete]
        totals = Counter()
        for q in queries:
            totals.update(q.get('availability', {}))
        fields = ['attempts', 'captured', 'source_shape_known', 'root_rows_available', 'any_child_rows_available']
        axes[index].bar(['已记录尝试', '求值前状态', '完整树形', '已有根基数', '已有子基数'], [totals[k] for k in fields])
        axes[index].set_title(('完整查询' if complete else '失败查询的已观测前缀') + f'：{len(queries)} 条')
        axes[index].tick_params(axis='x', labelrotation=25)
        for i, k in enumerate(fields):
            axes[index].text(i, totals[k], str(totals[k]), ha='center', va='bottom', fontsize=9)
    fig.suptitle('观测支持度：缺失基数不补零，失败前缀不冒充完整分布')
    fig.savefig(output / '状态可用性.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rule', action='append', required=True, help='HASH=display label')
    parser.add_argument('--physical-effects', action='store_true', help='audit direct-root physical work and local cost improvements')
    parser.add_argument('--font', type=Path, default=Path('output/fonts/NotoSansCJKsc-Regular.otf'))
    args = parser.parse_args()
    labels = dict(r.split('=', 1) for r in args.rule)
    if not all(re.fullmatch('[0-9a-f]{16}', h) and label for h, label in labels.items()):
        parser.error('invalid rule hash or empty label')
    args.output.mkdir(parents=True, exist_ok=False)
    report = collect(args.results, args.physical_effects)
    report['plotted_rules'] = labels
    (args.output / '条件分布.json').write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':')) + '\n')
    render(report, args.output, labels, args.font)
    if args.physical_effects:
        render_effects(report, args.output, labels, args.font)
    print(json.dumps({'queries': len(report['queries']), 'analysis_errors': sum('analysis_error' in q for q in report['queries']),
                      'complete': sum(q['complete'] for q in report['queries']), 'cells': len(report['query_cells'])}))


if __name__ == '__main__':
    main()
