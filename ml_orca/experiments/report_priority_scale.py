"""Audit and plot measured rule-count/ordering controls; never fill failed cells."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from statistics import median

from ml_orca.common.artifacts import artifact_snapshot, read_snapshot
from ml_orca.collect.run_workload_comparison import plan_tree, timing_plan, timing_schedule
from ml_orca.experiments.plot_stats_sweep import chinese_plotting

ARM = re.compile(r's([0-9]+)-(default|reverse|random)')
COST = re.compile(r'\[OPT\]: stage ([0-9]+) completed in ([0-9]+)ms,\s+plan with cost ([0-9eE+.-]+) was found')


def optimizer_stage_costs(trace):
    rows = [dict(stage=int(stage), elapsed_ms=int(elapsed), cost=float(cost))
            for stage, elapsed, cost in COST.findall(trace)]
    if any(not math.isfinite(r['cost']) or r['cost'] < 0 for r in rows):
        raise ValueError('invalid optimizer stage cost')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--run', type=Path,
                        help='alternate completed comparison root using this experiment policy manifest')
    parser.add_argument('--output', type=Path,
                        help='report directory; defaults to --experiment')
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    output = args.output or args.experiment
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.experiment / 'manifest.json').read_bytes())
    for receipt in manifest['identities'].values():
        read_snapshot(receipt)
    expected_arms = [f's{s:03d}-{arm}' for s in manifest['sizes']
                     for arm in ('default', 'reverse', 'random')]
    records, missing, receipts = [], [], {}
    if args.run:
        cases = [(path.parent.parent.name, path.parent.name, path.parent)
                 for path in sorted(args.run.glob('*/*/comparison.json'))]
    else:
        cases = [(manifest['workload'], str(query),
                  args.experiment / 'run' / manifest['workload'] / str(query))
                 for query in manifest['queries']]
    for workload, query, directory in cases:
        comparison = directory / 'comparison.json'
        if not comparison.is_file():
            missing.append(dict(query=query, reason='comparison_missing'))
            continue
        data = json.loads(comparison.read_bytes())
        profile = data['policy_comparison']
        if profile['arms'] != expected_arms or len(profile['scenarios']) != 1:
            raise ValueError(f'{query}: unexpected policy experiment shape')
        samples = profile['timing']['samples']
        expected = list(timing_schedule(1, profile['timing']['repeats'], profile['timing']['warmups'],
                                        profile['timing']['seed'], expected_arms))
        if [(s['phase'], s['block'], s['scenario'], s['arm']) for s in samples] != expected:
            raise ValueError(f'{query}: incomplete timing schedule')
        rows = []
        for arm in expected_arms:
            size, order = ARM.fullmatch(arm).groups()
            run = profile['scenarios'][0]['arms'][arm]
            trace_path, plan_path = directory / f'profile-0.{arm}.trace', directory / f'profile-0.{arm}.plan.json'
            receipts.update(artifact_snapshot({f'{query}:{arm}:trace': trace_path,
                                                f'{query}:{arm}:plan': plan_path}))
            errors = []
            if run['plan_rc'] or run['rows_rc'] or run['optimizer'] != 'pg_orca':
                errors.append('diagnostic_failed')
            if data['postgres_oracle']['mode_result_equal'].get(f'policy:0:{arm}') is not True:
                errors.append('independent_postgres_check_failed')
            stages = optimizer_stage_costs(trace_path.read_text(errors='replace'))
            if not stages:
                errors.append('optimizer_stage_cost_missing')
            rules = run['dsl_observability']['rules']
            if any('memo_inserted_alternatives' not in rule for rule in rules.values()):
                errors.append('memo_outcome_counters_missing')
            measured = [s for s in samples if s['phase'] == 'measurement' and s['arm'] == arm]
            bad = [s['sequence'] for s in measured if s['status'] != 'ok' or s['comparison_exclusions']
                   or not s['diagnostic_plan_matches']]
            if len(measured) != profile['timing']['repeats'] or bad:
                errors.append('timing_failed')
            plan = timing_plan(plan_tree(plan_path.read_text()))
            if plan is None:
                errors.append('plan_missing')
            rows.append(dict(arm=arm, size=int(size), order=order, complete=not errors,
                exclusions=errors, failed_timing_sequences=bad, optimizer_stages=stages,
                final_optimizer_cost=stages[-1]['cost'] if stages else None,
                plan=plan, planning_ms=[s['planning_ms'] for s in measured] if not bad else None,
                planning_ms_median=median(s['planning_ms'] for s in measured) if measured and not bad else None,
                routed_rules=len(rules),
                target_ready_rules=sum(1 for r in rules.values()
                                       if r['generated_alternatives'] or r['duplicate_alternatives']),
                memo_expanding_rules=sum(1 for r in rules.values()
                                         if r.get('memo_inserted_alternatives')),
                binding_attempts=sum(r['binding_attempts'] for r in rules.values()),
                generated_alternatives=sum(r['generated_alternatives'] for r in rules.values()),
                memo_inserted_alternatives=sum(r.get('memo_inserted_alternatives', 0)
                                               for r in rules.values()),
                duplicate_alternatives=sum(r['duplicate_alternatives'] for r in rules.values())))
        for size in manifest['sizes']:
            same = [r for r in rows if r['size'] == size]
            plans = [json.dumps(r['plan'], sort_keys=True, separators=(',', ':')) for r in same if r['complete']]
            equal = len(plans) == 3 and len(set(plans)) == 1
            for row in same:
                row['same_size_orders_plan_equal'] = equal
        records.append(dict(workload=workload, query=query,
                            postgres_rows=data['postgres_oracle']['rows_count'], rows=rows))
    for record in records:
        record['complete'] = all(row['complete'] for row in record['rows'])
    receipts.update(artifact_snapshot({'font': args.font}))
    report = dict(scope='nested_seeded_rule_count_and_local_order_diagnostic_not_global_promise',
        queries_planned=len(cases), queries_observed=len(records),
        queries_complete=sum(record['complete'] for record in records), missing=missing,
        sizes=manifest['sizes'], orders=['default', 'reverse', 'random'], timing_repeats=3,
        trace_scope='compact_rule_summaries_and_stage_costs_not_complete_candidate_chains',
        independent_query_families=1, query_independent_conclusion=False,
        rule_source=manifest['identities']['rules'],
        admitted_rules=next(iter(manifest['policies'].values()))['snapshot']['load']['admitted'],
        selection_scope=('completed_comparisons_discovered_under_alternate_run'
                         if args.run else 'queries_frozen_in_policy_experiment_manifest'),
        records=records, sources=receipts)
    (output / 'scale-summary.json').write_text(json.dumps(report, ensure_ascii=False,
        indent=2, allow_nan=False) + '\n')
    plot(report, output / '规则规模与优先级响应.png', args.font)
    plot_order_effect(report, output / '局部优先级相对规划耗时.png', args.font)
    plot_productive_response(report, output / '有效规则产出与搜索开销.png', args.font)
    print(json.dumps(dict(queries_observed=len(records),
        queries_complete=report['queries_complete'], missing=missing), ensure_ascii=False))


def plot(report, output, font):
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(len(report['records']), 3,
                            figsize=(13, 3.4 * max(1, len(report['records']))), squeeze=False)
    labels = {'default': '默认顺序', 'reverse': '反向顺序', 'random': '固定随机顺序'}
    colors = {'default': '#386cb0', 'reverse': '#e67e22', 'random': '#28a07a'}
    for panels, record in zip(axes, report['records']):
        for order in report['orders']:
            rows = sorted((r for r in record['rows'] if r['order'] == order), key=lambda r: r['size'])
            x = [r['size'] for r in rows]
            panels[0].plot(x, [r['planning_ms_median'] if r['complete'] else math.nan for r in rows],
                           'o-', label=labels[order], color=colors[order])
            panels[1].plot(x, [r['generated_alternatives'] if r['complete'] else math.nan for r in rows],
                           'o-', color=colors[order])
            panels[2].plot(x, [r['final_optimizer_cost'] if r['complete'] else math.nan for r in rows],
                           'o-', color=colors[order])
        for ax, title, ylabel in zip(panels,
                ('无跟踪规划时间（3 轮中位数）', 'DSL 生成替代数', '最终估算成本'),
                ('毫秒', '生成数', '估算成本（不是执行时间）')):
            ax.set_xticks(report['sizes'])
            ax.set_xlabel('启用规则数')
            ax.set_ylabel(ylabel)
            ax.set_title(f"{record['workload']}/{record['query']}｜{title}")
            ax.grid(alpha=.2)
        panels[0].legend(fontsize=8)
        if not record['complete']:
            for panel in panels:
                panel.text(.5, .5, '该查询的全部策略臂均超时', transform=panel.transAxes,
                           ha='center', va='center', color='#a33', fontsize=11)
    fig.suptitle('规则规模与局部优先级响应\n固定 seed 的嵌套成员集；同一规模只改变候选顺序', fontsize=14)
    footer = ('失败保持缺失；时间不含详细候选链 trace。'
              + (' 未完成：' + '、'.join(str(x['query']) for x in report['missing']) if report['missing'] else ''))
    fig.text(.5, .01, footer, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .91))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_order_effect(report, output, font):
    plt = chinese_plotting(font)
    records = [record for record in report['records'] if record['complete']]
    fig, axes = plt.subplots(1, len(records), figsize=(5 * max(1, len(records)), 4.8),
                             squeeze=False, sharey=True)
    for ax, record in zip(axes[0], records):
        by_size = {size: {row['order']: row for row in record['rows']
                          if row['size'] == size and row['complete']}
                   for size in report['sizes']}
        for order, label, color in (('reverse', '反向顺序', '#386cb0'),
                                    ('random', '固定随机顺序', '#e67e22')):
            values = []
            for size in report['sizes']:
                rows = by_size[size]
                baseline = rows['default']['planning_ms_median']
                values.append(100 * (rows[order]['planning_ms_median'] / baseline - 1))
            ax.plot(report['sizes'], values, 'o-', label=label, color=color)
        ax.axhline(0, color='#333333', linewidth=.8, label='默认顺序')
        ax.set_xticks(report['sizes'])
        ax.set_xlabel('启用规则数')
        ax.set_title(f"{record['workload']}/{record['query']}")
        ax.grid(alpha=.2)
    axes[0][0].set_ylabel('相对默认顺序的规划耗时变化（%）')
    axes[0][0].legend()
    fig.suptitle('不同规则规模下的局部优先级效果\n3 轮中位数；同规模计划与终止估算成本均一致')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def productive_response_points(report, order='default'):
    points = []
    for record in report['records']:
        rows = sorted((row for row in record['rows']
                       if row['complete'] and row['order'] == order), key=lambda row: row['size'])
        if not rows or not rows[0]['planning_ms_median'] or not rows[0]['binding_attempts']:
            continue
        base_ms, base_bindings = rows[0]['planning_ms_median'], rows[0]['binding_attempts']
        for row in rows:
            points.append(dict(workload=record.get('workload', report.get('workload', '')),
                query=record['query'], enabled_rules=row['size'],
                memo_inserted_alternatives=row['memo_inserted_alternatives'],
                planning_ratio=row['planning_ms_median'] / base_ms,
                binding_ratio=row['binding_attempts'] / base_bindings))
    return points


def plot_productive_response(report, output, font):
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    points = productive_response_points(report)
    keys = list(dict.fromkeys((point['workload'], point['query']) for point in points))
    colors = plt.cm.tab10.colors
    for index, key in enumerate(keys):
        rows = [point for point in points if (point['workload'], point['query']) == key]
        label = '/'.join(str(value) for value in key if value != '')
        for axis, field in zip(axes, ('planning_ratio', 'binding_ratio')):
            axis.plot([row['memo_inserted_alternatives'] for row in rows], [row[field] for row in rows],
                      'o-', label=label, color=colors[index % len(colors)])
            for row in rows:
                axis.annotate(str(row['enabled_rules']),
                              (row['memo_inserted_alternatives'], row[field]),
                              xytext=(3, 3), textcoords='offset points', fontsize=7)
    for axis, title, ylabel in zip(axes,
            ('规划时间响应', '绑定尝试响应'), ('相对最小规则集', '相对最小规则集')):
        axis.set_xscale('symlog', linthresh=1)
        axis.set_xlabel('实际插入 Memo 的根 alternative 数')
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=.2)
    if points:
        axes[0].legend(fontsize=8)
    fig.suptitle('Memo 有效扩张与搜索开销\n点标签为启用规则数；只比较默认顺序的查询内响应')
    fig.text(.5, .01, '0 产出规则仍有匹配成本；出现有效产出后可能经 Memo 与后继规则链放大',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .91))
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
