#!/usr/bin/env python3
"""Summarize observation-only OFF/CBO experiments using the shared contribution audit."""

import argparse
import json
from pathlib import Path
from statistics import median

from ml_orca.experiments.plot_stats_sweep import chinese_plotting
from ml_orca.trace.profile_rule_candidates import cbo_contribution


def paired_evidence(path):
    comparison = json.loads(path.read_text())
    scenarios = [s for s in comparison['rule_profile']['scenarios'] if s['stats_experiment'] is not None]
    if len(scenarios) != 1:
        raise ValueError('exactly one traced observation scenario is required per comparison')
    # Empty target lists are real no-intervention observations, not a fake 1x sweep.
    if any(o.get('stats_targets') for s in scenarios for arm in s['arms'].values()
           for o in arm.get('experiment_outcomes', [])):
        raise ValueError('use the sweep reporter for injected statistics')
    manifest = {'targets': [], 'points': [
        {'file': s['stats_experiment'], 'factors': [], 'requested_rows': []} for s in scenarios]}
    rows = cbo_contribution(manifest, comparison, path.parent, path.parent)
    for row in rows:
        for pair in row['pairs']:
            planning, execution = pair['delta_planning_ms'], pair['delta_execution_ms']
            pair['delta_plan_plus_execution_ms'] = planning + execution if planning is not None and execution is not None else None
    return {'artifact': str(path.resolve()), 'rule_hash': comparison['rule_profile']['rule_hash'],
            'runs': rows, 'scope': 'fixed_background_cbo_minus_off_not_population_effect',
            'timing': 'untraced_randomized_complete_blocks; descriptive_no_confidence_interval',
            'not_inferred': ['permanent_rbo_placement', 'benefit_of_every_rule_application',
                             'generalization_across_queries_or_data']}


def render(reports, output, font):
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    labels = list(reports)
    metrics = [('attempts', '全部规则尝试'), ('memo_expressions', 'Memo 表达式'), ('cost:costed', '完成物理计价')]
    for i, (key, label) in enumerate(metrics):
        for x, report in enumerate(reports.values()):
            valid = [r for r in report['runs'] if r['delta'] is not None and r['search']['complete']]
            if len(valid) != 1:
                continue  # Missing audit is not zero work; no pooling different scenarios.
            row = valid[0]
            value = row['search']['delta'][key] if key == 'cost:costed' else row['delta'][key]
            axes[0, 0].bar(x + (i-1)*.24, value, width=.24, color=('#4477aa', '#ee7733', '#228833')[i],
                           label=label if x == 0 else None)
            axes[0, 0].annotate(str(value), (x + (i-1)*.24, value), ha='center', fontsize=9)
    axes[0, 0].legend()
    axes[0, 0].set_title('搜索工作净增量：开启 − 关闭')
    for x, report in enumerate(reports.values()):
        valid = [r for r in report['runs'] if r['delta'] is not None]
        if len(valid) == 1:
            value = valid[0]['delta']['optimizer_cost']
            axes[0, 1].bar(x, value, color='#4477aa')
            axes[0, 1].annotate(f'{value:.3f}', (x, value), ha='center', fontsize=9)
        else:
            axes[0, 1].text(x, 0, '不可比', ha='center')
        for ax, field in zip(axes[1], ('delta_planning_ms', 'delta_execution_ms')):
            values = [p[field] for r in report['runs'] for p in r['pairs'] if p[field] is not None]
            if values:
                ax.scatter([x]*len(values), values, alpha=.55)
                ax.scatter(x, median(values), marker='_', s=350, color='#ee7733')
                ax.annotate(f'{len(values)}对；中位 {median(values):.3f}', (x, median(values)),
                            xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
            else:
                ax.text(x, 0, '无有效配对', ha='center')
    axes[0, 1].set_title('最终估算成本净差（负值较优，不是执行时间）')
    axes[1, 0].set_title('规划时间净差（毫秒，负值较快）')
    axes[1, 1].set_title('执行时间净差（毫秒，负值较快）')
    for ax in axes.flat:
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlim(-.6, len(labels)-.4)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.margins(y=.2)
    fig.suptitle('非空机制样本：规则搜索负担与最终收益同时观察\n固定背景、目标只在代价搜索中开关；不是查询无关画像的泛化结论')
    fig.savefig(output / '规则净影响.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    for x, report in enumerate(reports.values()):
        values = [p['delta_plan_plus_execution_ms'] for r in report['runs'] for p in r['pairs']
                  if p['delta_plan_plus_execution_ms'] is not None]
        if values:
            ax.scatter([x]*len(values), values, color='#4477aa', alpha=.6)
            ax.scatter(x, median(values), marker='_', s=400, color='#ee7733')
            ax.annotate(f'{len(values)}对；中位 {median(values):.3f}', (x, median(values)),
                        xytext=(0, 12), textcoords='offset points', ha='center')
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-.6, len(labels)-.4)
    ax.set_ylabel('开启 − 关闭（毫秒，对称对数轴；负值较快）')
    ax.set_title('每次重新规划并执行一次的净时间差\n同一配对内相加，再取中位数；不含网络时间，不代表计划复用收益')
    ax.margins(y=.2)
    fig.savefig(output / '单次规划加执行净差.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True, help='Chinese label=comparison.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, default=Path('output/fonts/NotoSansCJKsc-Regular.otf'))
    args = parser.parse_args()
    cases = [value.split('=', 1) for value in args.case]
    if any(len(value) != 2 or not all(value) for value in cases) or len({v[0] for v in cases}) != len(cases):
        parser.error('require unique nonempty labels and paths')
    reports = {label: paired_evidence(Path(path)) for label, path in cases}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / '净影响.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
    render(reports, args.output, args.font)


if __name__ == '__main__':
    main()
