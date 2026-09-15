"""Report a frozen policy subset against a full-policy baseline."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median

from ml_orca.collect.run_workload_comparison import plan_tree, timing_plan
from ml_orca.experiments.plot_stats_sweep import chinese_plotting


def arm_metrics(run):
    rules = run.get('dsl_observability', {}).get('rules', {})
    progress = run.get('optimizer_progress', [])
    return {
        'memo_outcome_complete': all('memo_inserted_alternatives' in r for r in rules.values()),
        'routed_rules': len(rules),
        'target_ready_rules': sum(bool(r['generated_alternatives'] or r['duplicate_alternatives'])
                                  for r in rules.values()),
        'memo_expanding_rules': sorted(h for h, r in rules.items()
                                       if r.get('memo_inserted_alternatives')),
        'binding_attempts': sum(r['binding_attempts'] for r in rules.values()),
        'generated_alternatives': sum(r['generated_alternatives'] for r in rules.values()),
        'memo_inserted_alternatives': sum(r.get('memo_inserted_alternatives', 0)
                                          for r in rules.values()),
        'duplicate_alternatives': sum(r['duplicate_alternatives'] for r in rules.values()),
        'final_cost': progress[-1]['cost'] if progress else None,
    }


def read_plan(directory, arm):
    path = directory / f'profile-0.{arm}.plan.json'
    try:
        return timing_plan(plan_tree(path.read_text()))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def valid_arm(run, oracle_equal, plan, metrics):
    return (run.get('plan_rc') == 0 and run.get('rows_rc') == 0 and not run.get('fallback')
            and run.get('optimizer') == 'pg_orca' and not run.get('server_failure')
            and not run.get('error') and oracle_equal is True and plan is not None
            and metrics['final_cost'] is not None and metrics['memo_outcome_complete'])


def analyze(results, candidate, baseline):
    rows, missing_rules = [], defaultdict(list)
    for path in sorted(results.glob('*/*/comparison.json')):
        data = json.loads(path.read_bytes())
        comparison = data.get('policy_comparison', {})
        if candidate not in comparison.get('arms', []) or baseline not in comparison.get('arms', []):
            raise ValueError(f'{path}: required policy arms missing')
        if len(comparison.get('scenarios', [])) != 1:
            raise ValueError(f'{path}: expected one policy scenario')
        arms = comparison['scenarios'][0]['arms']
        oracle = data.get('postgres_oracle', {}).get('mode_result_equal', {})
        item = {'workload': path.parent.parent.name, 'query': path.parent.name,
                'coverage_status': data.get('coverage_status'), 'arms': {}}
        plans = {}
        for name in (candidate, baseline):
            run = arms[name]
            metrics = arm_metrics(run)
            plan = read_plan(path.parent, name)
            equal = oracle.get(f'policy:0:{name}')
            item['arms'][name] = {**metrics, 'postgres_result_equal': equal,
                                  'valid': valid_arm(run, equal, plan, metrics),
                                  'plan_rc': run.get('plan_rc'), 'fallback': run.get('fallback'),
                                  'error': run.get('error', '')[-1000:]}
            plans[name] = plan
        left, right = item['arms'][candidate], item['arms'][baseline]
        if left['valid'] and right['valid']:
            item.update(status='both_valid', normalized_plan_equal=plans[candidate] == plans[baseline],
                        final_cost_equal=left['final_cost'] == right['final_cost'],
                        generated_equal=left['generated_alternatives'] == right['generated_alternatives'],
                        duplicate_equal=left['duplicate_alternatives'] == right['duplicate_alternatives'],
                        productive_set_equal=left['memo_expanding_rules'] == right['memo_expanding_rules'])
            item['fully_preserved'] = all(item[k] for k in ('normalized_plan_equal', 'final_cost_equal',
                'generated_equal', 'duplicate_equal', 'productive_set_equal'))
            item['binding_ratio_baseline_over_candidate'] = (
                right['binding_attempts'] / left['binding_attempts'] if left['binding_attempts'] else None)
            item['candidate_cost_delta_percent'] = (
                100 * (left['final_cost'] / right['final_cost'] - 1) if right['final_cost'] else None)
            for rule_hash in set(right['memo_expanding_rules']) - set(left['memo_expanding_rules']):
                missing_rules[rule_hash].append(f"{item['workload']}/{item['query']}")
        else:
            item['status'] = ('candidate_only_valid' if left['valid'] else
                              'baseline_only_valid' if right['valid'] else 'neither_valid')
            item['fully_preserved'] = False
        rows.append(item)
    both = [row for row in rows if row['status'] == 'both_valid']
    preserved = [row for row in both if row['fully_preserved']]
    ratios = [row['binding_ratio_baseline_over_candidate'] for row in preserved
              if row['binding_ratio_baseline_over_candidate'] is not None]
    return {
        'scope': 'frozen_policy_subset_holdout; diagnostic_trace_search_work_not_untraced_timing',
        'candidate': candidate, 'baseline': baseline, 'queries': len(rows),
        'status_counts': dict(Counter(row['status'] for row in rows)),
        'coverage_status_counts': dict(Counter(row['coverage_status'] for row in rows)),
        'both_valid': len(both), 'fully_preserved': len(preserved),
        'fully_preserved_fraction_of_both_valid': len(preserved) / len(both) if both else None,
        'fully_preserved_binding_ratio_median': median(ratios) if ratios else None,
        'new_baseline_productive_rules': dict(sorted(missing_rules.items())),
        'limitations': ['subset selected from prior outcomes', 'one frozen holdout batch',
                        'empty schemas', 'diagnostic trace is not a timing measurement'],
        'rows': rows,
    }


def render(report, output, font):
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    both = [row for row in report['rows'] if row['status'] == 'both_valid']
    for preserved, label, color in ((True, '计划、成本与搜索产出保持', '#228833'),
                                    (False, '计划或搜索产出变化', '#ee7733')):
        points = [row for row in both if row['fully_preserved'] is preserved]
        axes[0].scatter([row['arms'][report['baseline']]['binding_attempts'] for row in points],
                        [row['arms'][report['candidate']]['binding_attempts'] for row in points],
                        label=label, color=color, alpha=.8)
    limits = [value for row in both for value in
              (row['arms'][report['baseline']]['binding_attempts'],
               row['arms'][report['candidate']]['binding_attempts']) if value > 0]
    if limits:
        lo, hi = min(limits), max(limits)
        axes[0].plot([lo, hi], [lo, hi], '--', color='#777777', label='工作量相同')
        axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('全量策略绑定尝试数')
    axes[0].set_ylabel('规则子集绑定尝试数')
    axes[0].set_title('留出查询的搜索工作')
    axes[0].legend(fontsize=8)

    changed = [row for row in both if not row['final_cost_equal']]
    axes[1].bar(range(len(changed)), [row['candidate_cost_delta_percent'] for row in changed],
                color='#cc6677')
    axes[1].set_xticks(range(len(changed)),
                       [f"{row['workload']}/{row['query']}" for row in changed], rotation=25, ha='right')
    axes[1].axhline(0, color='#555555', linewidth=.8)
    axes[1].set_ylabel('子集相对全量的终止估算代价变化（%）')
    axes[1].set_title('发现集外的计划质量损失')

    labels = ['两臂有效', '仅子集有效', '仅全量有效', '两臂无效']
    keys = ['both_valid', 'candidate_only_valid', 'baseline_only_valid', 'neither_valid']
    axes[2].bar(labels, [report['status_counts'].get(key, 0) for key in keys],
                color=['#4477aa', '#66ccee', '#aa3377', '#999999'])
    axes[2].tick_params(axis='x', rotation=20)
    axes[2].set_title('保留全部失败分母')
    axes[2].set_ylabel('查询数')
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('规则子集留出检验：低产出匹配开销与遗漏收益必须同时建模')
    fig.text(.5, .01, '空表；独立 PostgreSQL 结果校验；诊断 trace 不作规划时间比较', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .92))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.results, args.candidate, args.baseline)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False,
                                                         indent=2, allow_nan=False) + '\n')
    render(report, args.output / '规则子集留出检验.png', args.font)
    print(json.dumps({k: report[k] for k in ('queries', 'status_counts', 'both_valid',
                                              'fully_preserved', 'new_baseline_productive_rules')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
