#!/usr/bin/env python3
"""Measure rule cost/benefit over a frozen synthetic-data scale design."""

import argparse
import json
import math
import random
import re
import subprocess
import sys
import zlib
from pathlib import Path
from statistics import median, quantiles
from string import Template

from ml_orca.common.paths import PGORCA_ROOT, TEST_ASSETS
from ml_orca.experiments.plot_stats_sweep import chinese_plotting
from ml_orca.experiments.profile_rule_pair import paired_evidence



def scale_setups(template, scales):
    if not scales or any(type(n) is not int or n < 1 for n in scales) or len(set(scales)) != len(scales):
        raise ValueError('data scales must be unique positive integers')
    if '${rows}' not in template:
        raise ValueError('setup template must contain ${rows}')
    return {n: Template(template).substitute(rows=n) for n in scales}


def render(report, output, font):
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    metrics = [('delta_planning_ms', 1, '规划开销净增量（毫秒）'),
               ('delta_execution_ms', -1, '执行时间节省（毫秒）'),
               ('delta_plan_plus_execution_ms', -1, '单次规划＋执行净收益（毫秒）')]
    points = sorted(report['points'], key=lambda p: p['rows'])
    for ax, (field, sign, title) in zip(axes.flat, metrics):
        xs, mids, low, high = [], [], [], []
        for point in points:
            values = [sign*p[field] for r in point.get('evidence', {}).get('runs', []) for p in r['pairs']
                      if p[field] is not None]
            xs.append(point['rows'])
            if values:
                q = quantiles(values, n=4, method='inclusive') if len(values) > 1 else values*3
                mids.append(median(values)); low.append(q[0]); high.append(q[2])
                ax.scatter([point['rows']]*len(values), values, alpha=.3, color='#4477aa')
                ax.annotate(f'{len(values)}对', (point['rows'], median(values)),
                            xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
            else:
                mids.append(math.nan); low.append(math.nan); high.append(math.nan)
        ax.plot(xs, mids, 'o-', color='#4477aa', label='配对差值中位数')
        ax.fill_between(xs, low, high, alpha=.2, color='#4477aa', label='配对差值四分位范围')
        ax.set_title(title)
        ax.axhline(0, color='gray', linewidth=.7)
    for field, label in [('attempts', '规则尝试净增'), ('memo_expressions', 'Memo 表达式净增'),
                         ('cost:costed', '完整物理计价净增')]:
        values = []
        for point in points:
            rows = point.get('evidence', {}).get('runs', [])
            r = rows[0] if len(rows) == 1 else {}
            valid = r.get('delta') is not None and r['search']['complete']
            values.append((r['search']['delta'].get(field, 0) if field == 'cost:costed' else r['delta'][field])
                          if valid else math.nan)
        axes[1, 1].plot([p['rows'] for p in points], values, 'o-', label=label)
    axes[1, 1].set_title('搜索工作净变化：开启 − 关闭')
    for ax in axes.flat:
        ax.set_xscale('log', base=2)
        # Failed points still need a positive domain when all measurements are missing.
        ax.set_xlim(min(p['rows'] for p in points) / 1.2, max(p['rows'] for p in points) * 1.2)
        ax.set_xticks([p['rows'] for p in points], [str(p['rows']) for p in points], rotation=25)
        ax.set_xlabel('数据模板行数参数（不是求值前已缓存基数）')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('规则在固定结构下的数据规模响应\n几何网格、规模点随机执行顺序；阴影不是置信区间；连线不是已拟合函数')
    fig.savefig(output / '数据规模响应.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pg-config', required=True)
    parser.add_argument('--audit-bin', required=True)
    parser.add_argument('--workload-dir', type=Path, required=True)
    parser.add_argument('--workload', required=True)
    parser.add_argument('--query', required=True, help='SQL file stem')
    parser.add_argument('--rule', required=True)
    parser.add_argument('--setup-template', type=Path, required=True)
    parser.add_argument('--rows', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=60790)
    parser.add_argument('--font', type=Path, default=Path('output/fonts/NotoSansCJKsc-Regular.otf'))
    args = parser.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*', n) for n in (args.workload, args.query)):
        parser.error('workload and query must be safe path components')
    if not re.fullmatch(r'[0-9a-f]{16}', args.rule) or not 1024 <= args.port <= 65535:
        parser.error('invalid rule identity or port')
    setups = scale_setups(args.setup_template.read_text(), args.rows)
    crc = lambda path: f'{zlib.crc32(path.read_bytes()):08x}'
    sources = [args.setup_template, args.workload_dir / args.workload / 'schema.sql',
               args.workload_dir / args.workload / 'sql' / (args.query + '.sql'), TEST_ASSETS / 'rules/orca_replacements.rules']
    manifest = {'design': 'declared_synthetic_data_scales_randomized_order_not_population_sample',
                'rule_hash': args.rule, 'seed': args.seed, 'sources': {str(p.resolve()): crc(p) for p in sources},
                'scales': args.rows, 'execution_order': random.Random(args.seed).sample(args.rows, len(args.rows)),
                'timing_repeats': 7, 'warmups': 1, 'statement_timeout_seconds': 60,
                'limitations': ['one_query_mechanism', 'data_and_statistics_change_together',
                                'no_heldout_validation', 'no_fitted_threshold']}
    args.output.mkdir(parents=True, exist_ok=False)
    for n, setup in setups.items():
        (args.output / f'setup-{n}.sql').write_text(setup)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = {'manifest': manifest, 'points': []}
    for n in manifest['execution_order']:
        output = args.output / f'rows-{n}'
        command = [sys.executable, '-m', 'ml_orca', 'compare-workload', '--pg-config', args.pg_config,
                   '--audit-bin', args.audit_bin, '--workload-dir', str(args.workload_dir), '--workload', args.workload,
                   '-t', f'{args.workload}/{args.query}', '--setup-sql', str(args.output / f'setup-{n}.sql'),
                   '--profile-rule', args.rule, '--profile-cbo-only', '--unbounded',
                   '--stats-experiment', str(TEST_ASSETS / 'rules/stats_input_observation.yaml'),
                   '--timing-repeats', '7', '--timing-warmups', '1', '--timing-seed', str(args.seed),
                   '--jobs', '1', '--timeout', '60', '--port', str(args.port), '--output', str(output)]
        with (args.output / f'rows-{n}.runner.log').open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        point = {'rows': n, 'runner_returncode': process.returncode}
        try:
            if any(crc(Path(p)) != value for p, value in manifest['sources'].items()):
                raise ValueError('experiment source changed')
            point['evidence'] = paired_evidence(output / args.workload / args.query / 'comparison.json')
        except (OSError, ValueError, KeyError) as error:
            point['analysis_error'] = str(error)
        report['points'].append(point)
        (args.output / '响应.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'rows={n}: rc={process.returncode}, analysis_error={point.get("analysis_error")}', flush=True)
    render(report, args.output, args.font)


if __name__ == '__main__':
    main()
