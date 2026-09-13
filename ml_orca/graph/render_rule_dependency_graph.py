#!/usr/bin/env python3
"""Render every rule and every position/evidence edge, without graph pruning."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import zlib

from ml_orca.trace.compare_rule_traces import read_records
from ml_orca.graph.merge_rule_graph import merge_graph, render_dot


def graph_counts(graph):
    known = {node['rule_hash'] for node in graph['nodes']}
    if len(known) != len(graph['nodes']) or any(
            e['src_rule'] not in known or e['dst_rule'] not in known for e in graph['edges']):
        raise ValueError('graph needs unique rule nodes and known edge endpoints')
    return {'rules': len(graph['nodes']), 'position_evidence_edges': len(graph['edges']),
            'directed_rule_pairs': len({(e['src_rule'], e['dst_rule']) for e in graph['edges']}),
            'edge_types': dict(Counter(e.get('evidence', 'unknown') for e in graph['edges'])),
            'unresolved_inputs': len(graph.get('unresolved_inputs', []))}


def query_edge_support(runs):
    """Descriptive support, not benefit, independence, or a population estimate.

    Keep partial-run positive evidence. Missing status is unknown, not failure;
    repeated bindings/positions count as events but give a pair only one query vote.
    Raw plan files without query identities are deliberately not included.
    """
    seen, pairs, datasets, complete = set(), {}, Counter(), 0
    for dataset, run in runs:
        key = dataset, run['case_id']
        if key in seen:
            raise ValueError('duplicate discovery query; use one consolidated summary')
        seen.add(key)
        datasets[dataset] += 1
        is_complete = run.get('complete') is True
        complete += is_complete
        local = {}
        for edge in run.get('edges', []):
            if edge.get('engine') != 'pgorca' or edge.get('kind') != 'rule_edge':
                continue
            pair = edge['src_rule'], edge['dst_rule']
            counts, outcomes = local.setdefault(pair, (Counter(), Counter()))
            status = edge.get('candidate_status')
            counts[status if isinstance(status, str) else None] += 1
            outcome = edge.get('dst_memo_outcome')
            if isinstance(outcome, str):
                outcomes[outcome] += 1
        for pair, (counts, outcomes) in local.items():
            row = pairs.setdefault(pair, {
                'src_rule': pair[0], 'dst_rule': pair[1], 'query_support': 0,
                'complete_query_support': 0, 'event_count': 0, 'max_events_per_query': 0,
                'dataset_query_support': Counter(), 'candidate_status_query_support': Counter(),
                'candidate_status_event_counts': Counter(), 'events_without_candidate_status': 0,
                'dst_memo_outcome_query_support': Counter(), 'dst_memo_outcome_event_counts': Counter()})
            events = sum(counts.values())
            row['query_support'] += 1
            row['complete_query_support'] += is_complete
            row['event_count'] += events
            row['max_events_per_query'] = max(row['max_events_per_query'], events)
            row['dataset_query_support'][dataset] += 1
            row['events_without_candidate_status'] += counts[None]
            for status, count in counts.items():
                if status is not None:
                    row['candidate_status_query_support'][status] += 1
                    row['candidate_status_event_counts'][status] += count
            row['dst_memo_outcome_query_support'].update(outcomes.keys())
            row['dst_memo_outcome_event_counts'].update(outcomes)
    return {'scope': 'identified_summary_queries_only; descriptive_positive_support_not_benefit',
            'query_units': len(seen), 'complete_query_units': complete,
            'incomplete_query_units': len(seen) - complete,
            'dataset_query_units': dict(datasets),
            'missing_status_is_unknown': True, 'raw_plan_files_included': False,
            'pairs': [pairs[key] for key in sorted(pairs)]}


def category(root):
    if root in {'Filter', 'InSubFilter', 'Exists', 'NotExists', 'Any', 'All'}:
        return '过滤与子查询'
    if root in {'Proj', 'Compute'}:
        return '投影与计算'
    if root in {'Proj*', 'Union', 'Union*', 'Intersect', 'Intersect*', 'Except', 'Except*'}:
        return '去重与集合'
    if 'Join' in root or 'Apply' in root:
        return '连接与关联'
    if root == 'Agg':
        return '聚合'
    if root in {'Window', 'WindowRows', 'SortAsc', 'Limit'}:
        return '排序与窗口'
    return '公共表及其他'


def render(graph, output, font, summary):
    import networkx as nx
    from matplotlib.lines import Line2D
    from ml_orca.experiments.plot_stats_sweep import chinese_plotting

    plt = chinese_plotting(font)
    colors = dict(zip(['过滤与子查询', '投影与计算', '去重与集合', '连接与关联', '聚合',
                       '排序与窗口', '公共表及其他'],
                      ['#5489c5', '#a680bf', '#52a790', '#dd9c4a', '#cc6971', '#65a8bc', '#85939b']))
    nodes = sorted(graph['nodes'], key=lambda n: (category(n['source_root']), n['rule_hash']))
    labels = {n['rule_hash']: f'规{i:03}' for i, n in enumerate(nodes, 1)}
    g = nx.MultiDiGraph()
    g.add_nodes_from(labels)  # Include rules without any observed or static edge.
    for edge in graph['edges']:
        g.add_edge(edge['src_rule'], edge['dst_rule'], evidence=edge)
    assert g.number_of_edges() == len(graph['edges'])
    # Keep dense hubs from hiding node labels. This changes only geometry;
    # every parallel edge remains in the MultiDiGraph and in both exports.
    positions = nx.circular_layout(g)
    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_facecolor('#fafbfd')
    ax.set_facecolor('#fafbfd')
    styles = ['arc3,rad=0']
    parallel = max(Counter((a, b) for a, b in g.edges()).values(), default=1)
    styles += [f'arc3,rad={(1 if i % 2 else -1) * (.04 + .4 * i / parallel)}'
               for i in range(1, parallel)]
    edge_styles = {
        'static': ('#9aa4b0', .08, '静态模板兼容（并非触发证明）'),
        'unknown': ('#9275ac', .16, '动态关系（未细分候选状态）'),
        'failed': ('#cf9350', .10, '尝试相关（无就绪候选观测）'),
        'ready': ('#268b87', .25, '含就绪候选观测（不等于最终入选）'),
    }
    for kind, (color, alpha, _) in edge_styles.items():
        edges = []
        for a, b, key, attrs in g.edges(keys=True, data=True):
            e = attrs['evidence']
            counts = e.get('candidate_status_counts', {})
            typ = ('static' if e.get('evidence') == 'static_template' else
                   'ready' if counts.get('ready_cbo', 0) else 'failed' if counts else 'unknown')
            if typ == kind:
                edges.append((a, b, key))
        nx.draw_networkx_edges(g, positions, edgelist=edges, ax=ax, arrows=True,
                               arrowstyle='-|>', arrowsize=9, width=.65,
                               edge_color=color, alpha=alpha, connectionstyle=styles, node_size=360)
    nx.draw_networkx_nodes(g, positions, nodelist=list(labels), node_size=360,
                          node_color=[colors[category(n['source_root'])] for n in nodes],
                          edgecolors='white', linewidths=1, ax=ax)
    nx.draw_networkx_labels(g, positions, labels, font_size=7, ax=ax,
                           font_family=plt.rcParams['font.family'][0], font_color='#15293a',
                           bbox={'facecolor': '#ffffffd9', 'edgecolor': 'none', 'pad': .4})
    counts = graph_counts(graph)
    ax.set_title('全量规则依赖观测图｜全部节点、全部位置与证据边\n'
                 f"{counts['rules']} 条独立规则 · {counts['position_evidence_edges']:,} 条边 · "
                 f"{counts['directed_rule_pairs']:,} 对有向规则关系\n"
                 '按源算子类别环形排布；排布位置不表示执行顺序', fontsize=19, pad=24)
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                      markersize=10, label=name) for name, color in colors.items()]
    handles += [Line2D([0], [0], color=color, lw=2, label=name)
                for color, _, name in edge_styles.values()]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(.5, .035))
    fig.text(.5, .015,
             f"历史发现查询 {summary['historical_queries']} 条（完整 {summary['historical_complete']}、"
             f"不完整 {summary['historical_queries'] - summary['historical_complete']}）"
             f"；补充最新计划跟踪 {summary['trace_files']} 份。保留失败前缀，不作收益或因果证明。",
             ha='center', fontsize=11)
    ax.set_axis_off()
    fig.tight_layout(rect=(0, .12, 1, .96))
    fig.savefig(output / '全量规则依赖图.png', dpi=180)
    fig.savefig(output / '全量规则依赖图.svg')
    plt.close(fig)
    with (output / '节点索引.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['图中编号', '规则身份', '审计编号', '算子类别', '源模板', '目标模板'])
        for node in nodes:
            writer.writerow([labels[node['rule_hash']], node['rule_hash'], node.get('rule_id'),
                             category(node['source_root']), node.get('source_pattern'), node.get('target_pattern')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--summary', type=Path, action='append', default=[])
    parser.add_argument('--trace', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, default=Path('output/fonts/NotoSansCJKsc-Regular.otf'))
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text())
    hashes = [n['rule_hash'] for n in graph['nodes']]
    if len(set(hashes)) != len(hashes):
        raise ValueError('base graph must have one node per canonical rule')
    snapshots, seen, complete, runs = [], set(), 0, []
    for path in [args.graph, *args.summary, *args.trace]:
        raw = path.read_bytes()
        snapshots.append({'path': str(path.resolve()), 'bytes': len(raw), 'crc32': f'{zlib.crc32(raw):08x}'})
    for path in args.summary:
        summary = json.loads(path.read_text())
        for dataset in summary['datasets']:
            for run in dataset['runs']:
                key = dataset['dataset'], run['case_id']
                if key in seen:
                    raise ValueError('duplicate discovery query; use one consolidated summary')
                seen.add(key)
                complete += run.get('complete') is True
                runs.append((dataset['dataset'], run))
                graph = merge_graph(graph, run.get('edges', []))
    for path in args.trace:
        graph = merge_graph(graph, read_records(path))
    summary = {**graph_counts(graph), 'historical_queries': len(seen), 'historical_complete': complete,
               'trace_files': len(args.trace), 'input_snapshots': snapshots,
               'scope': 'all_current_rules_and_available_observations; mixed_epochs_not_population_estimate',
               'pruned_nodes': 0, 'pruned_position_evidence_edges': 0}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'rule_graph.json').write_text(json.dumps(graph, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'rule_graph.dot').write_text(render_dot(graph))
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'query_edge_support.json').write_text(
        json.dumps(query_edge_support(runs), ensure_ascii=False, indent=2) + '\n')
    render(graph, args.output, args.font, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'input_snapshots'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
