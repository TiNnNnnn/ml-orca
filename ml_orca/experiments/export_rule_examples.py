#!/usr/bin/env python3
"""Export bounded WeTune rule-pair examples for the ordinary pgORCA workload runner.

This is joint L/R instantiation, NOT reverse rewriting of arbitrary workload trees.
Unsupported domains and failed exports stay in the denominator. No rule is changed.
"""

import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import subprocess
import zlib

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify

from ml_orca.common.paths import TEST_ASSETS

SCRIPT = Path(__file__).resolve().parent


def render_example(schema_sql, source_sql, target_sql, prefix):
    """Translate and validate the adapter's bounded integer-column domain."""
    ddl = sqlglot.parse(schema_sql, read='mysql')
    catalog, inserts = {}, []
    for create in ddl:
        if not isinstance(create, exp.Create) or not isinstance(create.this, exp.Schema):
            raise ValueError('require CREATE TABLE with explicit columns')
        table, columns = create.this.this, create.this.expressions
        if not isinstance(table, exp.Table) or table.name in catalog or not columns:
            raise ValueError('invalid or duplicate example table')
        catalog[table.name] = {}
        values = [[] for _ in range(8)]
        for offset, column in enumerate(columns):
            if (not isinstance(column, exp.ColumnDef) or column.name in catalog[table.name]
                    or column.args['kind'].sql() not in ('INT', 'INTEGER')):
                raise ValueError('require distinct integer columns')
            kinds = [c.kind for c in column.constraints]
            if any(not isinstance(k, (exp.UniqueColumnConstraint, exp.NotNullColumnConstraint)) for k in kinds):
                raise ValueError('unsupported column constraint')
            unique = any(isinstance(k, exp.UniqueColumnConstraint) for k in kinds)
            not_null = any(isinstance(k, exp.NotNullColumnConstraint) for k in kinds)
            # DSL Unique is a relational key. SQL UNIQUE alone permits repeated
            # NULLs; use a strict non-null key witness, without changing the rule.
            if unique and not not_null:
                column.append('constraints', exp.ColumnConstraint(kind=exp.NotNullColumnConstraint()))
                not_null = True
            catalog[table.name][column.name] = 'INT'
            for row, values_row in enumerate(values):
                value = row + 1 if unique else (row + offset) % 3
                values_row.append('NULL' if not not_null and (row + offset) % 4 == 0 else str(value))
        name = exp.to_identifier(prefix + table.name, quoted=True).sql(dialect='postgres')
        names = ', '.join(c.this.sql(dialect='postgres') for c in columns)
        inserts.append(f'INSERT INTO {name} ({names}) VALUES ' +
                       ', '.join('(' + ', '.join(row) + ')' for row in values) + ';')
    queries = []
    for sql in (source_sql, target_sql):
        statements = sqlglot.parse(sql, read='mysql')
        if len(statements) != 1 or not isinstance(statements[0], exp.Query):
            raise ValueError('require one query per side')
        query = qualify(statements[0], dialect='postgres', schema=catalog)
        for operation in query.find_all(exp.Union, exp.Intersect, exp.Except):
            if len(operation.left.selects) != len(operation.right.selects):
                raise ValueError('set-op input arity differs')
        queries.append(query)
    if len(queries[0].selects) != len(queries[1].selects):
        raise ValueError('source/target output arity differs: example Input exposes extra columns')
    for tree in [*ddl, *queries]:
        for table in tree.find_all(exp.Table):
            if table.name not in catalog:
                raise ValueError('unresolved example table')
            table.set('this', exp.to_identifier(prefix + table.name, quoted=True))
    source, target = (q.sql(dialect='postgres', pretty=True) for q in queries)
    # Bag comparison is a concrete fixture check, not another equivalence proof.
    # Parenthesize each directional EXCEPT independently of UNION associativity.
    comparison = ('SELECT COUNT(*) AS differences FROM (((' + source + ') EXCEPT ALL ('
                  + target + ')) UNION ALL ((' + target + ') EXCEPT ALL ('
                  + source + '))) AS difference_rows')
    return {'schema': '\n'.join(d.sql(dialect='postgres') + ';' for d in ddl) + '\n',
            'setup': '\n'.join(inserts) + '\n',
            'source': source + ';\n', 'target': target + ';\n',
            'equivalence': comparison + ';\n'}


def decode_records(text, originals):
    records, seen = [], set()
    for line in text.splitlines():
        parts = line.split('\t')
        number = int(parts[0])
        if number not in originals or number in seen or len(parts) < 3:
            raise ValueError('unexpected or duplicate adapter record')
        seen.add(number)
        status = parts[1]
        if len(parts) != (5 if status == 'ok' else 3):
            raise ValueError('invalid adapter record width')
        payload = [base64.b64decode(s, validate=True).decode('utf-8') for s in parts[2:]]
        records.append({'line': number, 'stage': status, 'payload': payload})
    if seen != set(originals):
        raise ValueError('adapter omitted rule records')
    return records


def summarize_examples(output, results, font, report_output=None):
    from ml_orca.experiments.plot_stats_sweep import chinese_plotting
    from ml_orca.trace.profile_rule_candidates import candidate_evidence

    design = json.loads((output / 'manifest.json').read_text())
    summary = json.loads((results / 'summary.json').read_text())
    fixture = summary['fixture']
    if (fixture['schema_crc32'] != design['schema_crc32']
            or fixture['setup']['crc32'] != design['setup_crc32']):
        raise ValueError('result fixture does not match exported examples')
    report = {'generation_counts': design['counts'], 'examples': [],
              'limitations': design['limitations'], 'results': str(results.resolve())}
    for entry in design['examples']:
        if entry['status'] != 'ok':
            continue
        item = {'line': entry['line'], 'sides': {}}
        results_by_side = {}
        for side in ('source', 'target'):
            case = json.loads((results / 'rule_examples' / f'r{entry["line"]}_{side}' / 'comparison.json').read_text())
            if case['query_crc32'] != entry['queries'][side]['crc32']:
                raise ValueError('result query does not match exported example')
            if len(case['stats_experiments']) != 1:
                raise ValueError('require exactly one observation-only experiment')
            run = case['stats_experiments'][0]['modes']['replacement']
            audit = candidate_evidence(run)
            exclusions = list(audit['exclusions'])
            provenance = case.get('artifact_provenance', {})
            if provenance.get('endpoints_equal') is False:
                exclusions.append('experiment_artifacts_changed')
            rules_crc = provenance.get('before_server_start', {}).get('rules', {}).get('crc32')
            if rules_crc is not None and rules_crc != design.get('rules_crc32'):
                exclusions.append('rule_file_differs_from_export')
            if any(e.get('status') == 'applied' for e in run['stats_events']):
                raise ValueError('example reachability must not mix in cardinality interventions')
            rows = [r for r in audit['rows'] if r['rule_id'] == entry['line']]
            hashes = {r['rule_hash'] for r in rows}
            if len(hashes) > 1:
                raise ValueError('ambiguous rule identity')
            item['sides'][side] = {'complete': audit['complete'] and not exclusions, 'exclusions': exclusions,
                                   'artifact_endpoints_equal': provenance.get('endpoints_equal'),
                                   'rule_hash': next(iter(hashes), None),
                                   'outcome_equal': case['outcome_equal'],
                                   'rows_bag_equal': case['rows_bag_equal'],
                                   'postgres_oracle': case.get('postgres_oracle'),
                                   'forbidden_native_origins': case['forbidden_native_origins'],
                                   'join_enumeration_replaced': case['join_enumeration_replaced'],
                                   'all_attempts': len(audit['rows']),
                                   'statuses': dict(Counter(r['status'] for r in rows)),
                                   'direct_insertions': (None if any(r['status'] == 'ready_cbo'
                                       and r.get('direct_insertions') is None for r in rows)
                                       else sum(r.get('direct_insertions') or 0 for r in rows))}
            results_by_side[side] = case
        item['fixture_bag_equal'] = {}
        for mode in ('native', 'replacement'):
            a, b = (results_by_side[s]['modes'][mode] for s in ('source', 'target'))
            item['fixture_bag_equal'][mode] = (
                a['rows_bag_hash'] == b['rows_bag_hash'] if
                a['rows_rc'] == b['rows_rc'] == 0 and a['rows_bag_hash'] and b['rows_bag_hash'] else None)
        report['examples'].append(item)
    report['valid'] = bool(report['examples']) and all(
        all(v is True for v in e['fixture_bag_equal'].values()) and
        all(s['complete'] and s['outcome_equal'] and s['rows_bag_equal']
            and not s['forbidden_native_origins'] and s['join_enumeration_replaced']
            and (not design.get('requires_postgres_oracle', False) or
                 (s['postgres_oracle'] and s['postgres_oracle']['valid']))
            for s in e['sides'].values()) for e in report['examples'])
    destination = report_output if report_output is not None else output
    destination.mkdir(parents=True, exist_ok=True)
    (destination / '生成场景验证.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    entries = report['examples']
    labels = [f'规则原文件第 {e["line"]} 行' for e in entries]
    for axis, metric, title in zip(axes, ('ready_cbo', 'direct_insertions'),
                                  ('原规则构造就绪次数', '原规则直接新增 Memo 表达式数')):
        for side, offset, label in [('source', -.18, '源侧生成查询'), ('target', .18, '目标侧生成查询')]:
            values = []
            for entry in entries:
                s = entry['sides'][side]
                values.append((s['statuses'].get(metric, 0) if metric == 'ready_cbo' else s[metric])
                              if s['complete'] and (metric == 'ready_cbo' or s[metric] is not None)
                              else float('nan'))
            axis.bar([i + offset for i in range(len(entries))], values, width=.36, label=label)
        axis.set_xticks(range(len(entries)), labels, rotation=20)
        axis.set_title(title)
        axis.set_ylabel('次数')
        axis.legend()
    warnings = [f'第 {e["line"]} 行' for e in entries
                if any(s['forbidden_native_origins'] for s in e['sides'].values())]
    note = ('\n' + '、'.join(warnings) + '存在原生枚举规则介入，仅作诊断') if warnings else ''
    invalid_pairs = [f'第 {e["line"]} 行' for e in entries
                     if any(v is not True for v in e['fixture_bag_equal'].values())]
    if invalid_pairs:
        note += '\n' + '、'.join(invalid_pairs) + '两侧结果不一致或未完成，不是有效正例'
    if not report['valid']:
        note += '\n验收未通过：仅作诊断，不纳入正例画像'
    width = design.get('minimum_input_columns', 1)
    fig.suptitle(f'规则驱动的生成场景：正常 CBO 路径验证\n整数输入至少 {width} 列；不是反向优化器，不代表真实负载频率' + note)
    fig.savefig(destination / '生成场景验证.png', dpi=150)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wetune-classpath', help='current compiled modules plus dependency jars')
    parser.add_argument('--rules', type=Path, default=TEST_ASSETS / 'rules/orca_replacements.rules')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results', type=Path, help='audit an existing export against runner results; do not regenerate')
    parser.add_argument('--report-output', type=Path, help='keep a new audit separate from an earlier report')
    parser.add_argument('--minimal-inputs', action='store_true',
                        help='instantiate Input without spare unbound columns; never trim query outputs')
    parser.add_argument('--input-columns', type=int, default=1,
                        help='minimum concrete Input width; positional AttrsSub chooses the first column')
    parser.add_argument('--font', type=Path, default=Path('output/fonts/NotoSansCJKsc-Regular.otf'))
    args = parser.parse_args()
    if args.input_columns < 1:
        parser.error('--input-columns must be positive')
    if args.results:
        report = summarize_examples(args.output, args.results, args.font, args.report_output)
        if not report['valid']:
            raise SystemExit('example validation failed; diagnostic report retained')
        return
    if args.report_output:
        parser.error('--report-output requires --results')
    if not args.wetune_classpath:
        parser.error('generation requires --wetune-classpath')
    raw = args.rules.read_bytes()
    originals = {n: s for n, s in enumerate(raw.decode().splitlines(), 1)
                 if s.strip() and not s.lstrip().startswith('#')}
    if not originals:
        parser.error('no rules')
    if args.output.exists():
        parser.error('output must be a new directory')
    # Isolated runners can share /tmp but reuse a JVM PID. This offline adapter
    # needs no JVM performance counters; avoid hsperfdata collisions on stdout.
    command = ['java', '-XX:-UsePerfData', '-ea', '--class-path', args.wetune_classpath,
               str(SCRIPT / 'ExportRuleExamples.java'), str(args.rules.resolve())]
    if args.minimal_inputs:
        command.append('--minimal-inputs')
    if args.input_columns != 1:
        command.append(f'--input-columns={args.input_columns}')
    run = subprocess.run(command, text=True, capture_output=True, timeout=60, check=True)
    records = decode_records(run.stdout, originals)
    crc = lambda b: f'{zlib.crc32(b):08x}'
    manifest = {'schema_version': 1, 'design': 'joint_rule_pair_integer_examples',
                'minimum_input_columns': args.input_columns,
                'minimal_inputs': args.minimal_inputs,
                'integrity_domain': 'unique_nonnull_key_witness',
                'requires_postgres_oracle': True,
                'rules_crc32': crc(raw), 'command': command, 'sqlglot_version': sqlglot.__version__,
                'adapter_crc32': crc((SCRIPT / 'ExportRuleExamples.java').read_bytes()),
                'exporter_crc32': crc(Path(__file__).read_bytes()), 'examples': [],
                'limitations': ['not_arbitrary_target_reverse_rewrite', 'not_population_sampling',
                                'not_proof_validation', 'no_statistics_injection']}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'adapter.stdout').write_text(run.stdout)
    (args.output / 'adapter.stderr').write_text(run.stderr)
    destination = args.output / 'rule_examples'
    (destination / 'sql').mkdir(parents=True)
    (args.output / 'checks').mkdir()
    schemas, setups = [], []
    for record in records:
        item = {'line': record['line'], 'rule_text_crc32': crc(originals[record['line']].encode()),
                'status': record['stage']}
        if record['stage'] == 'ok':
            try:
                name = f'r{record["line"]}'
                rendered = render_example(*record['payload'], prefix=name + '_')
                schemas.append(rendered.pop('schema'))
                setups.append(rendered.pop('setup'))
                item['queries'] = {}
                for side, sql in rendered.items():
                    # Verification wrappers introduce joins/set operations of their
                    # own. Never let them contaminate the profiled query population.
                    folder = args.output / 'checks' if side == 'equivalence' else destination / 'sql'
                    path = folder / f'{name}_{side}.sql'
                    path.write_text(sql)
                    item['queries'][side] = {'path': str(path), 'crc32': crc(sql.encode())}
            except (ValueError, sqlglot.errors.SqlglotError) as ex:
                item.update(status='postgres_export', error=str(ex))
        else:
            item['error'] = record['payload'][0]
        manifest['examples'].append(item)
    schema, setup = ''.join(schemas), ''.join(setups) + 'ANALYZE;\n'
    (destination / 'schema.sql').write_text(schema)
    (args.output / 'setup.sql').write_text(setup)
    manifest.update(schema_crc32=crc(schema.encode()), setup_crc32=crc(setup.encode()))
    manifest['counts'] = dict(Counter(e['status'] for e in manifest['examples']))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest['counts']))


if __name__ == '__main__':
    main()
