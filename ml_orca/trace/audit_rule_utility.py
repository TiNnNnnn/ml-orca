"""Read-only D/F/C readiness audit of saved comparison runs; no inferred credit.

Reuse the existing trace audits. Cost reductions are delayed context-local
observations, not causal rule labels. Missing producer instances stay missing.
"""
import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path

from ml_orca.objectives.search_work import observed_work_vector
from ml_orca.trace.profile_rule_candidates import (
    candidate_evidence, binding_origin_evidence, cost_lifecycle_evidence, state_coverage)


def best_cost_ledger(costs, lifecycle):
    """Never compare across contexts/statistics or interpret a first plan as zero gain.

    A global stats watermark equality is deliberately conservative: unrelated
    stats writes can mask a valid comparison. Per-context stats dependencies
    are needed before relaxing this guard.
    """
    candidates = {e['sequence']: e for e in costs}
    fields = ('group', 'optimization_context', 'search_stage', 'stats_lifecycle_sequence',
              'required_columns', 'required_order_columns', 'required_order_matching',
              'required_distribution_type')
    rows = []
    for event in lifecycle:
        if event.get('status') != 'best_updated':
            continue
        new = candidates.get(event.get('candidate_sequence'), {})
        old = candidates.get(event.get('previous_candidate_sequence'), {})
        reasons = []
        for field in ('group', 'optimization_context'):
            if event.get(field) is None or event.get(field) != new.get(field):
                reasons.append('lifecycle_context_mismatch')
        first = event.get('previous_candidate_sequence') == 0
        for candidate in ([new] if first else [old, new]):
            value = candidate.get('cost')
            if (candidate.get('status') != 'costed' or candidate.get('cost_kind') != 'computed'
                    or type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                reasons.append('missing_computed_cost')
        if not first:
            for field in fields:
                if old.get(field) is None or new.get(field) is None:
                    reasons.append('missing_' + field)
                elif old[field] != new[field]:
                    reasons.append('changed_' + field)
            if old.get('sequence', 0) >= new.get('sequence', 0):
                reasons.append('nonpreceding_cost_candidate')
        rows.append({'event_sequence': event.get('sequence'),
                     'group': event.get('group'), 'optimization_context': event.get('optimization_context'),
                     'candidate_sequence': event.get('candidate_sequence'),
                     'previous_candidate_sequence': event.get('previous_candidate_sequence'),
                     'first_feasible': first and not reasons,
                     'cost_reduction': old['cost'] - new['cost'] if not first and not reasons else None,
                     'exclusions': sorted(set(reasons))})
    return rows


def root_search_progress(run, root_updates, work):
    """At-update work watermarks, not the creation ordinal of a reused candidate.

    Caller supplies audited root updates and work streams. Legacy traces stay
    unsupported: candidate identity is not an elapsed-work measurement.
    Counters remain separate units; this is not milliseconds or scalar C.
    """
    outcomes = run.get('experiment_outcomes', [])
    final = outcomes[0] if len(outcomes) == 1 else {}
    errors = list(work['exclusions'])
    if not work['complete']:
        errors.append('incomplete_search_work')
    if final.get('cost_progress_version') != 1:
        errors.append('cost_progress_version_missing')
    fields = {'preceding_rule_candidates': 'rule_candidates',
              'preceding_cost_candidates': 'cost_candidates',
              'preceding_search_checks': 'search_checks',
              'stats_lifecycle_sequence': 'stats_lifecycle_events'}
    previous = dict.fromkeys(fields, 0)
    events = {}
    costs = {e['sequence']: e for e in run.get('cost_events', [])}
    for event in run.get('cost_lifecycle_events', []):
        if event['status'] == 'selected_plan':
            continue
        for key, total_key in fields.items():
            value, total = event.get(key), final.get(total_key)
            if (type(value) is not int or type(total) is not int
                    or not previous[key] <= value <= total):
                errors.append('missing_or_nonmonotone_work_watermark:' + key)
            else:
                previous[key] = value
        for ref in ('candidate_sequence', 'previous_candidate_sequence'):
            candidate = event.get(ref)
            if (type(candidate) is not int or candidate < 0
                    or type(event.get('preceding_cost_candidates')) is not int
                    or candidate > event['preceding_cost_candidates']):
                errors.append('cost_reference_beyond_work_watermark')
        events[event['sequence']] = event
    points = []
    epoch = None
    for update in root_updates:
        event = events.get(update['event_sequence'], {})
        candidate = costs.get(update['candidate_sequence'], {})
        if update['exclusions']:
            errors.extend(update['exclusions'])
        if not event or candidate.get('cost_kind') != 'computed':
            errors.append('missing_root_progress_event_or_cost')
        current = event.get('stats_lifecycle_sequence')
        if epoch is not None and current != epoch:
            errors.append('root_progress_statistics_changed')
        epoch = current
        dispatched = event.get('preceding_rule_candidates')
        if (type(dispatched) is not int
                or dispatched < candidate.get('preceding_rule_candidates', 0)):
            errors.append('root_progress_precedes_candidate_work')
        points.append(dict(event_sequence=update['event_sequence'], candidate_sequence=update['candidate_sequence'],
                           optimizer_cost=candidate.get('cost'), first_feasible=update['first_feasible'],
                           work={key: event.get(source) for key, source in (
                               ('dsl_dispatched_attempts', 'preceding_rule_candidates'),
                               ('cost_entries', 'preceding_cost_candidates'),
                               ('search_checks', 'preceding_search_checks'))}))
    if not points or not points[0]['first_feasible']:
        errors.append('missing_first_feasible_root')
    if not points or points[-1]['optimizer_cost'] != final.get('optimizer_cost'):
        errors.append('root_progress_terminal_cost_mismatch')
    return {'complete': not errors, 'exclusions': sorted(set(errors)),
            'scope': 'root_context_at_update_work_vector_not_time_or_per_rule_credit',
            'points': points if not errors else None,
            'terminal_work': work['values'] if not errors else None,
            'before_first_feasible': 'no_plan_not_zero_cost',
            'after_terminal_work': 'unobserved_do_not_extrapolate'}


def cost_origin_evidence(run, attempts):
    """Validate at-event generators, not a retrospective join by mutable Memo ids."""
    outcomes = run.get('experiment_outcomes', [])
    final = outcomes[0] if len(outcomes) == 1 else {}
    problems = set()
    if final.get('cost_origin_trace_version') != 1:
        problems.add('cost_origin_version_missing')
    by_id = {r['sequence']: r for r in attempts}
    sources = {}
    for event in run.get('cost_events', []):
        refs = event.get('dsl_origin_instances')
        if not isinstance(refs, list):
            problems.add('cost_origin_instances_missing')
            continue
        chain = event.get('origin_chain')
        if not isinstance(chain, list) or any(not isinstance(n, dict) for n in chain):
            problems.add('cost_origin_chain_missing_or_invalid')
            continue
        chain = [event] + chain
        seen, generators, exposed, inserters = set(), set(), set(), set()
        for ref in refs:
            if not isinstance(ref, dict):
                problems.add('invalid_cost_origin_instance')
                continue
            seq, depth = ref.get('candidate_sequence'), ref.get('origin_depth')
            source = by_id.get(seq, {}) if type(seq) is int else {}
            watermark = event.get('preceding_rule_candidates')
            if (type(seq) is not int or type(watermark) is not int or not 0 < seq <= watermark
                    or source.get('rule_hash') != ref.get('rule_hash')
                    or source.get('evaluated') is not True or source.get('status') != 'ready_cbo'):
                problems.add('unresolved_cost_origin_instance')
                continue
            if (type(depth) is not int or not 0 <= depth < len(chain)
                    or any(type(ref.get(k)) is not int or ref[k] != chain[depth].get(k)
                           for k in ('group', 'group_expression'))):
                problems.add('cost_origin_memo_mismatch')
                continue
            path = ref.get('target_path')
            relation, outcome = ref.get('relation'), ref.get('outcome')
            if (not isinstance(path, str) or re.fullmatch(r'r(?:/[0-9]+)*', path) is None
                    or relation not in ('memo_consumes', 'input_exposes')
                    or outcome not in ('memo_inserted', 'memo_duplicate', 'memo_rehashed')):
                problems.add('invalid_cost_origin_position')
                continue
            identity = (seq, depth, path, relation, outcome)
            if identity in seen:
                problems.add('duplicate_cost_origin_position')
            seen.add(identity)
            if type(seq) is int:
                (generators if relation == 'memo_consumes' else exposed).add(seq)
                if relation == 'memo_consumes' and outcome == 'memo_inserted':
                    inserters.add(seq)
        sources[event['sequence']] = {'generators': sorted(generators), 'exposed': sorted(exposed),
                                      'recorded_inserters': sorted(inserters)}
    return {'complete': not problems, 'exclusions': sorted(problems),
            'scope': 'known_generators_at_costing_not_exclusive_or_causal_credit',
            'sources': sources if not problems else None}


def physical_plan_sources(costs, sources, roots):
    """Follow the costed child chosen then, not the group's later best candidate.

    Generator sets are nonexclusive alternatives. A first inserter is recorded
    provenance, not causal necessity; background/native work remains explicit.
    Only requested roots are traversed. No recursive Python depth limit and no
    all-pairs closure across the whole Memo.
    """
    by_id = {c['sequence']: c for c in costs}
    plans = {}
    for root in sorted(set(roots)):
        visited, pending = set(), [root]
        generators, exposed, inserters, background = set(), set(), set(), set()
        while pending:
            seq = pending.pop()
            if seq in visited:
                continue
            candidate = by_id.get(seq, {})
            if candidate.get('status') != 'costed' or candidate.get('cost_kind') != 'computed':
                raise ValueError('physical_plan_source_requires_computed_candidate')
            children = candidate.get('child_contexts')
            if not isinstance(children, list) or seq not in sources:
                raise ValueError('physical_plan_source_missing_children_or_origins')
            visited.add(seq)
            generators.update(sources[seq]['generators'])
            exposed.update(sources[seq]['exposed'])
            inserters.update(sources[seq]['recorded_inserters'])
            if not sources[seq]['generators']:
                background.add(seq)
            for child in children:
                child_seq = child.get('cost_candidate_sequence') if isinstance(child, dict) else None
                if (type(child_seq) is not int or not 0 < child_seq < seq
                        or child_seq not in by_id
                        or any(by_id[child_seq].get(k) != child.get(k)
                               for k in ('group', 'optimization_context'))):
                    raise ValueError('physical_plan_source_invalid_child_reference')
                pending.append(child_seq)
        plans[root] = {'cost_candidates': sorted(visited), 'generators': sorted(generators),
                       'root_only_generators': sources[root]['generators'],
                       'recorded_inserters': sorted(inserters), 'exposed_inputs': sorted(exposed),
                       'no_known_generator_candidates': sorted(background)}
    return plans


def instance_work_ledger(rows, edges, cost_sources):
    """Unique event-set accounting; input exposure is not a new generator.

    Call only after candidate/origin/cost audits. Counts are separate work
    dimensions, not equal-cost operations or complete per-rule C labels.
    """
    by_id = {r['sequence']: r for r in rows}
    parents = {seq: set() for seq in by_id}
    for edge in edges:
        parents[edge['dst_candidate_sequence']].add(edge['src_candidate_sequence'])
    ancestors = {}
    for seq in sorted(by_id):
        ancestors[seq] = set(parents[seq])
        for parent in parents[seq]:
            if parent >= seq:
                raise ValueError('instance dependency must precede its consumer')
            ancestors[seq].update(ancestors[parent])
    descendants = {seq: set() for seq in by_id}
    for seq, sources in ancestors.items():
        for source in sources:
            descendants[source].add(seq)
    own_costs = {seq: set() for seq in by_id}
    downstream_costs = {seq: set() for seq in by_id}
    exposed_costs = {seq: set() for seq in by_id}
    for cost_seq, sources in cost_sources.items():
        for source in sources['generators']:
            own_costs[source].add(cost_seq)
            for ancestor in ancestors[source]:
                downstream_costs[ancestor].add(cost_seq)
        for source in sources['exposed']:
            exposed_costs[source].add(cost_seq)
    result = []
    for seq, row in by_id.items():
        result.append({'sequence': seq, 'rule_hash': row['rule_hash'], 'status': row['status'],
                       'evaluated': row['evaluated'], 'parents': sorted(parents[seq]),
                       'ancestors': sorted(ancestors[seq]),
                       'descendant_attempts': sorted(descendants[seq]),
                       'self_diagnostic_us': {k: row.get(k) for k in ('match_us', 'constraint_us', 'instantiate_us')},
                       'generator_cost_events': sorted(own_costs[seq]),
                       'descendant_cost_events': sorted(downstream_costs[seq] - own_costs[seq]),
                       'exposed_input_cost_events': sorted(exposed_costs[seq])})
    linked = set().union(*own_costs.values()) if own_costs else set()
    return {'scope': 'observed_event_sets_nonadditive_across_rules_not_causal_C', 'instances': result,
            'work_accounting': {'attempts': len(rows), 'evaluated_attempts': sum(r['evaluated'] for r in rows),
                                'cost_events': len(cost_sources), 'generator_linked_cost_events': len(linked),
                                'no_known_generator_cost_events': len(cost_sources) - len(linked)},
            'not_included': ['binding_construction', 'exclusive_insert_derive_cost_work',
                             'physical_child_dependency_credit', 'causal_necessity']}


def audit_utility_run(run):
    attempts = candidate_evidence(run)
    origins = binding_origin_evidence(run)
    costs = cost_lifecycle_evidence(run)
    rows, edges = attempts['rows'], origins['edges']
    ledger = best_cost_ledger(run.get('cost_events', []), costs['events']) if costs['complete'] else []
    cost_by_id = {e['sequence']: e for e in costs['cost_candidates']}
    # Extraction events reference a cost candidate; they do not duplicate its context.
    root_contexts = {(cost_by_id[e['candidate_sequence']]['group'],
                      cost_by_id[e['candidate_sequence']]['optimization_context'])
                     for e in costs['selected'] if e.get('parent_plan_node') == 0
                     and e['candidate_sequence'] in cost_by_id}
    producer_refs = sum(type(e.get('src_candidate_sequence')) is int
                        and e['src_candidate_sequence'] > 0 for e in edges)
    cost_origins = cost_origin_evidence(run, rows)
    instance_ready = (attempts['complete'] and origins['complete'] and costs['complete']
                      and origins['producer_instance_coverage'] == 'validated_source_attempts'
                      and cost_origins['complete'])
    roots = [e for e in costs['selected'] if e.get('parent_plan_node') == 0]
    outcomes = run.get('experiment_outcomes', [])
    reported_cost = outcomes[0].get('optimizer_cost') if len(outcomes) == 1 else None
    terminal_valid = (costs['complete'] and len(roots) == 1 and type(reported_cost) in (int, float)
                      and math.isfinite(reported_cost) and reported_cost >= 0
                      and math.isclose(reported_cost, roots[0]['cost'], rel_tol=1e-8, abs_tol=1e-8))
    plan_sources = None
    plan_source_problems = []
    if instance_ready:
        requested = {e['candidate_sequence'] for e in ledger}
        requested.update(e['previous_candidate_sequence'] for e in ledger if e['previous_candidate_sequence'])
        requested.update(e['candidate_sequence'] for e in roots)
        try:
            plan_sources = physical_plan_sources(run['cost_events'], cost_origins['sources'], requested)
        except ValueError as error:
            plan_source_problems.append(str(error))
    else:
        plan_source_problems.append('requires_complete_instance_and_cost_provenance')
    root_updates = [r for r in ledger if (r['group'], r['optimization_context']) in root_contexts]
    work = observed_work_vector(run)
    return {
        'scope': 'observability_audit_not_training_labels_or_policy_benefit',
        'audits': {name: {'complete': a['complete'], 'exclusions': a['exclusions']}
                   for name, a in (('attempts', attempts), ('origins', origins), ('cost_lifecycle', costs))},
        'attempts': len(rows), 'rules': len({r['rule_hash'] for r in rows}),
        'attempt_statuses': dict(Counter(r['status'] for r in rows)),
        'context_coverage': state_coverage(rows),
        'observed_search_work': {'evaluated_attempts': attempts['evaluations'] if attempts['complete'] else None,
                                 'cost_candidates': len(run.get('cost_events', [])) if costs['complete'] else None},
        'search_work_scope': 'separate_run_level_counters_not_equal_cost_operations_or_per_rule_C',
        'search_work_audit': work,
        'root_search_progress': root_search_progress(run, root_updates, work),
        'origin_edges': len(edges), 'edges_with_source_instance_field': producer_refs,
        'missing_source_instance_fields': len(edges) - producer_refs,
        'best_updates': len(ledger),
        'comparable_updates': sum(r['cost_reduction'] is not None for r in ledger),
        'masked_update_reasons': dict(Counter(reason for r in ledger for reason in r['exclusions'])),
        'root_cost_updates': root_updates,
        'cost_origin_audit': cost_origins,
        'instance_work_ledger': instance_work_ledger(rows, edges, cost_origins['sources']) if instance_ready else None,
        'physical_plan_source_audit': {
            'complete': not plan_source_problems, 'exclusions': plan_source_problems, 'plans': plan_sources,
            'scope': 'cost_time_child_dependency_and_generator_sets_not_D_F_credit',
            'selected_child_snapshot_mismatches': sum(not e['used_during_parent_costing']
                                                     for e in costs['selected_edges']) if costs['complete'] else None,
        },
        'terminal_plan_quality': {
            'available': terminal_valid,
            'optimizer_cost': reported_cost if terminal_valid else None,
            'candidate_sequence': roots[0]['candidate_sequence'] if terminal_valid else None,
            'scope': 'complete_policy_outcome_not_per_rule_D_or_F',
            'exclusions': [] if terminal_valid else ['missing_or_inconsistent_selected_root_cost'],
            'comparison_requires': ['same_query_data_statistics_and_cost_model', 'independent_result_validation'],
        },
        'dfc_training_ready': False,
        'remaining_contracts': ['validated_instance_provenance_and_credit_allocation',
                                'deduplicated_induced_search_work', 'frozen_search_work_metric_and_scales'],
    }


def comparison_runs(value, path=''):
    if isinstance(value, dict):
        if 'candidate_events' in value and 'experiment_outcomes' in value:
            yield path, value
        else:
            for key, item in value.items():
                yield from comparison_runs(item, path + '/' + key.replace('~', '~0').replace('/', '~1'))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from comparison_runs(item, path + '/' + str(index))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--run', action='append', help='exact JSON pointer; repeat to audit specific runs')
    parser.add_argument('--dsl-credit-share', type=float,
                        help='explicit [0,1] sensitivity share, not an estimated causal fraction')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    found = dict(comparison_runs(json.loads(args.comparison.read_text())))
    selected = args.run if args.run is not None else list(found)
    if not selected or len(set(selected)) != len(selected) or set(selected) - found.keys():
        parser.error('need distinct existing run pointers')
    report = {'schema_version': 1, 'input': str(args.comparison.resolve()),
              'runs': {path: audit_utility_run(found[path]) for path in selected}}
    if args.dsl_credit_share is not None:
        from ml_orca.objectives.rule_credit import root_gain_credit
        for run in report['runs'].values():
            run['root_gain_credit'] = root_gain_credit(run, dsl_share=args.dsl_credit_share)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({path: {key: run[key] for key in (
        'attempts', 'rules', 'origin_edges', 'missing_source_instance_fields', 'comparable_updates',
        'dfc_training_ready')} for path, run in report['runs'].items()}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
