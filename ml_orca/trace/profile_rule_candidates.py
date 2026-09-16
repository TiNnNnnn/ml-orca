#!/usr/bin/env python3
"""Audit complete CBO attempt traces and plot per-input rejection, time and insertion effects."""

from ml_orca.common.paths import PGORCA_ROOT

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
import re
from pathlib import Path
import zlib

from ml_orca.experiments.plot_stats_sweep import chinese_plotting, measured_points, PLAN_LABELS, profile_points, timing_points
from ml_orca.collect.run_workload_comparison import experiment_run_status, trace_records, plan_difference, plan_tree, timing_plan, profile_comparability


STAGES = {
    "match_rejected": ("match_rejected", "匹配拒绝", "#cc6677"),
    "constraint_rejected": ("constraint_rejected", "约束拒绝", "#ee7733"),
    "instantiate_rejected": ("instantiate_rejected", "构造拒绝", "#aa4499"),
    "duplicate": ("duplicate_alternatives", "构造结果重复", "#999999"),
    "budget_exhausted": ("budget_exhausted", "构造后预算不足", "#882255"),
    "budget_skipped": ("budget_skipped", "预算跳过（未求值）", "#bbbbbb"),
    "ready_cbo": ("generated_alternatives", "生成待入库候选", "#228833"),
}
TIMES = ("match_us", "constraint_us", "instantiate_us")


def input_context_features(context: dict) -> dict:
    """Normalize cached attributes; the caller must audit the capture stage."""
    def node(value):
        source = value.get("stats_source", "unknown")
        rows = value.get("rows")
        available = (source in {"expression", "memo_group"} and type(rows) in {int, float}
                     and math.isfinite(rows) and rows >= 0)
        expressions = value.get("memo_group_expressions")
        props = value.get("logical_properties") or {}
        fields = ("output_columns", "outer_columns", "not_null_columns", "key_count", "join_depth")
        properties_known = (props.get("source") == "complete_memo_group"
                            and all(type(props.get(k)) is int and props[k] >= 0 for k in fields))
        return {"operator": value.get("operator"), "arity": value.get("arity"),
                "stats_source": source, "rows_available": available,
                "rows": rows if available else None,
                "memo_group_expressions": expressions if type(expressions) is int and expressions >= 0 else None,
                "memo_state": {k: (value.get("memo_state") or {}).get(k) for k in (
                    "group_explored", "group_implemented", "expression_explored", "expression_implemented")},
                "logical_properties": {k: props[k] for k in fields} if properties_known else None,
                "empty": value.get("empty") if source in {"expression", "memo_group"} else None}

    children = context.get("children", [])
    return {"root": node(context.get("root", {})),
            "children": [{"position": c["position"], "node": node(c["node"])} for c in children],
            "source_shape": context.get("source_shape"), "source_tree": context.get("source_tree"),
            "relational_children": context.get("relational_children"),
            "omitted_children": context.get("omitted_children")}


def candidate_state(row: dict) -> dict:
    """Partial pre-evaluation observation, not a sufficient/Markov state or an identity."""
    context = row.get("input_context") or {}
    captured = (row.get("evaluated") is True and not row.get("context_resolution_errors")
                and context.get("capture") == "before_evaluation"
                and context.get("scope") == "source_before_match_view")
    if not captured:
        context = {}
    children = context.get("children", [])
    bound_tables = [symbol for symbol in (row.get("binding_context") or {}).get("symbols", [])
                    if isinstance(symbol, dict) and symbol.get("kind") == "table"]
    return {"capture": "before_evaluation", "captured": bool(captured),
            "features": input_context_features(context),
            # Run identity is supplied by the surrounding report. Never pool these IDs as features.
            "provenance": {**{key: row.get(key) for key in (
                "experiment", "sequence", "rule_hash", "group", "group_expression", "memo_version",
                "state_fingerprint", "source_fingerprint", "binding_fingerprint")},
                "root_reference": context.get("root", {}).get("reference_key"),
                "stats_lifecycle_sequence": context.get("stats_lifecycle_sequence"),
                "root_memo": {k: (context.get("root", {}).get("memo_state") or {}).get(k)
                              for k in ("group", "group_expression", "statistics_owner_group")},
                "child_memo": [{"position": c["position"], **{k: (c["node"].get("memo_state") or {}).get(k)
                                for k in ("group", "group_expression", "statistics_owner_group")}} for c in children],
                "bound_table_memo": [{k: symbol.get(k) for k in
                                      ("symbol_index", "memo_group", "memo_group_expression")}
                                     for symbol in bound_tables],
                "child_references": [{"position": c["position"], "reference_key": c["node"].get("reference_key")}
                                     for c in children]}}


def state_coverage(rows: list[dict]) -> dict:
    states = [candidate_state(row) for row in rows]
    features = [s["features"] for s in states if s["captured"]]

    def availability(nodes):
        # Count observed slots, not distinct Memo groups or independent samples.
        return {"observed_nodes": len(nodes),
                "rows_available": sum(n["rows_available"] for n in nodes),
                "logical_properties_available": sum(n["logical_properties"] is not None for n in nodes),
                "memo_group_expressions_available": sum(n["memo_group_expressions"] is not None for n in nodes),
                "stats_sources": dict(sorted(Counter(n["stats_source"] for n in nodes).items()))}

    return {"attempts": len(rows), "captured": len(features),
            "node_availability": {
                "root": availability([f["root"] for f in features]),
                "observed_relational_children": availability([c["node"] for f in features for c in f["children"]])},
            "root_rows_available": sum(f["root"]["rows_available"] for f in features),
            "any_child_rows_available": sum(any(c["node"]["rows_available"] for c in f["children"]) for f in features),
            "all_relational_child_rows_available": sum(
                type(f["relational_children"]) is int and f["relational_children"] > 0
                and f["omitted_children"] == 0 and len(f["children"]) == f["relational_children"]
                and all(c["node"]["rows_available"] for c in f["children"]) for f in features),
            "fully_expanded_source_shape": sum((f["source_shape"] or {}).get("complete") is True
                and f["source_shape"].get("pattern_nodes") == 0 for f in features)}


def subquery_class(shape: dict | None) -> str:
    if not shape or shape.get("complete") is not True or shape.get("pattern_nodes") != 0:
        return "unknown"
    kinds = [op for op, count in shape["operators"].items() if count > 0 and op.startswith("CScalarSubquery")]
    return kinds[0] if len(kinds) == 1 else "mixed_subqueries" if kinds else "no_subquery"


def binding_shape_features(row: dict) -> dict:
    context = row.get("input_context") or {}
    source = context.get("source_shape") if context.get("capture") == "before_evaluation" else None
    model = row.get("binding_context") or {}
    valid = (model.get("capture") == "after_evaluation" and model.get("omitted_symbols") == 0
             and model.get("scope") == "source_table_predicate_symbols")
    symbols = model.get("symbols", [])
    predicates = [s for s in symbols if s["kind"] == "predicate"]
    tables = [s for s in symbols if s["kind"] == "table"]
    predicate = "unknown"
    if valid:
        if not predicates:
            predicate = "no_predicate_symbol"
        elif any(s.get("bound") is not True for s in predicates):
            predicate = "unbound"
        elif any(s.get("derived") is not False for s in predicates):
            predicate = "derived"
        else:
            classes = {subquery_class(s.get("shape")) for s in predicates}
            predicate = "unknown" if "unknown" in classes else next(iter(classes)) if len(classes) == 1 else "mixed_subqueries"
    sizes_known = valid and bool(tables) and all(
        s.get("bound") is True and s.get("derived") is False and s.get("shape")
        and s["shape"].get("complete") is True and s["shape"].get("pattern_nodes") == 0 for s in tables)
    predicate_structure = "unknown"
    if valid and predicates and all(s.get("bound") is True and s.get("derived") is False
            and s.get("shape") and s["shape"].get("complete") is True
            and s["shape"].get("pattern_nodes") == 0 for s in predicates):
        predicate_structure = "constant_only" if all(s["shape"]["operators"] == {"CScalarConst": 1}
                                                     for s in predicates) else "nonconstant_shape"
    table_statistics = ([{"symbol_index": s.get("symbol_index"),
                          "stats_source": s.get("stats_source", "missing"), "rows": s.get("rows")}
                         for s in sorted(tables, key=lambda value: value.get("symbol_index", -1))]
                        if valid else None)
    return {"source_subqueries": subquery_class(source), "bound_predicate_subqueries": predicate,
            "bound_predicate_structure": predicate_structure,
            "bound_table_nodes": sum(s["shape"]["nodes"] for s in tables) if sizes_known else None,
            "bound_table_depth": max(s["shape"]["depth"] for s in tables) if sizes_known else None,
            "bound_table_statistics": table_statistics}


def candidate_evidence(run: dict) -> dict:
    """Fail closed on missing/old/truncated streams; retain partial records for diagnostics."""
    problems = []
    status = experiment_run_status(run, True)
    if status != "ok":
        problems.append(status)
    outcomes = run.get("experiment_outcomes", [])
    final = outcomes[0] if len(outcomes) == 1 else {}
    if final.get("candidate_trace_version") != 2:
        problems.append("candidate_trace_version_missing")
    events = run.get("candidate_events", [])
    attempts = [e for e in events if e["kind"] == "rule_candidate"]
    if final.get("rule_candidates") != len(attempts):
        problems.append("candidate_count_mismatch")
    by_sequence = {}
    for event in attempts:
        seq = event.get("sequence")
        if type(seq) is not int or seq < 1 or seq in by_sequence:
            raise ValueError("invalid or duplicate candidate sequence")
        by_sequence[seq] = event
    if sorted(by_sequence) != list(range(1, len(attempts) + 1)):
        problems.append("candidate_sequence_gap")
    if any(e.get("experiment") != final.get("experiment") for e in events):
        problems.append("experiment_identity_mismatch")
    insertions = {}
    for event in events:
        if event["kind"] != "rule_candidate_outcome":
            continue
        seq = event.get("candidate_sequence")
        candidate = by_sequence.get(seq, {})
        if (seq in insertions or candidate.get("status") != "ready_cbo"
                or candidate.get("rule_hash") != event.get("rule_hash")):
            raise ValueError("orphan, duplicate or mismatched Memo outcome")
        insertions[seq] = event
    rows = []
    for seq, event in sorted(by_sequence.items()):
        if event.get("placement") != "cbo":
            continue
        stage = event.get("status")
        if event.get("context_resolution_errors"):
            problems.append("unresolved_candidate_context")
        if stage not in STAGES:
            problems.append("unknown_cbo_stage")
        evaluated = stage != "budget_skipped"
        context = event.get("input_context")
        if event.get("evaluated") is not evaluated or (evaluated and (
                not context or context.get("capture") != "before_evaluation")):
            problems.append("missing_evaluation_context")
        if any(type(event.get(t)) is not int or event[t] < 0 for t in TIMES):
            problems.append("invalid_candidate_timing")
        outcome = insertions.get(seq)
        if stage == "ready_cbo" and outcome is None:
            problems.append("unresolved_ready_candidate")
        direct = None
        if outcome:
            if outcome.get("status") not in {"memo_inserted", "memo_duplicate", "memo_cycle_rejected"}:
                problems.append("unknown_memo_outcome")
            before = outcome.get("memo_version_at_insertion_start")
            after = outcome.get("memo_version_after")
            if type(before) is int and type(after) is int and 0 <= before <= after:
                direct = after - before
            else:
                problems.append("missing_insertion_interval")
        rows.append({**event, "memo_outcome": outcome, "direct_insertions": direct,
                     "shape_features": binding_shape_features(event), "pre_evaluation_state": candidate_state(event)})
    counters = run.get("dsl_observability", {}).get("rules", {})
    for rule in sorted(set(counters) | {r["rule_hash"] for r in rows}):
        subset = [r for r in rows if r["rule_hash"] == rule]
        expected = counters.get(rule, {})
        counts = Counter(r["status"] for r in subset)
        if expected.get("binding_attempts") != len(subset) or any(
                expected.get(field) != counts[stage] for stage, (field, _, _) in STAGES.items()):
            problems.append(f"rule_counter_mismatch:{rule}")
        for field in TIMES:
            if expected.get(field) != sum(r.get(field, 0) for r in subset if type(r.get(field)) is int):
                problems.append(f"rule_timing_mismatch:{rule}:{field}")
    groups = defaultdict(list)
    for row in rows:
        root = (row.get("input_context") or {}).get("root", {})
        groups[row["rule_hash"], root.get("operator"), row["status"], row.get("failed_constraint")].append(row)
    grouped = [{"rule_hash": rule, "source_operator": op, "status": stage, "failed_constraint": constraint,
                "attempts": len(subset), **{t: sum(r.get(t, 0) for r in subset if type(r.get(t)) is int) for t in TIMES}}
               for (rule, op, stage, constraint), subset in sorted(groups.items(), key=lambda item: str(item[0]))]
    return {"complete": not problems, "exclusions": sorted(set(problems)), "scope": "cbo_dispatched_attempts",
            "attempts": len(rows), "evaluations": sum(r.get("evaluated") is True for r in rows),
            "not_included": ["trie_rejected_rules", "binding_construction_cost", "future_descendant_rewrites"],
            "timing_source": "traced_diagnostics_not_untraced_performance", "rows": rows, "groups": grouped}


def binding_origin_evidence(run: dict) -> dict:
    """Actual extracted-tree origins, not a complete causal or DSL-template graph."""
    audit = candidate_evidence(run)
    problems = list(audit["exclusions"])
    finals = run.get("experiment_outcomes", [])
    final = finals[0] if len(finals) == 1 else {}
    version = final.get("binding_edge_trace_version")
    if version not in (1, 2, 3, 4, 5, 6):
        problems.append("binding_edge_trace_version_missing")
    edges = run.get("rule_edges")
    if not isinstance(edges, list):
        problems.append("binding_edges_missing")
        edges = []
    edges = [e for e in edges if e.get("scheduler") == "cbo"]
    if final.get("binding_origin_edges") != len(edges):
        problems.append("binding_edge_count_mismatch")
    if [e.get("binding_edge_sequence") for e in edges] != list(range(1, len(edges) + 1)):
        problems.append("binding_edge_sequence_gap_or_duplicate")
    candidates = {r["sequence"]: r for r in audit["rows"]}
    observed = set()
    for edge in edges:
        if 'experiment' in edge and edge['experiment'] != final.get('experiment'):
            problems.append('binding_edge_experiment_mismatch')
        candidate = candidates.get(edge.get("dst_candidate_sequence"), {})
        if (candidate.get("evaluated") is not True or candidate.get("rule_hash") != edge.get("dst_rule")
                or candidate.get("status") != edge.get("candidate_status")):
            problems.append("unresolved_binding_edge_candidate")
        path = edge.get("dst_binding_path")
        if not isinstance(path, str) or re.fullmatch(r"r(?:/[0-9]+)*", path) is None:
            problems.append("invalid_binding_edge_path")
        elif edge.get("dst_source_path") != ("r" if path == "r" else None):
            problems.append("binding_path_mislabeled_as_template")
        if any(type(edge.get(k)) is not int or edge[k] < 0 for k in ("binding_group", "binding_group_expression")):
            problems.append("invalid_binding_edge_memo_identity")
        relation = edge.get("producer_relation")
        if (relation not in {"memo_consumes", "input_exposes"} or not edge.get("src_rule")
                or edge.get("relation") != (relation if candidate.get("status") == "ready_cbo" else "binding_observed")):
            problems.append("invalid_binding_edge_relation")
        producer_outcomes = {"memo_inserted", "memo_duplicate"}
        if version in (3, 4, 5, 6):
            producer_outcomes.add("memo_rehashed")
        if version in (2, 3, 4, 5, 6) and edge.get("producer_outcome") not in producer_outcomes:
            problems.append("invalid_binding_producer_outcome")
        identity = (edge.get("dst_candidate_sequence"), path)
        if version in (2, 3, 4, 5, 6):
            identity += (edge.get("src_rule"), edge.get("src_target_path"), relation,
                         edge.get("producer_outcome"))
        if version in (4, 5, 6):
            seq = edge.get('src_candidate_sequence')
            source = candidates.get(seq, {})
            dst = edge.get('dst_candidate_sequence')
            if (type(seq) is not int or type(dst) is not int or not 0 < seq < dst
                    or source.get('rule_hash') != edge.get('src_rule')
                    or source.get('evaluated') is not True or source.get('status') != 'ready_cbo'):
                problems.append('unresolved_binding_edge_producer_instance')
            identity += (seq,)
        if identity in observed:
            problems.append("duplicate_binding_position")
        observed.add(identity)
    return {"complete": not problems, "exclusions": sorted(set(problems)), "edges": edges,
            "producer_instance_coverage": "validated_source_attempts" if version in (4, 5, 6) else "unavailable",
            "producer_coverage": ("observed_before_binding_including_memo_duplicates_and_rehash" if version in (3, 4, 5, 6) else
                                  "observed_before_binding_including_memo_duplicates" if version == 2 else "first_inserter_only"),
            "scope": "direct_DSL_origins_in_extracted_CBO_bindings_including_failed_evaluations",
            "not_included": ["unbound_memo_alternatives", "template_path_mapping", "all_causal_enablers",
                             "transitive_native_xform_origins",
                             "producers_observed_after_binding"] +
                            (["origins_coalesced_by_later_group_rehash"] if version not in (3, 4, 5, 6) else []) +
                            (["duplicate_candidate_producers"] if version not in (2, 3, 4, 5, 6) else [])}


def cost_evidence(run: dict) -> dict:
    """Audit the shared costing boundary; earlier property/job rejections are not included."""
    problems = []
    if experiment_run_status(run, True) != "ok":
        problems.append("unsuccessful_experiment")
    outcomes = run.get("experiment_outcomes", [])
    final = outcomes[0] if len(outcomes) == 1 else {}
    events = run.get("cost_events", [])
    if final.get("cost_trace_version") != 1:
        problems.append("cost_trace_version_missing")
    if final.get("cost_candidates") != len(events):
        problems.append("cost_count_mismatch")
    if [e.get("sequence") for e in events] != list(range(1, len(events) + 1)):
        problems.append("cost_sequence_gap_or_duplicate")
    statuses = {"invalid_context", "duplicate_context", "invalid_plan", "costed", "pruned"}
    for event in events:
        if event.get("experiment") != final.get("experiment") or event.get("status") not in statuses:
            problems.append("invalid_cost_identity_or_status")
        if any(type(event.get(k)) is not int or event[k] < 0 for k in (
                "group", "group_expression", "optimization_context", "optimization_request", "search_stage",
                "memo_version", "preceding_rule_candidates")):
            problems.append("invalid_cost_context")
        before = event.get("preceding_rule_candidates")
        total = final.get("rule_candidates")
        if type(before) is not int or type(total) is not int or not 0 <= before <= total:
            problems.append("invalid_candidate_watermark")
        if event.get("status") in {"costed", "pruned"}:
            value = event.get("cost")
            expected = "lower_bound" if event["status"] == "pruned" else "computed"
            if (type(value) not in {int, float} or not math.isfinite(value) or value < 0
                    or event.get("cost_kind") != expected):
                problems.append("invalid_cost_or_bound")
        if "rows" in event and (type(event["rows"]) not in {int, float}
                                 or not math.isfinite(event["rows"]) or event["rows"] < 0):
            problems.append("invalid_cost_rows")
        if "origin_chain" in event:
            chain = event["origin_chain"]
            if not isinstance(chain, list) or any(not isinstance(node, dict) or any(
                    type(node.get(key)) is not int or node[key] < 0
                    for key in ("group", "group_expression")) for node in chain):
                problems.append("invalid_origin_chain")
            else:
                identities = [(node["group"], node["group_expression"]) for node in chain]
                immediate = (event.get("origin_group"), event.get("origin_expression"))
                if len(set(identities)) != len(identities):
                    problems.append("cyclic_origin_chain")
                if (identities and identities[0] != immediate) or (not identities and immediate != (None, None)):
                    problems.append("origin_chain_head_mismatch")
    observed_groups = {e["group"] for e in run.get("stats_events", []) if type(e.get("group")) is int}
    return {"complete": not problems, "exclusions": sorted(set(problems)), "events": events,
            "scope": "PccComputeCost_before_best_context_insertion",
            "status_counts": dict(Counter(e.get("status") for e in events)),
            "stats_groups_observed": len(observed_groups),
            "events_in_observed_stats_group": sum(e.get("group") in observed_groups for e in events),
            "costed_with_rows": sum(e.get("status") == "costed" and "rows" in e for e in events),
            "not_included": ["earlier_property_and_job_rejections", "all_pruning_checks",
                             "best_context_retention", "final_plan_selection", "causal_rule_attribution"],
            "group_overlap_is_not_stats_consumption_proof": True}


def render_costs(report: dict, output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    events = report["events"]
    labels = {"costed": "完成计价", "pruned": "下界剪枝", "duplicate_context": "重复上下文",
              "invalid_context": "上下文无效", "invalid_plan": "计划无效"}
    counts = Counter(e["status"] for e in events)
    bars = axes[0, 0].bar(list(labels.values()), [counts[k] for k in labels])
    axes[0, 0].bar_label(bars)
    axes[0, 0].set(title="共享成本入口的结果分布", ylabel="事件次数")
    for status, label in labels.items():
        rows = [e for e in events if e["status"] == status]
        axes[0, 1].scatter([e["sequence"] for e in rows], [e["preceding_rule_candidates"] for e in rows],
                           label=label, s=12, alpha=0.5)
    axes[0, 1].set(title="逻辑尝试与成本搜索的先后关系", xlabel="成本事件序号", ylabel="此前已记录规则尝试数")
    axes[0, 1].legend(fontsize=8)
    costed = [e for e in events if e["status"] == "costed" and "rows" in e]
    axes[1, 0].scatter([e["rows"] for e in costed], [e["cost"] for e in costed], s=15, alpha=0.5)
    axes[1, 0].set(xscale="symlog", yscale="symlog", title="已计价候选的缓存行数与估算成本",
                   xlabel="估计行数（含零，对称对数轴）", ylabel="计算成本；排除下界与缺失值")
    contexts = Counter((e["group"], e["optimization_context"]) for e in events)
    axes[1, 1].hist(list(contexts.values()), bins=min(20, max(1, len(contexts))))
    axes[1, 1].set(title="每个组内优化上下文收到的成本尝试", xlabel="成本事件数", ylabel="上下文数量")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    audit = "事件流完整" if report["complete"] else "不完整，仅供诊断"
    fig.suptitle(f"物理成本阶段的局部状态观测｜{audit}\n"
                 "成本与行数为求值后观测，不回填匹配前特征；并非最终选中计划或规则因果收益")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def search_check_evidence(run: dict) -> dict:
    costs = cost_evidence(run)
    problems = list(costs["exclusions"])
    outcomes = run.get("experiment_outcomes", [])
    final = outcomes[0] if len(outcomes) == 1 else {}
    events = run.get("search_checks", [])
    if final.get("search_check_version") != 1:
        problems.append("search_check_version_missing")
    if final.get("search_checks") != len(events):
        problems.append("search_check_count_mismatch")
    if [e.get("sequence") for e in events] != list(range(1, len(events) + 1)):
        problems.append("search_check_sequence_gap")
    statuses = {"prune": {"disabled", "dpe_unsafe", "no_incumbent", "pruned", "bound_not_worse"},
                "properties": {"accepted", "missing_columns", "sort_without_order", "motion_any_distribution",
                               "spool_without_rewind", "partition_selector_without_scan"}}
    candidates = {e["sequence"]: e for e in costs["events"] if type(e.get("sequence")) is int}
    margins = []
    for event in events:
        if (event.get("experiment") != final.get("experiment")
                or event.get("status") not in statuses.get(event.get("check"), set())):
            problems.append("invalid_search_check_identity_or_status")
        watermark = event.get("preceding_cost_candidates")
        rule_watermark = event.get("preceding_rule_candidates")
        if (type(watermark) is not int or not 0 <= watermark <= len(costs["events"])
                or type(rule_watermark) is not int or type(final.get("rule_candidates")) is not int
                or not 0 <= rule_watermark <= final["rule_candidates"]):
            problems.append("invalid_search_check_watermark")
        for field in ("incumbent_candidate", "optimized_child_candidate"):
            if field not in event:
                continue
            seq = event[field]
            candidate = candidates.get(seq)
            if (type(seq) is not int or candidate is None or type(watermark) is not int or seq > watermark
                    or candidate.get("status") not in {"costed", "pruned"}):
                problems.append("unresolved_search_check_cost_reference")
        compared = event.get("check") == "prune" and event.get("status") in {"pruned", "bound_not_worse"}
        if compared:
            bound, incumbent = event.get("lower_bound"), event.get("incumbent_cost")
            candidate = candidates.get(event.get("incumbent_candidate"), {})
            if (any(type(x) not in {int, float} or not math.isfinite(x) or x < 0 for x in (bound, incumbent))
                    or type(candidate.get("cost")) not in {int, float}
                    or not math.isclose(incumbent, candidate["cost"], rel_tol=1e-8, abs_tol=1e-8)):
                problems.append("invalid_pruning_comparison")
                continue
            # Trace decimals can round a near-tie. Keep the optimizer's decision, never re-decide it.
            margin = bound - incumbent
            tolerance = 1e-6 * max(1, bound, incumbent)
            if (event["status"] == "pruned" and margin < -tolerance
                    or event["status"] == "bound_not_worse" and margin > tolerance):
                problems.append("contradictory_pruning_comparison")
            margins.append({**event, "margin": margin, "relative_margin": margin / incumbent if incumbent > 0 else None,
                            "near_display_precision": abs(margin) <= tolerance})
        elif "lower_bound" in event or "incumbent_cost" in event:
            problems.append("unexpected_pruning_comparison")
    return {"complete": not problems, "exclusions": sorted(set(problems)), "events": events, "margins": margins,
            "status_counts": {check: dict(Counter(e["status"] for e in events if e.get("check") == check)) for check in statuses},
            "scope": "FSafeToPrune_and_FCheckReqdProps", "not_included": ["all_job_scheduler_rejections", "causal_rule_attribution"],
            "near_ties_keep_optimizer_decision": True}


def render_search_checks(report: dict, output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    labels = {"prune": {"disabled": "未开启", "dpe_unsafe": "动态分区统计不安全", "no_incumbent": "尚无最优候选",
                          "pruned": "下界超过最优", "bound_not_worse": "下界未超过最优"},
              "properties": {"accepted": "通过", "missing_columns": "缺少所需列", "sort_without_order": "无排序需求",
                  "motion_any_distribution": "任意分布无需移动", "spool_without_rewind": "无需回卷",
                  "partition_selector_without_scan": "无对应分区扫描"}}
    for axis, check, title in zip(axes[:2], ("properties", "prune"), ("前置属性检查", "每次剪枝检查的实际结果")):
        counts = report["status_counts"][check]
        keys = [key for key in labels[check] if counts.get(key, 0)]
        bars = axis.bar([labels[check][k] for k in keys], [counts[k] for k in keys])
        axis.bar_label(bars)
        axis.set(title=title, ylabel="检查次数")
        axis.tick_params(axis="x", labelrotation=25)
    for status, label in (("pruned", "剪枝"), ("bound_not_worse", "继续搜索")):
        rows = [r for r in report["margins"] if r["status"] == status and r["relative_margin"] is not None]
        axes[2].scatter([r["sequence"] for r in rows], [r["relative_margin"] for r in rows], s=13, alpha=0.6, label=label)
    axes[2].axhline(0, color="gray", linewidth=0.7)
    axes[2].set(title="距离剪枝阈值还有多远", xlabel="检查事件序号", ylabel="（下界 − 当前最优）／当前最优", yscale="symlog")
    axes[2].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    audit = "引用与比较审计通过" if report["complete"] else "不完整，仅供诊断"
    fig.suptitle(f"搜索边界的细粒度观测｜{audit}\n实际比较值，不用事后成本冒充剪枝阈值；零成本不计算相对距离")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def cost_lifecycle_evidence(run: dict) -> dict:
    costs = cost_evidence(run)
    problems = list(costs["exclusions"])
    outcomes = run.get("experiment_outcomes", [])
    final = outcomes[0] if len(outcomes) == 1 else {}
    events = run.get("cost_lifecycle_events", [])
    if final.get("cost_lifecycle_version") != 1:
        problems.append("cost_lifecycle_version_missing")
    if final.get("cost_lifecycle_events") != len(events):
        problems.append("cost_lifecycle_count_mismatch")
    if [e.get("sequence") for e in events] != list(range(1, len(events) + 1)):
        problems.append("cost_lifecycle_sequence_gap")
    candidates = {e["sequence"]: e for e in costs["events"] if e.get("status") in {"costed", "pruned"}
                  and type(e.get("sequence")) is int and e["sequence"] > 0}
    decisions, active, best, selected = {}, set(), {}, []
    for event in events:
        seq = event.get("candidate_sequence")
        candidate = candidates.get(seq)
        if event.get("experiment") != final.get("experiment") or type(seq) is not int or candidate is None:
            problems.append("unresolved_cost_lifecycle_reference")
            continue
        status = event.get("status")
        previous = event.get("previous_candidate_sequence", 0)
        if previous and previous not in candidates:
            problems.append("unresolved_previous_cost_candidate")
        if status in {"retained_new", "retained_replacement", "discarded"}:
            if seq in decisions:
                problems.append("duplicate_retention_decision")
            decisions[seq] = status
            if status == "retained_new":
                if previous != 0:
                    problems.append("unexpected_previous_candidate")
                active.add(seq)
            else:
                if previous not in active:
                    problems.append("previous_candidate_not_retained")
                if status == "retained_replacement":
                    active.discard(previous)
                    active.add(seq)
        elif status == "best_updated":
            key = (event.get("group"), event.get("optimization_context"))
            if seq not in active or previous != best.get(key, 0):
                problems.append("inconsistent_best_update")
            best[key] = seq
        elif status == "selected_plan":
            if seq not in active or candidate["status"] != "costed":
                problems.append("selected_candidate_not_retained_and_costed")
            selected_cost, computed_cost = event.get("cost"), candidate.get("cost")
            if (event.get("operator") != candidate.get("operator")
                    or any(type(c) not in {int, float} or not math.isfinite(c) for c in (selected_cost, computed_cost))
                    or not math.isclose(selected_cost, computed_cost, rel_tol=1e-8, abs_tol=1e-8)):
                problems.append("selected_candidate_mismatch")
            node, parent = event.get("plan_node"), event.get("parent_plan_node")
            if type(node) is not int or type(parent) is not int or not 0 <= parent < node:
                problems.append("invalid_selected_plan_position")
                continue
            selected.append(event)
        else:
            problems.append("unknown_cost_lifecycle_status")
    if set(decisions) != set(candidates):
        problems.append("missing_retention_decisions")
    if len({e["plan_node"] for e in selected}) != len(selected):
        problems.append("duplicate_selected_plan_position")
    if not selected:
        problems.append("missing_selected_physical_plan")
    if any(e["candidate_sequence"] not in best.values() for e in selected):
        problems.append("selected_candidate_not_current_best")
    child_links = []
    for seq, candidate in candidates.items():
        for child in candidate.get("child_contexts", []):
            child_seq = child.get("cost_candidate_sequence")
            source = candidates.get(child_seq)
            if (type(child_seq) is not int or source is None or child_seq >= seq or source["status"] != "costed"
                    or source.get("group") != child.get("group")
                    or source.get("optimization_context") != child.get("optimization_context")):
                problems.append("unresolved_costed_child_reference")
            child_links.append({"parent_candidate": seq, "child_candidate": child_seq})
    selected_by_node = {e["plan_node"]: e for e in selected}
    selected_edges = []
    for child in selected:
        parent = selected_by_node.get(child["parent_plan_node"])
        if parent is None:
            continue  # Scalar parents are not costed physical occurrences.
        at_cost = candidates[parent["candidate_sequence"]].get("child_contexts", [])
        selected_edges.append({"parent_node": parent["plan_node"], "child_node": child["plan_node"],
            "parent_candidate": parent["candidate_sequence"], "child_candidate": child["candidate_sequence"],
            "used_during_parent_costing": any(c.get("cost_candidate_sequence") == child["candidate_sequence"] for c in at_cost)})
    return {"complete": not problems, "exclusions": sorted(set(problems)), "events": events,
            "status_counts": dict(Counter(e.get("status") for e in events)),
            "materialized_candidates": len(candidates), "retained_without_observed_replacement": len(active),
            "ever_best_candidates": len({e["candidate_sequence"] for e in events if e.get("status") == "best_updated"}),
            "selected_occurrences": len(selected), "selected_distinct_candidates": len({e["candidate_sequence"] for e in selected}),
            "selected": selected, "cost_candidates": costs["events"],
            "cost_child_links": child_links, "selected_edges": selected_edges,
            "scope": "observed_retention_best_updates_and_final_memo_extraction",
            "not_guaranteed": ["causal_rule_benefit", "ranked_alternative_exact_cost_identity",
                               "all_pruning_reasons", "retained_candidates_survive_future_context_cleanup"]}


def render_cost_lifecycle(report: dict, output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    labels = {"retained_new": "首次保留", "retained_replacement": "替换旧候选",
              "discarded": "未获保留", "best_updated": "更新当前最优", "selected_plan": "最终物理节点"}
    bars = axes[0].bar(list(labels.values()), [report["status_counts"].get(k, 0) for k in labels])
    axes[0].bar_label(bars)
    axes[0].set(title="计价后的保留与选择事件", ylabel="事件次数；非同一分母的转化率")
    values = [report[k] for k in ("materialized_candidates", "retained_without_observed_replacement", "ever_best_candidates", "selected_distinct_candidates")]
    bars = axes[1].bar(["已建成本候选", "未观察到替换", "曾被设为最优", "最终采用去重"], values)
    axes[1].bar_label(bars)
    axes[1].set(title="候选身份的不同层次", ylabel="候选数量；下界候选也可能保留")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelrotation=15)
    audit = "引用与状态迁移审计通过" if report["complete"] else "不完整，仅供诊断"
    fig.suptitle(f"物理候选生命周期｜{audit}\n最终节点由提取时的候选序号关联，不按相近成本猜测；不是规则收益")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_selected_cost_tree(report: dict, output: Path, font: Path) -> None:
    selected = report["selected"]
    # ponytail: bounded visual preview; the JSON retains every node for large plans.
    if not selected or len(selected) > 80:
        return
    plt = chinese_plotting(font)
    depth, levels = {}, defaultdict(list)
    for event in sorted(selected, key=lambda e: e["plan_node"]):
        node = event["plan_node"]
        depth[node] = depth.get(event["parent_plan_node"], -1) + 1
        levels[depth[node]].append(node)
    positions = {node: ((i + 1) / (len(nodes) + 1), -level)
                 for level, nodes in levels.items() for i, node in enumerate(nodes)}
    fig, axis = plt.subplots(figsize=(12, max(6, len(levels) * 0.85)), layout="constrained")
    for edge in report["selected_edges"]:
        axis.annotate("", xy=positions[edge["child_node"]], xytext=positions[edge["parent_node"]],
                      arrowprops={"arrowstyle": "->", "color": "#228833" if edge["used_during_parent_costing"] else "#ee7733",
                                  "shrinkA": 20, "shrinkB": 20})
    names = {"CPhysicalTableScan": "表扫描", "CPhysicalFilter": "过滤", "CPhysicalComputeScalar": "计算",
             "CPhysicalInnerHashJoin": "内哈希连接", "CPhysicalInnerNLJoin": "内嵌套循环连接",
             "CPhysicalLimit": "行数限制", "CPhysicalSort": "排序", "CPhysicalSequence": "顺序执行",
             "CPhysicalCTEProducer": "公共表达式生产", "CPhysicalCTEConsumer": "公共表达式消费",
             "CPhysicalHashAgg": "哈希聚合", "CPhysicalScalarAgg": "标量聚合", "CPhysicalStreamAgg": "流式聚合"}
    for event in selected:
        label = f"{names.get(event['operator'], '物理算子')} · 节点{event['plan_node']}\n成本候选 {event['candidate_sequence']}"
        axis.text(*positions[event["plan_node"]], label, ha="center", va="center", fontsize=8,
                  bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef3fa", "edgecolor": "#4477aa"})
    axis.set(xlim=(0, 1), ylim=(-len(levels), 0.7))
    axis.axis("off")
    audit = "审计通过" if report["complete"] else "不完整，仅供诊断"
    fig.suptitle(f"最终物理计划与成本候选的精确关联｜{audit}\n箭头：父计划 → 子计划；绿线为父计价时所用候选，橙线为未匹配该引用\n仅画物理节点；完整算子和引用见生命周期结果文件，不代表规则因果收益")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_cost_response(experiments: list[dict], output: Path, font: Path, artifact_dir: Path) -> None:
    """Keep all interventions visible, including failed/incomplete points."""
    response = []
    for index, experiment in enumerate(experiments, 1):
        run = experiment["modes"]["replacement"]
        audit = cost_evidence(run)
        outcome = next(iter(run.get("experiment_outcomes", [])), {})
        comparison = dict(experiment["comparisons"]["replacement"])
        original_plan_class = comparison.get("plan_comparison")
        baseline_plan = artifact_dir / "replacement.plan.json"
        observed_plan = artifact_dir / f"stats-{index}.replacement.plan.json"
        reclassified = baseline_plan.is_file() and observed_plan.is_file()
        if reclassified:
            comparison["plan_comparison"] = plan_difference(plan_tree(baseline_plan.read_text()), plan_tree(observed_plan.read_text()))
        valid = audit["complete"] and comparison.get("rows_equal") is True
        checks = search_check_evidence(run)
        response.append({"experiment": experiment["path"], "complete": valid,
            "exclusions": audit["exclusions"], "comparison": comparison,
            "original_plan_class": original_plan_class, "reclassified_saved_plans": reclassified,
            "targets": outcome.get("stats_targets"),
            "counts": audit["status_counts"] if valid else None,
            "checks": checks["status_counts"] if valid and checks["complete"] else None,
            "check_exclusions": checks["exclusions"],
            "rule_attempts": outcome.get("rule_candidates") if valid else None,
            "rule_applications": outcome.get("rule_applications") if valid else None,
            "selected_plan_cbo_dsl_rules": outcome.get("selected_plan_cbo_dsl_rules") if valid else None,
            "memo_expressions": outcome.get("memo_group_expressions") if valid else None,
            "optimizer_cost": outcome.get("optimizer_cost") if valid else None})
    output.with_suffix(".json").write_text(json.dumps(response, indent=2) + "\n")
    plt = chinese_plotting(font)
    fig, panels = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    axes = panels.flat
    positions = list(range(len(response)))
    bottom = [0] * len(response)
    for status, label in (("costed", "完成计价"), ("pruned", "下界剪枝"),
                          ("duplicate_context", "重复上下文"), ("invalid_context", "上下文无效"),
                          ("invalid_plan", "计划无效")):
        values = [(r["counts"] or {}).get(status, 0) if r["complete"] else math.nan for r in response]
        axes[0].bar(positions, values, bottom=bottom, label=label)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0].set(title="物理成本入口：次数可能保持不变", ylabel="事件次数；失败留空")
    axes[0].legend(fontsize=8)
    bars = axes[1].bar(positions, [r["optimizer_cost"] if r["optimizer_cost"] is not None else math.nan for r in response])
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set(title="最终选中计划的估算成本响应", ylabel="优化器成本，不是实际执行时间")
    for check, status, label in (("prune", "pruned", "下界剪枝"), ("prune", "bound_not_worse", "下界未超过最优")):
        axes[2].plot(positions, [r["checks"][check].get(status, 0) if r["checks"] is not None else math.nan for r in response],
                     "o", label=label)
    axes[2].set(title="逐次阈值检查：搜索分支是否改变", ylabel="检查次数；各点为独立配置")
    axes[2].legend()
    for field, label in (("rule_attempts", "规则尝试"), ("memo_expressions", "Memo 表达式")):
        axes[3].plot(positions, [r[field] if r[field] is not None else math.nan for r in response], "o", label=label)
    axes[3].set(title="逻辑尝试与 Memo 规模单独观察", ylabel="计数；不是完整空间等价判断")
    axes[3].legend()
    for axis in axes:
        labels = ["仅观测" if r["targets"] == [] else
                  f"{ {'CLogicalGet': '叶子', 'CLogicalInnerJoin': '连接', 'CLogicalSelect': '过滤'}.get(r['targets'][0]['operator'], '目标')}\n{r['targets'][0]['requested_rows']:.0f}\n行" if r["targets"] and len(r["targets"]) == 1
                  else f"实验{i + 1}：多目标或未知" for i, r in enumerate(response)]
        axis.set_xticks(positions, labels)
        axis.set_xlabel("横轴行数取整显示；精确请求与配置见同名结果文件")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("同一查询不同统计假设：搜索次数与成本分开刻画\n有限干预点，不是规则的总体收益函数")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_shapes(report: dict, output: Path, font: Path, rule: str | None) -> None:
    rows = [r for r in report["rows"] if rule is None or r["rule_hash"] == rule]
    if not rows:
        raise ValueError("no CBO attempts for the requested rule")
    plt = chinese_plotting(font)
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    labels = {"unknown": "未知／未展开／截断", "no_subquery": "无标量子查询", "unbound": "尚未绑定",
              "derived": "约束派生，非直接输入", "no_predicate_symbol": "无谓词占位符",
              "mixed_subqueries": "多种子查询类别", "CScalarSubqueryExists": "存在子查询",
              "CScalarSubqueryNotExists": "不存在子查询", "CScalarSubqueryAny": "任一量化子查询",
              "CScalarSubqueryAll": "全部量化子查询", "CScalarSubquery": "单值子查询"}
    for axis, field, title in zip(axes[0], ("source_subqueries", "bound_predicate_subqueries"),
                                 ("求值前：完整来源表达式的子查询类别", "求值后：实际绑定谓词的子查询类别")):
        classes = sorted({r["shape_features"][field] for r in rows})
        bottom = [0] * len(classes)
        for stage, (_, label, color) in STAGES.items():
            values = [sum(r["status"] == stage and r["shape_features"][field] == c for r in rows) for c in classes]
            if any(values):
                axis.bar([labels.get(c, "其他子查询类别") for c in classes], values, bottom=bottom, label=label, color=color)
                bottom = [b + v for b, v in zip(bottom, values)]
        axis.set(title=title, ylabel="已调度尝试数（不是独立随机样本）")
        axis.set_ylim(0, max(bottom) * 1.12)
        for index, total in enumerate(bottom):
            axis.annotate(str(total), (index, total), xytext=(0, 3),
                          textcoords="offset points", ha="center", fontsize=9)
        axis.tick_params(axis="x", labelrotation=20)
        axis.legend(fontsize=8)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    for stage, (_, label, color) in STAGES.items():
        subset = [r for r in rows if r["status"] == stage and r["shape_features"]["bound_table_nodes"] is not None]
        if not subset:
            continue
        x = [r["shape_features"]["bound_table_nodes"] for r in subset]
        axes[1, 0].scatter(x, [r["shape_features"]["bound_table_depth"] for r in subset],
                           label=label, color=color, s=35, alpha=0.6)
        axes[1, 1].scatter(x, [sum(r[t] for t in TIMES) for r in subset], color=color, s=35, alpha=0.6)
    axes[1, 0].set(title="实际占位子树：规模与深度", ylabel="最大子树深度")
    axes[1, 1].set(title="占位子树规模与本次总求值开销", ylabel="匹配＋约束＋构造（微秒）")
    for axis in axes[1]:
        axis.set_xlabel("已绑定输入子树节点数之和（含标量；重叠子树重复计数）")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    if axes[1, 0].collections:
        axes[1, 0].legend(fontsize=8)
    missing = sum(r["shape_features"]["bound_table_nodes"] is None for r in rows)
    audit = "事件流审计通过" if report["complete"] else "事件流不完整，仅作诊断"
    fig.suptitle(f"规则输入形态剖析｜{rule or '全部已调度规则'}｜{len(rows)} 次尝试｜{audit}\n"
                 f"下排排除占位规模未知的 {missing} 次尝试；未知不填零，重叠点可能遮盖\n"
                 "绑定模型在求值后取得，不能冒充调度前特征；观测关联不是因果结论")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)


def family_design_weights(cohort: dict) -> dict:
    selected = [q for q in cohort["queries"] if q["selected"]]
    unit = cohort.get("sampling_unit")
    if unit in (None, "uniform_discovery_family_then_uniform_member"):
        return {q["query"]: 1.0 for q in selected}
    if unit != "stratified_discovery_family_then_uniform_member":
        raise ValueError("unknown family sampling design")
    strata = {s["stratum"]: s for s in cohort["strata"]}
    if len(strata) != len(cohort["strata"]):
        raise ValueError("duplicate sampling stratum")
    counts = Counter(q.get("stratum") for q in selected)
    if set(counts) != set(strata):
        raise ValueError("sampled strata do not cover the declared frame")
    if any(q.get("split") != "discovery" or q["family"] in cohort.get("excluded_prior_families", [])
           for q in selected):
        raise ValueError("stratified sample must contain only previously unseen discovery families")
    weights = {}
    for key, stratum in strata.items():
        size, take = stratum["families"], stratum["sampled_families"]
        if (type(size) is not int or type(take) is not int or not 0 < take <= size
                or counts[key] != take):
            raise ValueError("invalid stratum population or sample count")
        for query in selected:
            if query.get("stratum") != key:
                continue
            pi = query.get("family_inclusion_probability")
            if (type(pi) not in (int, float) or not math.isfinite(pi) or not 0 < pi <= 1
                    or not math.isclose(pi, take / size, rel_tol=1e-12, abs_tol=0)):
                raise ValueError("invalid family inclusion probability")
            weights[query["query"]] = size / take
    if sum(s["families"] for s in strata.values()) != cohort["eligible_families"]:
        raise ValueError("sampling frame size mismatch")
    return weights


def injection_order(records: list[dict], audit: dict) -> dict:
    """Log-order evidence, not a claim that an injected value reached a matcher."""
    attempts, injections = [], []
    for record in records:
        if record.get("kind") == "rule_candidate" and record.get("placement") == "cbo":
            attempts.append((record["sequence"], record["rule_hash"], record["status"]))
        elif record.get("kind") == "stats_injection":
            injections.append({**record, "preceding_attempts": len(attempts)})
    if attempts != [(r["sequence"], r["rule_hash"], r["status"]) for r in audit["rows"]]:
        raise ValueError("raw trace and comparison candidate sequences differ")
    sources = Counter((r.get("input_context") or {}).get("root", {}).get("stats_source", "unknown")
                      for r in audit["rows"] if r.get("evaluated"))
    return {"injections": injections, "root_stats_sources": dict(sources),
            "attempts_before_first_injection": injections[0]["preceding_attempts"] if injections else None,
            "attempts_after_first_injection": len(attempts) - injections[0]["preceding_attempts"] if injections else None}


def rule_stage_counts(rows: list[dict]) -> dict:
    """Aggregate already audited candidates; do not infer dependency edges from counts."""
    rules = defaultdict(Counter)
    for row in rows:
        counts = rules[row["rule_hash"]]
        counts["attempts"] += 1
        counts[row["status"]] += 1
        counts["direct_insertions"] += row["direct_insertions"] or 0
        if row["memo_outcome"]:
            counts[row["memo_outcome"]["status"]] += 1
        if row.get("failed_constraint"):
            counts["constraint:" + row["failed_constraint"]] += 1
    return {r: dict(c) for r, c in rules.items()}


def sweep_candidate_evidence(manifest: dict, comparison: dict, manifest_dir: Path, traces: Path) -> list[dict]:
    """Audit injected runs before computing paired rule responses to a 1x control."""
    identities = [(t["fingerprint"], t["operator"]) for t in manifest["targets"]]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("sweep requires nonempty distinct targets")
    points = measured_points(manifest, comparison, manifest_dir, "replacement")
    experiments = {str(Path(e["path"]).resolve()): (i, e) for i, e in enumerate(comparison["stats_experiments"], 1)}
    for point in points:
        point.update(rules=None, order=None, paired=False)
        if point["status"] != "ok":
            continue
        index, experiment = experiments[str((manifest_dir / point["file"]).resolve())]
        run = experiment["modes"]["replacement"]
        try:
            audit = candidate_evidence(run)
            if not audit["complete"]:
                point.update(status="incomplete_trace", exclusions=audit["exclusions"])
                continue
            if any(e.get("placement") != "cbo" for e in run["candidate_events"] if e["kind"] == "rule_candidate"):
                raise ValueError("non-CBO candidate in CBO response experiment")
            trace = traces / f"stats-{index}.replacement.trace"
            records = trace_records(trace.read_text() if trace.is_file() else
                                    gzip.decompress(trace.with_suffix(".trace.gz").read_bytes()).decode())
            plan = timing_plan(plan_tree((traces / f"stats-{index}.replacement.plan.json").read_text()))
            if plan is None:
                raise ValueError("missing or invalid plan artifact")
            order = injection_order(records, audit)
            expected = dict(zip(identities, point["requested_rows"]))
            if {(e["fingerprint"], e["operator"]) for e in order["injections"]} != set(expected) or any(
                    e["experiment"] != run["experiment_outcomes"][0]["experiment"]
                    or not math.isclose(e["rows"], expected[e["fingerprint"], e["operator"]],
                                        rel_tol=1e-9, abs_tol=1e-6)
                    for e in order["injections"]):
                raise ValueError("injection log does not match requested target/value")
        except (ValueError, OSError) as error:
            point.update(status="invalid_trace", exclusions=[str(error)])
            continue
        point.update(rules=rule_stage_counts(audit["rows"]), order=order, plan=plan,
                     stage_path=[(r["rule_hash"], r["status"], r.get("failed_constraint")) for r in audit["rows"]])
        # Physical completeness is independent: never turn missing events into zero work.
        physical = {name: check(run) for name, check in (
            ("cost", cost_evidence), ("search", search_check_evidence), ("lifecycle", cost_lifecycle_evidence))}
        point["physical_audit"] = {name: {k: evidence[k] for k in ("complete", "exclusions")}
                                   for name, evidence in physical.items()}
        point["physical_counts"] = ({
            **{"cost:" + k: v for k, v in physical["cost"]["status_counts"].items()},
            **{"search:" + check + ":" + k: v for check, counts in physical["search"]["status_counts"].items()
               for k, v in counts.items()},
            **{"lifecycle:" + k: v for k, v in physical["lifecycle"]["status_counts"].items()}}
            if all(a["complete"] for a in physical.values()) else None)
    controls = [p for p in points if p["design"] == "control" and p["factors"] == [1] * len(identities)]
    if len(controls) != 1:
        raise ValueError("require exactly one 1x control")
    control = controls[0]
    for point in points:
        if point["status"] == control["status"] == "ok":
            point.update(paired=True, stage_path_equal_to_control=point["stage_path"] == control["stage_path"],
                         plan_change_from_control=plan_difference(control["plan"], point["plan"]),
                         delta_memo_expressions=point["memo_expressions"] - control["memo_expressions"],
                         rule_deltas={r: {k: point["rules"].get(r, {}).get(k, 0) - control["rules"].get(r, {}).get(k, 0)
                            for k in set(point["rules"].get(r, {})) | set(control["rules"].get(r, {}))}
                            for r in set(point["rules"]) | set(control["rules"])})
            if point.get("physical_counts") is not None and control.get("physical_counts") is not None:
                point["physical_deltas"] = {k: point["physical_counts"].get(k, 0) - control["physical_counts"].get(k, 0)
                                            for k in set(point["physical_counts"]) | set(control["physical_counts"])}
    for point in points:
        point.pop("stage_path", None)
        point.pop("plan", None)
        if point["status"] != "ok":
            for key in ("optimizer_cost", "memo_expressions", "binding_attempts", "rule_applications"):
                point[key] = None
    return points


def render_sweep_response(points: list[dict], targets: list[dict], output: Path, font: Path) -> None:
    """One-coordinate curves only; joint points stay in the audit, not on a fictitious 1D axis."""
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    metrics = [("binding_attempts", "逻辑规则尝试数"), ("memo_expressions", "搜索表达式数（含物理与标量）"),
               ("optimizer_cost", "估算计划成本（非耗时）"), ("cost:costed", "完成计价次数"),
               ("search:prune:pruned", "下界剪枝次数"), ("lifecycle:best_updated", "最优候选更新次数")]
    for dimension in range(len(targets)):
        selected = sorted((p for p in points if p["design"] in {"control", "one_at_a_time"}
                           and all(f == 1 for i, f in enumerate(p["factors"]) if i != dimension)),
                          key=lambda p: p["factors"][dimension])
        for axis, (metric, title) in zip(axes.flat, metrics):
            values = [((p.get("physical_counts") or {}).get(metric, 0)
                       if ":" in metric and p.get("physical_counts") is not None else p.get(metric))
                      if p["paired"] else None for p in selected]
            axis.plot([p["factors"][dimension] for p in selected],
                      [v if v is not None else math.nan for v in values], "o-", label=f"输入 {dimension + 1}")
            axis.set(title=title, xlabel="本输入基数倍率；其他输入保持一倍", xscale="log")
            axis.grid(alpha=0.2)
    axes[0, 0].legend()
    ticks = sorted({f for p in points if p["design"] in {"control", "one_at_a_time"} for f in p["factors"]})
    from matplotlib.ticker import NullLocator
    for axis in axes.flat:
        axis.set_xticks(ticks, [f"{t:g}" for t in ticks])
        axis.xaxis.set_minor_locator(NullLocator())
    invalid = sum(not p["paired"] for p in points)
    joint = sum(p["design"] not in {"control", "one_at_a_time"} for p in points)
    fig.suptitle(f"局部基数响应｜{len(points)} 个设计点，{invalid} 个不可配对，{joint} 个联合点仅保留审计\n"
                 "连线仅连接观测点；不是拟合、规则因果收益或查询无关结论")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)


def cohort_sweep_evidence(suite: dict, results: Path, contribution_rule: str | None = None) -> dict:
    cohort_path = Path(suite["cohort"])
    if f"{zlib.crc32(cohort_path.read_bytes()):08x}" != suite["cohort_crc32"]:
        raise ValueError("frozen cohort has changed")
    cohort = json.loads(cohort_path.read_text())
    selected = {q["query"]: q for q in cohort["queries"] if q["selected"]}
    if any(q.get("split") != "discovery" for q in selected.values()):
        raise ValueError("sweep analysis must not consume holdout or anchor traces")
    if len(suite["queries"]) != len(selected) or {q["query"] for q in suite["queries"]} != set(selected):
        raise ValueError("sweep suite must preserve all sampled queries")
    weights = family_design_weights(cohort)
    runs = []
    policy_snapshots = None
    for query in suite["queries"]:
        path = results / query["query"].split("/")[1] / query["query"] / "comparison.json"
        manifest_path = Path(query["manifest"])
        manifest = json.loads(manifest_path.read_text())
        original = selected[query["query"]]
        size = query["eligible_inputs"]
        if (query["family"] != original["family"] or query["query_crc32"] != original["query_crc32"]
                or manifest["targets"] != [query["target"]] or manifest["baseline_kind"] != "frozen_native_estimate"
                or type(size) is not int or size < 1 or query["input_probability"] != 1 / size):
            raise ValueError("sweep target, identity or input sampling probability mismatch")
        comparison = json.loads(path.read_text()) if path.is_file() else {"stats_experiments": [], "modes": {"replacement": {}}}
        if path.is_file() and comparison.get("query_crc32") != selected[query["query"]]["query_crc32"]:
            raise ValueError("sweep SQL identity mismatch")
        if contribution_rule:
            if path.is_file():
                profile = comparison.get("rule_profile") or {}
                if profile.get("rule_hash") != contribution_rule:
                    raise ValueError("contribution target rule mismatch")
                snapshots = profile["policy_snapshots"]
                if policy_snapshots is not None and snapshots != policy_snapshots:
                    raise ValueError("cannot pool different background/target policy snapshots")
                policy_snapshots = snapshots
                points = cbo_contribution(manifest, comparison, manifest_path.parent, path.parent)
            else:
                points = [{"factors": p["factors"], "statuses": {"off": "missing", "cbo": "missing"},
                           "delta": None, "pairs": [], "target_attempts": None, "target_ready": None,
                           "target_root_inserted": None, "off_fallback_cbo_ok": False, "exclusions": ["missing"]}
                          for p in manifest["points"]]
        else:
            points = sweep_candidate_evidence(manifest, comparison, manifest_path.parent, path.parent)
        runs.extend({**p, "query": query["query"], "family_weight": weights[query["query"]],
                     "target_operator": query["target"]["operator"]} for p in points)
    return {"scope": "cbo_minus_off_contribution" if contribution_rule else "paired_one_coordinate_rule_response_default_cbo",
            "runs": runs, "rule_hash": contribution_rule, "policy_snapshots": policy_snapshots,
            "measure": "equal_family_mean_of_uniform_observed_input_response",
            "stage_path_fields": ["rule_hash", "status", "failed_constraint"],
            "stage_path_does_not_compare": ["binding_identity", "full_expression_tree", "native_physical_xforms"],
            "not_guaranteed": ["joint_statistics_coverage", "actual_cardinality", "generalization", "runtime_speedup"],
            "target_design": suite["target_design"]}


def render_cohort_sweep(report: dict, output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    runs = report["runs"]
    factors = sorted({r["factors"][0] for r in runs})
    paired = [r for r in runs if r["paired"]]
    rules = sorted({rule for r in paired for rule in r["rule_deltas"]},
                   key=lambda rule: (-sum(abs(r["rule_deltas"].get(rule, {}).get("attempts", 0)) for r in paired), rule))[:6]
    for rule in rules:
        values = []
        for factor in factors:
            subset = [r for r in paired if r["factors"] == [factor]]
            values.append(sum(r["family_weight"] * r["rule_deltas"].get(rule, {}).get("attempts", 0) for r in subset)
                          / sum(r["family_weight"] for r in subset) if subset else math.nan)
        axes[0, 0].plot(factors, values, "o--", label=f"规则 {rule[:8]}")
    axes[0, 0].set(title="规则尝试数的配对响应（变化最多六条）", ylabel="相对一倍控制点的设计加权差值")
    if rules:
        axes[0, 0].legend(fontsize=8)
    # SQL identities remain provenance, never a learned feature. These are
    # individual experimental observations, not fitted per-query portraits.
    axes[0, 1].scatter([r["factors"][0] for r in paired], [r["delta_memo_expressions"] for r in paired], alpha=0.5)
    axes[0, 1].set(title="搜索空间响应分布（散点可能重叠）", ylabel="相对一倍控制点的表达式数量差")
    for field, label, marker in (("attempts_before_first_injection", "注入日志前", "o"),
                                 ("attempts_after_first_injection", "注入日志后", "x")):
        subset = [r for r in paired if r["order"][field] is not None]
        axes[1, 0].scatter([r["factors"][0] for r in subset], [r["order"][field] for r in subset],
                           label=label, marker=marker, alpha=0.6)
    axes[1, 0].set(title="逐条日志揭示注入与规则尝试的先后", ylabel="每次运行的尝试次数")
    axes[1, 0].legend()
    def plan_class(run):
        return run["plan_change_from_control"] if run["paired"] else run["status"] if run["status"] != "ok" else "baseline_not_comparable"
    bottom = [0] * len(factors)
    for kind in sorted({plan_class(r) for r in runs}):
        counts = [sum(plan_class(r) == kind and r["factors"] == [f] for r in runs) for f in factors]
        axes[1, 1].bar(factors, counts, bottom=bottom, width=[f * 0.25 for f in factors],
                       label=PLAN_LABELS.get(kind, "其他不可比状态"))
        bottom = [b + c for b, c in zip(bottom, counts)]
    axes[1, 1].set(title="最终计划相对一倍控制点的变化（保留失败）", ylabel="抽中样本的运行数，不是总体占比")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("随机选中输入相对冻结原生估计的倍率")
        axis.grid(alpha=0.2)
    fig.suptitle("查询无关规则画像：固定样本、随机输入、单坐标基数扰动\n"
                 "虚线仅连接实测点；失败不填零；一倍控制点内配对；不覆盖联合统计分布\n"
                 "日志先后不是统计已被匹配器使用的证明；无泛化或性能加速结论")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)


def target_root_costs(run: dict, target: list[dict], include_ancestors: bool = False) -> dict:
    """Exact new roots; optional recorded ancestry, never group overlap or co-occurrence."""
    roots = {(r["memo_outcome"]["group"], r["memo_outcome"]["group_expression"])
             for r in target if (r.get("memo_outcome") or {}).get("status") == "memo_inserted"}
    if include_ancestors and any("origin_chain" not in e for e in run["cost_events"]):
        raise ValueError("missing_origin_chain")
    if include_ancestors:
        costs = [e for e in run["cost_events"] if any(
            (n["group"], n["group_expression"]) in roots for n in e["origin_chain"])]
    else:
        costs = [e for e in run["cost_events"] if (e.get("origin_group"), e.get("origin_expression")) in roots]
    sequences = {e["sequence"] for e in costs}
    lifecycle = [e for e in run["cost_lifecycle_events"] if e["candidate_sequence"] in sequences]
    comparisons = []
    for event in costs:
        if event["status"] != "costed":
            continue
        cost, incumbent = event.get("cost"), event.get("context_best_cost_at_event")
        known = all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                    for value in (cost, incumbent))
        comparisons.append({**{key: event.get(key) for key in (
            "sequence", "group", "optimization_context", "operator", "rows", "cost",
            "required_columns", "required_order_columns", "required_order_matching",
            "required_distribution_type", "context_best_cost_at_event")},
            "cost_gap": cost - incumbent if known else None})
    return {"inserted_roots": [list(root) for root in sorted(roots)],
            "cost_sequences": sorted(sequences),
            "comparisons": comparisons,
            "cost_status_counts": dict(Counter(e["status"] for e in costs)),
            "lifecycle_status_counts": dict(Counter(e["status"] for e in lifecycle)),
            "scope": "same_run_inserted_root_recorded_ancestry_nonexclusive_not_child_dependency_or_causal_benefit"
                if include_ancestors else "same_run_inserted_root_direct_physical_origin_not_all_descendants_or_causal_benefit"}


def target_generated_costs(run: dict, target: list[dict]) -> dict:
    """Compact physical observations whose provenance directly consumes an inserted target."""
    inserted = {r["sequence"] for r in target
                if (r.get("memo_outcome") or {}).get("status") == "memo_inserted"}
    by_sequence = {event["sequence"]: event for event in run.get("cost_events", [])}
    rows, seen = [], set()
    for event in run.get("cost_events", []):
        if event.get("status") != "costed":
            continue
        paths = sorted({origin.get("target_path") for origin in event.get("dsl_origin_instances", [])
                        if origin.get("candidate_sequence") in inserted
                        and origin.get("relation") == "memo_consumes"
                        and origin.get("target_path") is not None})
        if not paths:
            continue
        children = [{"sequence": child.get("cost_candidate_sequence"),
                     "operator": by_sequence.get(child.get("cost_candidate_sequence"), {}).get("operator"),
                     "rows": by_sequence.get(child.get("cost_candidate_sequence"), {}).get("rows")}
                    for child in event.get("child_contexts", [])]
        key = (tuple(paths), event.get("operator"), event.get("rows"),
               tuple((child["operator"], child["rows"]) for child in children))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"sequence": event["sequence"], "target_paths": paths,
                     "operator": event.get("operator"), "rows": event.get("rows"),
                     "cost": event.get("cost"), "children": children})
    source_roots, source_children = [], []
    stats = {event.get("group"): event for event in run.get("stats_events", [])
             if event.get("site") == "memo_group"}
    for candidate in target:
        if candidate.get("sequence") not in inserted:
            continue
        provenance = (candidate.get("pre_evaluation_state") or {}).get("provenance") or {}
        root = provenance.get("root_memo") or {}
        event = stats.get(root.get("group"), {})
        source_roots.append({"candidate_sequence": candidate["sequence"], "group": root.get("group"),
                             "operator": event.get("operator"), "rows": event.get("native_rows")})
        for child in provenance.get("child_memo", []):
            event = stats.get(child.get("group"), {})
            source_children.append({"candidate_sequence": candidate["sequence"],
                                    "position": child.get("position"), "group": child.get("group"),
                                    "operator": event.get("operator"), "rows": event.get("native_rows")})
    return {"inserted_candidate_sequences": sorted(inserted), "source_roots": source_roots,
            "source_children": source_children,
            "physical_observations": rows,
            "scope": "same_run_direct_inserted_target_provenance_post_search_not_pre_evaluation_features"}


def post_search_evidence(run: dict) -> dict:
    """Observed physical descendants of inserted roots, never pre-decision features."""
    candidates = candidate_evidence(run)
    lifecycle = cost_lifecycle_evidence(run)
    problems = set(candidates['exclusions'] + lifecycle['exclusions'])
    if any('origin_chain' not in e for e in run.get('cost_events', [])):
        problems.add('missing_origin_chain')
    rules, union = {}, set()
    if not problems:
        inserted = defaultdict(list)
        for row in candidates['rows']:
            if (row.get('memo_outcome') or {}).get('status') == 'memo_inserted':
                inserted[row['rule_hash']].append(row)
        for rule, rows in sorted(inserted.items()):
            direct = target_root_costs(run, rows)
            descendants = target_root_costs(run, rows, include_ancestors=True)
            union.update(descendants['cost_sequences'])
            rules[rule] = {'direct': direct, 'recorded_descendants': descendants}
    return {'complete': not problems, 'exclusions': sorted(problems),
            'scope': 'post_search_recorded_origin_ancestry_nonexclusive_not_causal_benefit',
            'usable_as_pre_evaluation_features': False,
            'rules': rules if not problems else None,
            'cost_events': len(run.get('cost_events', [])),
            'attributed_distinct_cost_events': len(union) if not problems else None,
            'unattributed_cost_sequences': sorted(e['sequence'] for e in run.get('cost_events', [])
                                                 if e['sequence'] not in union) if not problems else None,
            'not_included': ['duplicate_producer_credit', 'unrecorded_rehash_ancestry',
                             'child_dependency_closure', 'causal_rule_benefit']}


def search_contribution(arms: dict, exclusions: list[str]) -> dict:
    """Compare audited stage counts, never match run-local candidate IDs across arms."""
    summaries, problems = {}, list(exclusions)
    for arm in ("off", "cbo"):
        costs = cost_evidence(arms[arm])
        lifecycle = cost_lifecycle_evidence(arms[arm])
        checks = search_check_evidence(arms[arm])
        errors = sorted(set(costs["exclusions"] + lifecycle["exclusions"] + checks["exclusions"]))
        problems.extend(f"{arm}:{error}" for error in errors)
        metrics = {f"cost:{status}": count for status, count in costs["status_counts"].items()}
        metrics.update({f"lifecycle:{status}": count for status, count in lifecycle["status_counts"].items()})
        metrics.update({f"{check}:{status}": count for check, counts in checks["status_counts"].items()
                        for status, count in counts.items()})
        metrics.update({field: lifecycle[field] for field in (
            "materialized_candidates", "ever_best_candidates", "selected_distinct_candidates")})
        summaries[arm] = {"complete": not errors, "exclusions": errors,
                          "metrics": metrics if not errors else None}
    delta = None
    if not problems:
        off, cbo = (summaries[arm]["metrics"] for arm in ("off", "cbo"))
        delta = {key: cbo.get(key, 0) - off.get(key, 0) for key in sorted(off.keys() | cbo.keys())}
    return {"complete": not problems, "exclusions": sorted(set(problems)), "arms": summaries, "delta": delta,
            "scope": "fixed_background_cbo_minus_off_stage_counts_not_cross_run_candidate_identity"}


def observed_rule_edges(run: dict, artifact: Path | None) -> list[dict] | None:
    """Retain emitted provenance, not co-occurrence-derived dependency edges."""
    if "rule_edges" in run:
        return run["rule_edges"]
    if artifact is None or not artifact.is_file():
        return None  # Missing evidence differs from an observed empty edge stream.
    return [record for record in trace_records(artifact.read_text()) if record.get("kind") == "rule_edge"]


def rule_edge_summary(run: dict, artifact: Path | None) -> dict:
    source = run.get("rule_edges_source")
    if isinstance(source, dict):
        return {"edge_count": source.get("count"),
                "count_complete": source.get("count_complete"),
                "source": str(artifact) if artifact is not None else source.get("artifact")}
    edges = run.get("rule_edges")
    return {"edge_count": len(edges) if isinstance(edges, list) else None,
            "count_complete": None,
            "source": str(artifact) if artifact is not None else None}


def cbo_contribution(manifest: dict, comparison: dict, manifest_dir: Path,
                     artifacts: Path | None = None) -> list[dict]:
    """CBO minus OFF, separating optimizer feasibility from comparable performance."""
    profile = comparison["rule_profile"]
    if len(profile.get("intervention_rule_hashes", [])) > 1:
        raise ValueError("single-rule contribution cannot attribute a joint rule bundle")
    if profile.get("arms") != ["off", "cbo"] or set(profile["policy_snapshots"]) != {"off", "cbo"}:
        raise ValueError("contribution report requires explicit OFF/CBO-only policies")
    diagnostics = profile_points(manifest, comparison, manifest_dir)
    samples = timing_points(manifest, comparison, manifest_dir) if "timing" in profile else []
    scenarios = {s["stats_experiment"]: s for s in profile["scenarios"]}
    rows = []
    for index, point in enumerate(manifest["points"]):
        scenario = scenarios[str((manifest_dir / point["file"]).resolve())]
        scenario_index = profile["scenarios"].index(scenario)
        arms = scenario["arms"]
        statuses = {arm: diagnostics[arm][index]["status"] for arm in arms}
        reasons = [f"{arm}:{status}" for arm, status in statuses.items() if status != "ok"]
        if comparison.get("artifact_provenance", {}).get("endpoints_equal") is False:
            reasons.append("experiment_artifacts_changed")
        reasons.extend(profile_comparability(arms["off"], arms["cbo"], True)["exclusions"])
        audited = {}
        for arm in arms:
            if statuses[arm] != "ok":
                continue
            try:
                audit = candidate_evidence(arms[arm])
                if any(e.get("placement") != "cbo" for e in arms[arm].get("candidate_events", []) if e["kind"] == "rule_candidate"):
                    raise ValueError("non-CBO candidate in contribution experiment")
                if not audit["complete"]:
                    raise ValueError(",".join(audit["exclusions"]))
                audited[arm] = audit
            except ValueError as error:
                reasons.append(f"{arm}:invalid_candidate_trace:{error}")
        target = [r for r in audited.get("cbo", {}).get("rows", []) if r["rule_hash"] == profile["rule_hash"]]
        if any(r["rule_hash"] == profile["rule_hash"] for r in audited.get("off", {}).get("rows", [])):
            reasons.append("disabled_rule_has_candidates")
        plan_change = None
        if not reasons and artifacts is not None:
            try:
                plans = [timing_plan(plan_tree((artifacts / f"profile-{scenario_index}.{arm}.plan.json").read_text()))
                         for arm in ("off", "cbo")]
                if any(plan is None for plan in plans):
                    raise ValueError("invalid plan JSON")
                plan_change = plan_difference(*plans)
            except (OSError, ValueError) as error:
                reasons.append(f"plan_artifact:{error}")
        pairs = [dict(s) for s in samples if s["file"] == point["file"] and s["arm"] == "cbo"]
        for pair in pairs:
            pair["exclusions"] = sorted(set([*pair["exclusions"], *reasons]))
            if pair["exclusions"]:
                for field in ("planning_ms", "execution_ms", "delta_planning_ms", "delta_execution_ms"):
                    pair[field] = None
        delta = None
        rule_deltas = None
        if not reasons:
            counts = {arm: rule_stage_counts(audited[arm]["rows"]) for arm in arms}
            rule_deltas = {r: {k: counts["cbo"].get(r, {}).get(k, 0) - counts["off"].get(r, {}).get(k, 0)
                for k in set(counts["cbo"].get(r, {})) | set(counts["off"].get(r, {}))}
                for r in set(counts["cbo"]) | set(counts["off"])}
            delta = {"attempts": audited["cbo"]["attempts"] - audited["off"]["attempts"],
                     "memo_expressions": diagnostics["cbo"][index]["memo_expressions"] - diagnostics["off"][index]["memo_expressions"],
                     "optimizer_cost": diagnostics["cbo"][index]["optimizer_cost"] - diagnostics["off"][index]["optimizer_cost"]}
        search = search_contribution(arms, reasons)
        states = Counter((binding_shape_features(r)["bound_predicate_structure"], r["status"],
                          r.get("failed_constraint")) for r in target)
        rows.append({"experiment": point.get("experiment"), "factors": point["factors"],
                     "requested_rows": point.get("requested_rows"),
                     "stats_targets": {arm: ((run.get("experiment_outcomes") or [{}])[0].get("stats_targets"))
                                       for arm, run in arms.items()},
                     "statuses": statuses, "exclusions": sorted(set(reasons)),
                     "dependencies": {"scope": "successful_cbo_source_root_provenance_not_complete_binding_dependencies",
                         "arms": {arm: {"complete_candidate_trace": arm in audited,
                             **rule_edge_summary(run, artifacts / f"profile-{scenario_index}.{arm}.trace"
                                                if artifacts is not None else None)}
                                  for arm, run in arms.items()}},
                     "target_states": [{"predicate_structure": shape, "status": status,
                                        "failed_constraint": constraint, "attempts": count}
                                       for (shape, status, constraint), count in states.items()]
                         if "cbo" in audited else None,
                     "delta": delta, "rule_deltas": rule_deltas, "pairs": pairs,
                     "other_rule_attempts_delta": delta["attempts"] - len(target) if delta is not None else None,
                     "result_rows": {arm: run.get("rows_count") for arm, run in arms.items()},
                     "native_origins": {arm: run.get("native_memo_origins") for arm, run in arms.items()},
                     "replacement_audit": {key: comparison.get(key) for key in (
                         "join_enumeration_replaced", "forbidden_native_origins")},
                     "target_attempts": len(target) if "cbo" in audited else None,
                     "target_ready": sum(r["status"] == "ready_cbo" for r in target) if "cbo" in audited else None,
                     "target_root_inserted": sum((r["memo_outcome"] or {}).get("status") == "memo_inserted" for r in target)
                         if "cbo" in audited else None,
                     "off_fallback_cbo_ok": statuses["off"] == "fallback" and statuses["cbo"] == "ok",
                     "plan_change": plan_change, "search": search,
                     "target_root_costs": target_root_costs(arms["cbo"], target)
                         if "cbo" in audited and search["arms"]["cbo"]["complete"] else None,
                     "target_generated_costs": target_generated_costs(arms["cbo"], target)
                         if "cbo" in audited and search["arms"]["cbo"]["complete"] else None})
    return rows


def factor_axis(report: dict) -> tuple[list[tuple], dict[tuple, float], list[str], bool]:
    """Use a log axis for one positive factor and labeled cells for factorial designs."""
    factors = sorted({tuple(row["factors"]) for row in report["runs"]})
    dimensions = {len(factor) for factor in factors}
    if len(dimensions) != 1:
        raise ValueError("inconsistent factor dimensions")
    logarithmic = bool(factors and dimensions == {1} and all(factor[0] > 0 for factor in factors))
    positions = {factor: factor[0] if logarithmic else float(index)
                 for index, factor in enumerate(factors)}
    names = report.get("factor_labels") or report.get("factor_coordinates") or [
        f"因子{i + 1}" for i in range(next(iter(dimensions), 0))]
    labels = ["／".join(f"{names[i] if i < len(names) else f'因子{i + 1}'}={value:g}"
                       for i, value in enumerate(factor)) for factor in factors]
    return factors, positions, labels, logarithmic


def render_cbo_contribution(report: dict, output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), layout="constrained")
    rows = report["runs"]
    factors, positions, labels, logarithmic = factor_axis(report)
    for axis, field, title in zip(axes[0], ("delta_planning_ms", "delta_execution_ms"), ("规划时间的条件贡献", "执行时间的条件贡献")):
        for factor in factors:
            selected = [r for r in rows if tuple(r["factors"]) == factor]
            values = [(r.get("family_weight", 1.0), [p[field] for p in r["pairs"] if p[field] is not None])
                      for r in selected]
            x = positions[factor]
            axis.scatter([x for _, vs in values for _ in vs], [v for _, vs in values for v in vs], alpha=0.35)
            means = [(w, sum(vs) / len(vs)) for w, vs in values if vs]
            if means:
                axis.scatter(x, sum(w * v for w, v in means) / sum(w for w, _ in means), marker="_", s=180, color="#ee7733")
        axis.set(title=title, ylabel="开启 − 关闭（毫秒）；负值为降低")
        axis.axhline(0, color="gray", linewidth=0.7)
    valid = sorted((r for r in rows if r["delta"] is not None), key=lambda r: r["factors"][0])
    xs = [positions[tuple(r["factors"])] for r in valid]
    axes[1, 0].scatter(xs, [r["delta"]["memo_expressions"] for r in valid],
                       color="#4477aa", label="表达式净增")
    cost_axis = axes[1, 0].twinx()
    cost_axis.plot(xs, [-r["delta"]["optimizer_cost"] for r in valid],
                   "o-", color="#cc3311", label="估算成本降低")
    axes[1, 0].set(title="搜索空间与终止计划成本收益", ylabel="开启 − 关闭的表达式数量")
    cost_axis.set_ylabel("关闭 − 开启的估算计划成本；正值为改善", color="#cc3311")
    bottom = [0] * len(factors)
    for kind, label in (("ok", "诊断可比"), ("fallback", "关闭回退，开启正常"), ("other", "其他不可比或未运行")):
        def category(r):
            return "ok" if r["delta"] is not None else "fallback" if r["off_fallback_cbo_ok"] else "other"
        counts = [sum(category(r) == kind and tuple(r["factors"]) == factor for r in rows) for factor in factors]
        axes[1, 1].bar([positions[factor] for factor in factors], counts, bottom=bottom,
                       width=[factor[0] * 0.25 for factor in factors] if logarithmic else 0.7, label=label)
        bottom = [b + c for b, c in zip(bottom, counts)]
    axes[1, 1].set(title="可执行性必须先于性能比较", ylabel="固定样本运行数；不是总体比例")
    axes[1, 1].legend()
    for field, label, marker in (("target_attempts", "目标自身尝试", "o"),
                                 ("target_ready", "目标生成候选", "x"),
                                 ("target_root_inserted", "目标根实际插入", "+")):
        available = [r for r in rows if r[field] is not None]
        axes[2, 0].scatter([positions[tuple(r["factors"])] for r in available], [r[field] for r in available],
                           label=label, marker=marker, alpha=0.5)
    axes[2, 0].set(title="目标规则从尝试到根插入的漏斗", ylabel="开启臂次数；散点可能重叠")
    axes[2, 0].legend()
    axes[2, 1].scatter(xs, [r["other_rule_attempts_delta"] for r in valid], alpha=0.5)
    axes[2, 1].axhline(0, color="gray", linewidth=0.7)
    axes[2, 1].set(title="其他规则尝试的净变化（不是依赖边证明）", ylabel="开启 − 关闭；扣除目标自身尝试")
    for axis in axes.flat:
        if logarithmic:
            axis.set_xscale("log", base=2)
        else:
            axis.set_xticks([positions[factor] for factor in factors], labels, rotation=15, ha="right")
        axis.set_xlabel("冻结原生估计的注入倍率" if logarithmic else "预注册的联合注入因子组合")
        axis.grid(alpha=0.2)
    fig.suptitle(f"规则 {report['rule_hash']} 的代价搜索条件贡献｜未执行预改写实验臂\n"
                 "同统计、同随机区组配对；橙线仅对可比样本的组内均值设计加权，不代表完整总体\n"
                 "回退不算加速，散点不是置信区间；固定背景消融不等于原生替代验收，未完成泛化验证")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_search_contribution(report: dict, output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    metrics = (("cost:costed", "完成成本计算"), ("cost:duplicate_context", "成本入口重复上下文"),
               ("prune:pruned", "实际下界剪枝检查"), ("properties:accepted", "物理属性检查通过"),
               ("lifecycle:best_updated", "更新当前最优"), ("selected_distinct_candidates", "最终采用的去重成本候选"))
    valid = [r for r in report["runs"] if (r.get("search") or {}).get("complete")]
    factors, positions, labels, logarithmic = factor_axis(report)
    for axis, (field, label) in zip(axes.flat, metrics):
        for query in sorted({r.get("query", "单查询") for r in valid}):
            rows = sorted((r for r in valid if r.get("query", "单查询") == query),
                          key=lambda r: positions[tuple(r["factors"])])
            axis.plot([positions[tuple(r["factors"])] for r in rows],
                      [r["search"]["delta"].get(field, 0) for r in rows],
                      "o--", label=query, alpha=0.75)
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.set(title=label, xlabel="该样本冻结基数的注入倍率" if logarithmic else "预注册的联合注入因子组合",
                 ylabel="开启 − 关闭（次数）")
        if logarithmic:
            axis.set_xscale("log")
        axis.set_xticks([positions[factor] for factor in factors],
                        [f"{factor[0]:g}" for factor in factors] if logarithmic else labels,
                        rotation=0 if logarithmic else 15, ha="center" if logarithmic else "right")
        axis.minorticks_off()
        axis.grid(alpha=0.2)
        if valid:
            axis.legend(fontsize=8)
    fig.suptitle(f"规则 {report['rule_hash']} 的物理搜索条件贡献｜完整对照点 {len(valid)}\n"
                 "查询编号仅标识实验输入；虚线连接同一输入的干预点，不代表总体拟合\n"
                 "计数净差不是候选身份对应，也不是规则依赖边；最终未选中不等于无用")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def cohort_candidate_evidence(cohort: dict, results: Path, experiment: int = 1) -> dict:
    """Conditional rule profiles with family design weights, not attempt weights."""
    if experiment < 1:
        raise ValueError("experiment index must be positive")
    selected = [q for q in cohort["queries"] if q["selected"]]
    if any(q.get("split") == "holdout" for q in selected):
        raise ValueError("discovery analysis must not consume holdout traces")
    if len({(q.get("split"), q.get("family")) for q in selected}) != len(selected):
        raise ValueError("one selected member per family is required")
    weights = family_design_weights(cohort)
    runs, cells = [], []
    paths = set()
    for query in selected:
        path = results / query["query"] / "comparison.json"
        item = {k: query[k] for k in ("query", "family", "split")}
        item.update(family_weight=weights[query["query"]], stratum=query.get("stratum"))
        item.update(status="missing", attempts=None, exclusions=[])
        runs.append(item)
        if not path.is_file():
            continue
        comparison = json.loads(path.read_text())
        # Observation transparency and native replacement correctness are
        # different gates. Retain the latter even when the former passes.
        item.update(native_rows_equal=comparison.get("rows_equal"),
                    native_rows_bag_equal=comparison.get("rows_bag_equal"),
                    forbidden_native_origins=comparison.get("forbidden_native_origins"))
        if comparison.get("query_crc32") != query["query_crc32"]:
            item["status"] = "query_identity_mismatch"
            continue
        experiments = comparison.get("stats_experiments", [])
        if len(experiments) < experiment:
            item["status"] = "missing_experiment"
            continue
        point = experiments[experiment - 1]
        paths.add(point["path"])
        run = point["modes"]["replacement"]
        item["result_rows"] = run.get("rows_count")
        item["status"] = experiment_run_status(run, True)
        if item["status"] != "ok":
            item["exclusions"] = [item["status"]]
            continue  # A censored prefix is not a sample of all attempts.
        try:
            audit = candidate_evidence(run)
        except ValueError as error:
            item.update(status="invalid_trace", exclusions=[str(error)])
            continue
        item["exclusions"] = audit["exclusions"]
        if item["status"] == "ok" and not audit["complete"]:
            item["status"] = "incomplete_trace"
        if item["status"] != "ok":
            continue
        if run["experiment_outcomes"][0]["stats_targets"]:
            item["status"] = "intervention_not_observation"
            continue
        if any(e.get("placement") != "cbo" for e in run["candidate_events"] if e["kind"] == "rule_candidate"):
            item["status"] = "non_cbo_candidates"
            continue
        # Successful SQL alone does not establish instrumentation transparency.
        baseline = comparison["modes"]["replacement"]
        check = point["comparisons"]["replacement"]
        if (experiment_run_status(baseline, False) != "ok" or check.get("rows_equal") is not True
                or check.get("plan_comparison") != "identical"):
            item["status"] = "baseline_not_comparable"
            continue
        item["attempts"] = audit["attempts"]
        groups = defaultdict(list)
        for row in audit["rows"]:
            context = row.get("input_context") or {}
            shape = context.get("source_shape") or {}
            nodes = shape.get("nodes")
            band = (1 << (nodes.bit_length() - 1)) if (shape.get("complete") is True
                and shape.get("pattern_nodes") == 0 and type(nodes) is int and nodes > 0) else None
            groups[row["rule_hash"], context.get("root", {}).get("operator"),
                   row["shape_features"]["source_subqueries"], band].append(row)
        for (rule, operator, subqueries, band), rows in groups.items():
            cells.append({**{k: item[k] for k in ("query", "family", "split", "family_weight", "stratum")}, "rule_hash": rule,
                "source_operator": operator, "source_subqueries": subqueries, "source_nodes_band": band,
                "attempts": len(rows), "ready": sum(r["status"] == "ready_cbo" for r in rows),
                "statuses": dict(Counter(r["status"] for r in rows)),
                "memo_outcomes": dict(Counter(r["memo_outcome"]["status"] for r in rows if r.get("memo_outcome"))),
                "evaluation_us": sum(sum(r[t] for t in TIMES) for r in rows),
                "direct_insertions": sum(r["direct_insertions"] or 0 for r in rows)})
    if len(paths) > 1:
        raise ValueError("cannot pool different experiment paths in one batch")
    # A family can occupy several structural cells; each cell estimates a
    # conditional mean over families where that rule was actually dispatched.
    grouped = defaultdict(list)
    for cell in cells:
        grouped[tuple(cell[k] for k in ("split", "rule_hash", "source_operator", "source_subqueries", "source_nodes_band"))].append(cell)
    groups = [{**dict(zip(("split", "rule_hash", "source_operator", "source_subqueries", "source_nodes_band"), key)),
        "families": len(rows), "attempts": sum(r["attempts"] for r in rows),
        "family_weight_sum": sum(r["family_weight"] for r in rows),
        "mean_family_status_rates": {
            status: sum(r["family_weight"] * r["statuses"].get(status, 0) / r["attempts"] for r in rows)
                    / sum(r["family_weight"] for r in rows)
            for status in sorted({s for r in rows for s in r["statuses"]})},
        "mean_family_direct_insertions_per_attempt": sum(r["family_weight"] * r["direct_insertions"] / r["attempts"] for r in rows)
                    / sum(r["family_weight"] for r in rows),
        "mean_family_root_inserted_per_attempt": sum(r["family_weight"] * r["memo_outcomes"].get("memo_inserted", 0) / r["attempts"] for r in rows)
                    / sum(r["family_weight"] for r in rows),
        "mean_family_ready_rate": sum(r["family_weight"] * r["ready"] / r["attempts"] for r in rows) / sum(r["family_weight"] for r in rows),
        "mean_family_evaluation_us": sum(r["family_weight"] * r["evaluation_us"] / r["attempts"] for r in rows) / sum(r["family_weight"] for r in rows)}
        for key, rows in sorted(grouped.items(), key=lambda pair: str(pair[0]))]
    return {"scope": "cbo_dispatched_attempts_single_batch_observation_only", "runs": runs,
            "selected_families": len(selected), "complete_families": sum(r["status"] == "ok" for r in runs),
            "measure": "family_design_weighted_ratio_conditional_on_complete_dispatch_and_input_cell",
            "sampling_unit": cohort.get("sampling_unit"), "strata": cohort.get("strata", []),
            "estimand": cohort.get("estimand"),
            "weighted_status_share": {status: sum(r["family_weight"] for r in runs if r["status"] == status) / sum(r["family_weight"] for r in runs)
                                      for status in sorted({r["status"] for r in runs})},
            "size_bands": "dyadic_display_bins_not_a_fitted_function",
            "generalization_validated": False, "timing_source": "traced_diagnostics",
            "cells": cells, "groups": groups}


def render_candidate_cohort(report: dict, output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), layout="constrained")
    names = {"ok": "观测完整可比", "missing": "未运行", "plan_timeout": "优化或执行超时",
             "rows_timeout": "结果超时", "fallback": "回退原生规划器", "incomplete_trace": "日志不完整",
             "baseline_not_comparable": "基线不可比", "plan_error": "计划阶段错误"}
    counts = Counter(r["status"] for r in report["runs"])
    axes[0].bar([names.get(k, f"其他不可比状态{i + 1}") for i, k in enumerate(counts)],
                list(counts.values()), color="#4477aa")
    axes[0].set(title="观测开关对照（不是原生替代验收）", ylabel="结构组数量")
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].tick_params(axis="x", labelrotation=20)
    discovery = [c for c in report["cells"] if c["split"] == "discovery"]
    by_rule = defaultdict(list)
    for cell in discovery:
        by_rule[cell["rule_hash"]].append(cell)
    top = sorted(by_rule, key=lambda rule: (-sum(c["attempts"] for c in by_rule[rule]), rule))[:6]
    positive = sorted((rule for rule in by_rule if any(c["ready"] for c in by_rule[rule])),
                      key=lambda rule: (-sum(c["ready"] for c in by_rule[rule]), rule))[:6]
    # Display-only diagnostic selection, never used to select workload samples
    # or to estimate a distribution over all registered rules.
    top += [rule for rule in positive if rule not in top]
    for rule in top:
        by_family = defaultdict(lambda: [0, 0, 0])
        for cell in by_rule[rule]:
            by_family[cell["family"]][0] += cell["attempts"]
            by_family[cell["family"]][1] += cell["ready"]
            by_family[cell["family"]][2] = cell["family_weight"]
        rate = sum(weight * ready / attempts for attempts, ready, weight in by_family.values()) / sum(weight for _, _, weight in by_family.values())
        label = f"规则 {rule[:8]}（{len(by_family)}组）"
        axes[1].barh(label, rate * 100)
        axes[1].text(rate * 100 + 0.5, label, f"{rate * 100:.1f}%", va="center", fontsize=8)
        # Keep each root/subquery category separate; no cross-category line fit.
        points = [g for g in report["groups"] if g["split"] == "discovery" and g["rule_hash"] == rule
                  and g["source_nodes_band"] is not None]
        axes[2].scatter([g["source_nodes_band"] for g in points],
                        [g["mean_family_evaluation_us"] for g in points], label=f"规则 {rule[:8]}")
    axes[1].set(title="生成待入库候选的比例", xlabel="组内比例再按抽样设计加权（%）", xlim=(0, 110))
    axes[2].set(title="求值前来源树规模与失败／成功开销", xlabel="节点数分箱下界（上界为两倍，不含）",
                ylabel="设计加权的单次求值均值（微秒）")
    if top:
        axes[2].set_xscale("log", base=2)
        axes[2].legend(fontsize=7)
    mismatches = sum(r.get("native_rows_equal") is False for r in report["runs"])
    fig.suptitle("规则画像探索｜规则是画像对象，查询身份不作输入特征\n"
                 "诊断展示：尝试最多六条与候选生成最多六条的并集；不代表规则总体分布\n"
                 f"与原生结果逐行不同的样本：{mismatches}（单独分类，不能自动判错）；不是因果曲线或泛化验证")
    for axis in axes:
        axis.grid(axis="x", alpha=0.15)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)

    if report["strata"]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
        labels, sizes, sampled = [], [], []
        for s in report["strata"]:
            # The sampler records SQL syntax, not optimizer matchability.
            flags = s["stratum"].split(":")[1:]
            labels.append("／".join(("多查询块" if flags[0] == "blocks1" else "单查询块",
                                     "有分组" if flags[1] == "group1" else "无分组",
                                     "有窗口" if flags[2] == "window1" else "无窗口")))
            sizes.append(s["families"])
            sampled.append(s["sampled_families"])
        axes[0].barh(range(len(labels)), sizes, label="剩余探索总体", color="#4477aa")
        axes[0].barh(range(len(labels)), sampled, label="本批抽中", color="#ee7733")
        axes[0].set_yticks(range(len(labels)), labels)
        for i, (size, take) in enumerate(zip(sizes, sampled)):
            axes[0].text(size + 0.5, i, f"{take}/{size}", va="center", fontsize=9)
        axes[0].set(title="预先声明的语法结构分层", xlabel="结构组数；标注为抽中数／层大小", xlim=(0, max(sizes) * 1.2))
        axes[0].legend()
        statuses = list(report["weighted_status_share"])
        axes[1].bar([names.get(k, f"其他状态{i + 1}") for i, k in enumerate(statuses)],
                    [report["weighted_status_share"][k] * 100 for k in statuses], color="#228833")
        axes[1].set(title="保留失败分母的状态分布估计", ylabel="按层大小加权的总体占比估计（%）", ylim=(0, 110))
        fig.suptitle("结构分层随机抽样｜先冻结样本，再查看运行结果\n"
                     "规则格内采用组内均值再设计加权；非普查层仅一个样本时，不能估计层内方差\n"
                     "只代表剩余探索总体；未使用留出集，不表示已验证泛化")
        for suffix in (".png", ".svg"):
            fig.savefig(output.with_name(output.name + "-sampling").with_suffix(suffix), dpi=160)
        plt.close(fig)


def parameter_candidate_evidence(manifest: dict, results: Path, contribution_rule: str | None = None) -> dict:
    """Each parameter instance is retained; no template-level search equivalence assumption."""
    sampling_unit = manifest.get("sampling_unit")
    if sampling_unit not in ("declared_parameter_instances", "predeclared_application_case"):
        raise ValueError("require declared parameter instances or application cases")
    queries = manifest["queries" if sampling_unit == "declared_parameter_instances" else "cases"]
    if len({q["query"] for q in queries}) != len(queries):
        raise ValueError("duplicate parameter instance")
    rows = []
    for query in queries:
        row = {**query, "status": "missing", "exclusions": [], "attempts": None,
               "memo_expressions": None, "rules": None, "result_rows": None,
               "state_coverage": None, "rule_state_coverage": None}
        rows.append(row)
        path = results / query["query"] / "comparison.json"
        if not path.is_file():
            continue
        comparison = json.loads(path.read_text())
        if comparison.get("query_crc32") != query["query_crc32"]:
            row["status"] = "query_identity_mismatch"
            continue
        row.update(native_rows_equal=comparison.get("rows_equal"),
                   forbidden_native_origins=comparison.get("forbidden_native_origins"))
        experiments = comparison.get("stats_experiments", [])
        if len(experiments) != 1:
            row["status"] = "require_one_observation_experiment"
            continue
        experiment = experiments[0]
        run = experiment["modes"]["replacement"]
        row["result_rows"] = run.get("rows_count")
        if contribution_rule:
            if (comparison.get("rule_profile") or {}).get("rule_hash") != contribution_rule:
                raise ValueError("parameter contribution target rule mismatch")
            # Reuse the sweep contrast with zero intervention coordinates: these
            # are parameter instances, not invented cardinality factors.
            observation = {"targets": [], "points": [{"file": experiment["path"],
                           "factors": [], "requested_rows": []}]}
            row["contribution"] = cbo_contribution(observation, comparison, path.parent, path.parent)[0]
            for arm, dependency in row["contribution"]["dependencies"]["arms"].items():
                if "edges" in dependency:  # legacy reports and injected test evidence
                    edges = dependency.pop("edges")
                    dependency["edge_count"] = len(edges) if edges is not None else None
                dependency.setdefault("source", f"{path}:stats_experiments[0].modes.{arm}.rule_edges")
            row["target_states"] = row["contribution"].get("target_states")
        try:
            audit = candidate_evidence(run)
        except ValueError as error:
            row.update(status="invalid_trace", exclusions=[str(error)])
            continue
        row["exclusions"] = list(audit["exclusions"])
        outcomes = run.get("experiment_outcomes", [])
        if len(outcomes) != 1 or outcomes[0].get("stats_targets") != []:
            row["exclusions"].append("not_observation_only")
        if any(e.get("placement") != "cbo" for e in run.get("candidate_events", []) if e["kind"] == "rule_candidate"):
            row["exclusions"].append("non_cbo_candidates")
        baseline = comparison["modes"]["replacement"]
        check = experiment["comparisons"]["replacement"]
        if (experiment_run_status(baseline, False) != "ok" or check.get("rows_equal") is not True
                or check.get("plan_comparison") != "identical"):
            row["exclusions"].append("observation_not_transparent")
        memo = run.get("dsl_observability", {}).get("memo") or {}
        if "group_expressions" not in memo:
            row["exclusions"].append("missing_memo_summary")
        status = experiment_run_status(run, True)
        row["status"] = status if status != "ok" else "incomplete_or_not_comparable" if row["exclusions"] else "ok"
        if row["status"] == "ok":
            row.update(attempts=audit["attempts"], rules=rule_stage_counts(audit["rows"]),
                       memo_expressions=memo["group_expressions"], state_coverage=state_coverage(audit["rows"]))
            by_rule = defaultdict(list)
            for attempt in audit["rows"]:
                by_rule[attempt["rule_hash"]].append(attempt)
            row["rule_state_coverage"] = {rule: state_coverage(attempts) for rule, attempts in by_rule.items()}
    return {"scope": ("predeclared_application_holdout" if sampling_unit == "predeclared_application_case"
                       else "within_template_parameter_mechanism_observations"), "runs": rows,
            "rule_hash": contribution_rule,
            "sampling_unit": sampling_unit, "split": manifest.get("split"),
            "parameter_design": manifest["parameter_design"],
            "state_observation": {"capture": "before_evaluation", "scope": "source_before_match_view",
                "identity": "run_and_attempt_sequence_not_expression_fingerprint",
                "not_observed": ["column_distributions", "column_types_and_key_membership", "required_physical_properties",
                                 "incumbent_and_search_budget", "complete_memo_alternatives",
                                 "full_predicates_and_ordered_tree_topology"],
                "excluded_from_pre_features": ["post_evaluation_bindings", "later_derived_statistics",
                                               "rule_outcomes", "query_local_identifiers"]},
            "not_guaranteed": list(dict.fromkeys([*manifest.get("not_guaranteed", []),
                "population_representativeness", "complete_search_space_identity", "runtime_speedup"]))}


def render_parameters(report: dict, output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    templates = list(dict.fromkeys(r["template"] for r in report["runs"]))
    fig, axes = plt.subplots(len(templates), 2, figsize=(14, 4 * len(templates)), squeeze=False, layout="constrained")
    for index, template in enumerate(templates):
        rows = [r for r in report["runs"] if r["template"] == template]
        positions = list(range(len(rows)))
        bottom = [0] * len(rows)
        for stage, (_, label, color) in STAGES.items():
            counts = [sum(rule.get(stage, 0) for rule in (r["rules"] or {}).values())
                      if r["status"] == "ok" else math.nan for r in rows]
            axes[index, 0].bar(positions, counts, bottom=bottom, label=label, color=color)
            bottom = [a + b for a, b in zip(bottom, counts)]
        axes[index, 0].set(title=f"模板{index + 1}：逐规则阶段汇总", ylabel="尝试次数；失败留空")
        bars = axes[index, 1].bar(positions, [r["memo_expressions"] if r["memo_expressions"] is not None else math.nan for r in rows])
        axes[index, 1].bar_label(bars, labels=[str(r["memo_expressions"]) if r["memo_expressions"] is not None else ""
                                             for r in rows], padding=3)
        axes[index, 1].set(title="同模板实例的实际搜索规模", ylabel="最终 Memo 表达式数；不是完整空间等价判定")
        for axis in axes[index]:
            axis.set_xticks(positions, [f"实例{i + 1}" + ("\n不可比" if r["status"] != "ok" else "")
                                       for i, r in enumerate(rows)])
            axis.set_xlabel("参数与实例映射见同名结果文件；不按语法相似合并")
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("同一查询模板，不预设相同搜索空间\n有限参数场景的机制验证；没有基数注入；不是总体分布或性能收益曲线")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_parameter_contribution(report: dict, output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    templates = list(dict.fromkeys(r["template"] for r in report["runs"]))
    fig, axes = plt.subplots(len(templates), 4, figsize=(20, 4 * len(templates)), squeeze=False, layout="constrained")
    for index, template in enumerate(templates):
        rows = [r for r in report["runs"] if r["template"] == template]
        positions = list(range(len(rows)))
        for stage, (_, label, color) in STAGES.items():
            axes[index, 0].plot(positions, [sum(s["attempts"] for s in r["target_states"] if s["status"] == stage)
                                            if r.get("target_states") is not None else math.nan for r in rows], "o", label=label, color=color)
        axes[index, 0].set(title=f"模板{index + 1}：目标规则各阶段", ylabel="开启时尝试数；见结果文件的谓词分类")
        for field, label in (("memo_expressions", "表达式净增"), ("cost:costed", "成本计算净增")):
            values = []
            for row in rows:
                contribution = row.get("contribution") or {}
                delta = contribution.get("delta") if field == "memo_expressions" else contribution.get("search", {}).get("delta")
                values.append(delta.get(field, 0) if delta is not None else math.nan)
            axes[index, 1].plot(positions, values, "o", label=label)
        axes[index, 1].set(title="目标规则的搜索工作量贡献", ylabel="开启 − 关闭（次数）")
        if not any((r.get("contribution") or {}).get("delta") is not None for r in rows):
            axes[index, 1].text(0.5, 0.5, "无可比净差\n回退不作为同优化器对照", transform=axes[index, 1].transAxes,
                                ha="center", va="center")
        for field, label, color in (("delta_planning_ms", "规划", "#4477aa"), ("delta_execution_ms", "执行", "#ee7733")):
            for i, row in enumerate(rows):
                values = [p[field] for p in (row.get("contribution") or {}).get("pairs", []) if p[field] is not None]
                axes[index, 2].scatter([i] * len(values), values, color=color, alpha=0.5, label=label if i == 0 else None)
        axes[index, 2].legend(fontsize=8)
        axes[index, 2].axhline(0, color="gray", linewidth=0.7)
        axes[index, 2].set(title="独立无日志耗时配对", ylabel="开启 − 关闭（毫秒）；散点不是置信区间")
        if not any(p["delta_planning_ms"] is not None for r in rows for p in (r.get("contribution") or {}).get("pairs", [])):
            axes[index, 2].text(0.5, 0.5, "无可比耗时配对\n不报告加速比", transform=axes[index, 2].transAxes,
                                ha="center", va="center")
        for i, row in enumerate(rows):
            root = (row.get("contribution") or {}).get("target_root_costs") or {}
            values = [c["cost_gap"] for c in root.get("comparisons", []) if c["cost_gap"] is not None]
            axes[index, 3].scatter([i] * len(values), values, color="#ee7733")
        axes[index, 3].axhline(0, color="gray", linewidth=0.7)
        axes[index, 3].set(title="直接根候选相对当时最优的成本差", ylabel="估算成本差；无可用候选留空，不填零")
        application_cases = report.get("sampling_unit") == "predeclared_application_case"
        for axis in axes[index]:
            axis.set_xticks(positions, [(f"样本{i + 1}" if application_cases else
                                        "／".join("空值" if v is None else str(v)
                                                  for v in r["parameters"].values()))
                                       + f"\n输出{r.get('result_rows')}行"
                                       + ("\n关闭回退" if (r.get("contribution") or {}).get("off_fallback_cbo_ok") else "")
                                       for i, r in enumerate(rows)])
            axis.set_xlabel("样本编号（参数映射见同名 JSON）；不同模板不是因果对照"
                            if application_cases else "声明的参数值；不同模板不是因果对照")
            axis.grid(axis="y", alpha=0.2)
        axes[index, 0].legend(fontsize=7)
        axes[index, 1].legend(fontsize=8)
    fig.suptitle(f"规则 {report['rule_hash']} 的参数状态与条件贡献\n有限机制实验；规则始终在代价搜索阶段，未放宽安全约束，尚未验证总体泛化\n"
                 "关闭回退时不比较性能：净差留空；开启臂的候选观测独立保留")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_candidates(report: dict, output: Path, font: Path, rule: str | None) -> None:
    rows = [r for r in report["rows"] if rule is None or r["rule_hash"] == rule]
    if not rows:
        raise ValueError("no CBO candidate records for the requested rule")
    plt = chinese_plotting(font)
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    operators = sorted({(r.get("input_context") or {}).get("root", {}).get("operator", "unknown") for r in rows})
    names = {"CLogicalSelect": "过滤", "CLogicalGet": "扫描", "CLogicalProject": "投影",
             "CLogicalGbAgg": "聚合", "CLogicalInnerJoin": "内连接", "CLogicalLeftOuterJoin": "左外连接",
             "CLogicalLeftOuterCorrelatedApply": "相关左应用", "CLogicalLeftSemiApply": "半应用",
             "CLogicalLeftSemiJoin": "半连接", "CLogicalCTEConsumer": "公共表达式引用",
             "CLogicalCTEAnchor": "公共表达式锚点",
             "CLogicalLeftSemiCorrelatedApply": "相关半应用", "CLogicalLimit": "限制", "unknown": "未采集"}
    labels = [names.get(op, f"算子 {i + 1}") for i, op in enumerate(operators)]
    subsets = [[r for r in rows if (r.get("input_context") or {}).get("root", {}).get("operator", "unknown") == op]
               for op in operators]
    bottom = [0] * len(operators)
    for stage, (_, label, color) in STAGES.items():
        values = [sum(r["status"] == stage for r in subset) for subset in subsets]
        if any(values):
            axes[0, 0].bar(labels, values, bottom=bottom, label=label, color=color)
            bottom = [b + v for b, v in zip(bottom, values)]
        subset = [r for r in rows if r["status"] == stage]
        axes[1, 0].scatter([r["sequence"] for r in subset], [sum(r.get(t, 0) for t in TIMES) for r in subset],
                           label=label, color=color, s=12, alpha=0.65)
    axes[0, 0].set(title="局部根算子 × 每次尝试的终止阶段", ylabel="尝试次数；含明确标记的预算跳过")
    axes[0, 0].set_ylim(0, max(bottom) * 1.1)
    axes[0, 0].legend(fontsize=8)
    bottom = [0] * len(operators)
    for field, label in zip(TIMES, ("匹配", "约束检查", "目标构造")):
        values = [sum(r.get(field, 0) for r in subset) / 1000 for subset in subsets]
        axes[0, 1].bar(labels, values, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, values)]
    axes[0, 1].set(title="按输入根算子分解求值开销", ylabel="带跟踪的阶段计时合计（毫秒）")
    axes[0, 1].set_ylim(0, max(max(bottom) * 1.1, 1))
    axes[0, 1].legend()
    axes[1, 0].set(title="搜索过程中每次尝试的开销", xlabel="候选事件序号（不是独立随机样本）", ylabel="求值阶段合计（微秒）")
    for status, label, color in (("memo_inserted", "根插入成功", "#228833"),
                                 ("memo_duplicate", "根已存在", "#999999"),
                                 ("memo_cycle_rejected", "根因循环被拒绝", "#cc6677")):
        subset = [r for r in rows if (r["memo_outcome"] or {}).get("status") == status and r["direct_insertions"] is not None]
        if subset:
            axes[1, 1].scatter([r["sequence"] for r in subset], [r["direct_insertions"] for r in subset],
                               label=label, color=color, s=30)
    axes[1, 1].set(title="单次目标入库递归内新增的表达式", xlabel="对应的候选事件序号", ylabel="新增数（含标量及中间节点；不含未来规则链）")
    if axes[1, 1].collections:
        axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in axes[1]:
        axis.set_xlim(0, max(r["sequence"] for r in rows) + 1)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in axes[0]:
        axis.tick_params(axis="x", labelrotation=30)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    audit = "逐规则次数与阶段耗时校验通过" if report["complete"] else "记录不完整：仅作诊断，不能估计分布"
    fig.suptitle(f"代价搜索逐次尝试剖析｜{rule or '全部已调度规则'}｜{len(rows)} 次尝试\n{audit}\n"
                 "输入 → 匹配／约束／构造 → 目标入库；耗时不包含绑定构建，也不是执行收益")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--parameter-manifest", type=Path, help="retain every generated parameter instance in an observation report")
    parser.add_argument("--cohort", type=Path, help="frozen discovery sample, keeping failed/missing members")
    parser.add_argument("--sweep-suite", type=Path, help="fixed-cohort cardinality sweep suite.json")
    parser.add_argument("--sweep-manifest", type=Path, help="single-query one/joint-input manifest with --comparison")
    parser.add_argument("--cbo-contribution", help="with --sweep-suite or --sweep-manifest, compare this rule's OFF/CBO-only contribution")
    parser.add_argument("--results", type=Path, help="one runner batch directory for cohort analysis")
    parser.add_argument("--experiment-index", type=int, default=1, help="1-based pure-observation experiment in a batch")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--rule-hash")
    parser.add_argument("--shape-view", action="store_true", help="separate pre-evaluation shape from post-evaluation symbol bindings")
    parser.add_argument("--cost-view", action="store_true", help="audit shared costing events in a stats experiment")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=PGORCA_ROOT / "output/fonts/NotoSansCJKsc-Regular.otf")
    args = parser.parse_args()
    if args.sweep_manifest:
        if not args.comparison or any((args.results, args.cohort, args.sweep_suite, args.parameter_manifest,
                                       args.shape_view, args.cost_view, args.rule_hash)):
            parser.error("--sweep-manifest requires --comparison and optionally --cbo-contribution")
        manifest = json.loads(args.sweep_manifest.read_text())
        comparison = json.loads(args.comparison.read_text())
        if args.cbo_contribution:
            rule_hash = comparison.get("rule_profile", {}).get("rule_hash")
            if args.cbo_contribution != rule_hash:
                parser.error("--cbo-contribution must name the comparison's profiled rule")
            report = {"scope": "single_query_cardinality_cbo_minus_off_not_generalization",
                      "rule_hash": rule_hash,
                      "factor_coordinates": manifest.get("factor_design", {}).get("coordinates"),
                      "factor_labels": manifest.get("factor_design", {}).get("coordinate_labels"),
                      "runs": cbo_contribution(manifest, comparison, args.sweep_manifest.parent,
                                                args.comparison.parent)}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            render_cbo_contribution(report, args.output, args.font)
            render_search_contribution(report, args.output.with_name(args.output.name + "-search"), args.font)
            print(json.dumps({"points": len(report["runs"]),
                              "comparable": sum(row["delta"] is not None for row in report["runs"])}))
            return
        points = sweep_candidate_evidence(manifest, comparison, args.sweep_manifest.parent, args.comparison.parent)
        report = {"source": str(args.comparison.resolve()), "query_crc32": comparison["query_crc32"],
                  "manifest": str(args.sweep_manifest.resolve()), "targets": manifest["targets"], "points": points,
                  "postgres_oracle": comparison.get("postgres_oracle"),
                  "scope": "conditional_response_not_rule_causal_benefit_or_generalization",
                  "stage_path_fields": ["rule_hash", "status", "failed_constraint"]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        render_sweep_response(points, manifest["targets"], args.output, args.font)
        print(json.dumps(dict(Counter(p["status"] for p in points))))
        return
    if args.cost_view:
        if not args.comparison or any((args.parameter_manifest, args.results, args.cohort, args.sweep_suite,
                                      args.cbo_contribution, args.shape_view, args.rule_hash)):
            parser.error("--cost-view requires only --comparison and an optional --experiment-index")
        comparison = json.loads(args.comparison.read_text())
        experiments = comparison.get("stats_experiments", [])
        if not 1 <= args.experiment_index <= len(experiments):
            parser.error("experiment index is out of range")
        experiment = experiments[args.experiment_index - 1]
        report = cost_evidence(experiment["modes"]["replacement"])
        report.update(source=str(args.comparison.resolve()), experiment_index=args.experiment_index,
                      comparison=experiment["comparisons"]["replacement"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        render_costs(report, args.output, args.font)
        lifecycle = cost_lifecycle_evidence(experiment["modes"]["replacement"])
        lifecycle_output = args.output.with_name(args.output.name + "-lifecycle")
        lifecycle_output.with_suffix(".json").write_text(json.dumps(lifecycle, indent=2) + "\n")
        render_cost_lifecycle(lifecycle, lifecycle_output, args.font)
        render_selected_cost_tree(lifecycle, args.output.with_name(args.output.name + "-selected-plan"), args.font)
        checks = search_check_evidence(experiment["modes"]["replacement"])
        check_output = args.output.with_name(args.output.name + "-checks")
        check_output.with_suffix(".json").write_text(json.dumps(checks, indent=2) + "\n")
        render_search_checks(checks, check_output, args.font)
        if len(experiments) > 1:
            render_cost_response(experiments, args.output.with_name(args.output.name + "-response"), args.font, args.comparison.parent)
        print(json.dumps({**{k: report[k] for k in ("complete", "status_counts", "costed_with_rows", "exclusions")},
                          "lifecycle_complete": lifecycle["complete"], "lifecycle_exclusions": lifecycle["exclusions"],
                          "checks_complete": checks["complete"], "check_exclusions": checks["exclusions"],
                          "selected_distinct_candidates": lifecycle["selected_distinct_candidates"]}))
        return
    if args.parameter_manifest:
        if not args.results or any((args.cohort, args.comparison, args.sweep_suite, args.shape_view, args.rule_hash)):
            parser.error("--parameter-manifest requires --results and cannot be combined with other report modes")
        report = parameter_candidate_evidence(json.loads(args.parameter_manifest.read_text()), args.results, args.cbo_contribution)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        render_parameters(report, args.output, args.font)
        if args.cbo_contribution:
            render_parameter_contribution(report, args.output.with_name(args.output.name + "-contribution"), args.font)
        print(json.dumps(dict(Counter(r["status"] for r in report["runs"]))))
        return
    if args.cbo_contribution and not args.sweep_suite:
        parser.error("--cbo-contribution requires --sweep-suite")
    if args.sweep_suite:
        if args.cohort or args.comparison or not args.results or args.shape_view or args.rule_hash:
            parser.error("--sweep-suite requires --results and cannot be combined with other report modes")
        report = cohort_sweep_evidence(json.loads(args.sweep_suite.read_text()), args.results, args.cbo_contribution)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        if args.cbo_contribution:
            render_cbo_contribution(report, args.output, args.font)
            render_search_contribution(report, args.output.with_name(args.output.name + "-search"), args.font)
            print(json.dumps({"points": len(report["runs"]), "comparable": sum(r["delta"] is not None for r in report["runs"])}))
        else:
            render_cohort_sweep(report, args.output, args.font)
            print(json.dumps(dict(Counter(r["status"] for r in report["runs"]))))
        return
    if args.cohort:
        if args.comparison or not args.results or args.shape_view or args.rule_hash:
            parser.error("--cohort requires --results and cannot be combined with single-run plotting")
        report = cohort_candidate_evidence(json.loads(args.cohort.read_text()), args.results, args.experiment_index)
        summary = args.results / "summary.json"
        report.update(cohort=str(args.cohort.resolve()), results=str(args.results.resolve()),
                      fixture=json.loads(summary.read_text()).get("fixture") if summary.is_file() else None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        render_candidate_cohort(report, args.output, args.font)
        print(json.dumps({k: report[k] for k in ("selected_families", "complete_families")}))
        return
    if not args.comparison or args.results:
        parser.error("provide --comparison or --cohort with --results")
    comparison = json.loads(args.comparison.read_text())
    scenarios = comparison["rule_profile"]["scenarios"]
    if not 0 <= args.scenario < len(scenarios):
        parser.error("scenario is out of range")
    scenario = scenarios[args.scenario]
    report = candidate_evidence(scenario["arms"]["cbo"])
    report.update(source=str(args.comparison.resolve()), scenario=args.scenario,
                  stats_experiment=scenario["stats_experiment"], plot_rule=args.rule_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    renderer = render_shapes if args.shape_view else render_candidates
    renderer(report, args.output, args.font, args.rule_hash)
    print(json.dumps({key: report[key] for key in ("complete", "attempts", "evaluations", "exclusions")}))


if __name__ == "__main__":
    main()
