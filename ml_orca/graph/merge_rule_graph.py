#!/usr/bin/env python3
"""Merge observed DSL rule edges into a persistent static rule graph."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from ml_orca.trace.compare_rule_traces import read_records


EDGE_FIELDS = (
    "src_rule",
    "dst_rule",
    "target_path",
    "src_target_path",
    "dst_source_path",
    "dst_binding_path",
    "path_kind",
    "scheduler",
    "evidence",
    "relation",
    "producer_relation",
    "producer_outcome",
)

TRACE_SUFFIXES = {".plan", ".trace"}


def edge_key(edge: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(edge.get(field) for field in EDGE_FIELDS)


def read_trace_inputs(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        traces = (
            sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix in TRACE_SUFFIXES
            )
            if path.is_dir()
            else [path]
        )
        for trace in traces:
            yield from read_records(trace)


def merge_graph(
    base: dict[str, Any], records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    graph = deepcopy(base)
    node_by_hash = {
        node.get("rule_hash"): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
    }
    edges = graph.setdefault("edges", [])
    indexed = {
        edge_key(edge): edge
        for edge in edges
        if isinstance(edge, dict) and edge.get("evidence") == "runtime_observed"
    }
    selected_by_state: dict[tuple[Any, ...], dict[str, Any]] = {}
    candidates: dict[tuple[Any, Any], dict[str, Any]] = {}

    def observe(edge: dict[str, Any], binding: object = None, status: object = None) -> None:
        key = edge_key(edge)
        merged = indexed.get(key)
        if merged is None:
            merged = edge
            merged["observations"] = 0
            merged["binding_path_counts"] = {}
            edges.append(merged)
            indexed[key] = merged
        merged["observations"] = int(merged.get("observations", 0)) + 1
        if isinstance(status, str):
            counts = merged.setdefault("candidate_status_counts", {})
            counts[status] = int(counts.get(status, 0)) + 1
        if isinstance(binding, str) and binding:
            counts = merged.setdefault("binding_path_counts", {})
            counts[binding] = int(counts.get(binding, 0)) + 1

    for record in records:
        if record.get("engine") != "pgorca":
            continue
        if record.get("kind") == "experiment_outcome":
            selected_rules = record.get("selected_plan_cbo_dsl_rules", {})
            if isinstance(selected_rules, dict):
                for rule, count in selected_rules.items():
                    if rule not in node_by_hash:
                        raise ValueError(
                            f"selected plan references unknown rule: {rule}"
                        )
                    node = node_by_hash[rule]
                    node["selected_plan_observations"] = int(
                        node.get("selected_plan_observations", 0)
                    ) + int(count)
                    node["selected_plan_queries"] = int(
                        node.get("selected_plan_queries", 0)
                    ) + 1
            selected_by_state.clear()
            candidates.clear()
            continue
        if record.get("kind") == "rule_candidate":
            rule = record.get("rule_hash")
            if rule not in node_by_hash:
                raise ValueError(f"candidate references unknown rule: {rule}")
            node = node_by_hash[rule]
            status = str(record.get("status", "unknown"))
            scheduler = str(record.get("placement", "unknown"))
            node["candidate_observations"] = int(
                node.get("candidate_observations", 0)
            ) + 1
            statuses = node.setdefault("candidate_status_counts", {})
            statuses[status] = int(statuses.get(status, 0)) + 1
            schedulers = node.setdefault("candidate_scheduler_counts", {})
            schedulers[scheduler] = int(schedulers.get(scheduler, 0)) + 1
            candidates[(record.get("experiment"), record.get("sequence"))] = record

            if scheduler != "rbo":
                continue
            state_key = (
                record.get("experiment"),
                record.get("state_fingerprint"),
                record.get("binding_fingerprint"),
            )
            if status == "applied_rbo":
                selected_by_state[state_key] = record
                continue
            selected = selected_by_state.get(state_key)
            if status != "applicable_rbo" or selected is None:
                continue
            src = selected.get("rule_hash")
            if src == rule:
                continue
            binding = record.get("binding_path", "r")
            observe(
                {
                    "src_rule": src,
                    "dst_rule": rule,
                    "target_path": binding,
                    "src_target_path": binding,
                    "dst_source_path": binding,
                    "path_kind": "candidate_state",
                    "scheduler": "rbo",
                    "evidence": "runtime_observed",
                    "relation": "ordered_before",
                },
                binding,
            )
            continue
        if record.get("kind") == "rule_candidate_outcome":
            rule = record.get("rule_hash")
            if rule not in node_by_hash:
                raise ValueError(f"candidate outcome references unknown rule: {rule}")
            candidate = candidates.get(
                (record.get("experiment"), record.get("candidate_sequence"))
            )
            if candidate is None or candidate.get("rule_hash") != rule:
                raise ValueError("candidate outcome has no matching candidate")
            node = node_by_hash[rule]
            status = str(record.get("status", "unknown"))
            outcomes = node.setdefault("candidate_outcome_counts", {})
            outcomes[status] = int(outcomes.get(status, 0)) + 1
            before = record.get("memo_version_before")
            after = record.get("memo_version_after")
            if isinstance(before, int) and isinstance(after, int):
                node["memo_version_delta_total"] = int(
                    node.get("memo_version_delta_total", 0)
                ) + after - before
            continue
        if record.get("kind") != "rule_edge":
            continue
        src = record.get("src_rule")
        dst = record.get("dst_rule")
        path = record.get("target_path")
        if src not in node_by_hash or dst not in node_by_hash:
            raise ValueError(f"runtime edge references unknown rule: {src} -> {dst}")
        if not isinstance(path, str) or not path:
            raise ValueError("runtime edge has no target_path")

        observed = {
            "src_rule": src,
            "dst_rule": dst,
            "target_path": path,
            "src_target_path": record.get("src_target_path", path),
            "dst_source_path": record.get("dst_source_path", "r"),
            "path_kind": record.get("path_kind", "instantiated_expression"),
            "scheduler": record.get("scheduler", "unknown"),
            "evidence": "runtime_observed",
            "relation": record.get("relation", "followed_by"),
        }
        for field in ("dst_binding_path", "producer_relation", "producer_outcome"):
            if field in record:
                observed[field] = record[field]
        observe(observed, record.get("binding_path"), record.get("candidate_status"))

    graph["schema_version"] = max(2, int(graph.get("schema_version", 1)))
    return graph


def dot_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def render_dot(graph: dict[str, Any]) -> str:
    lines = ["digraph dsl_rules {", "  rankdir=LR;"]
    for node in graph.get("nodes", []):
        rule = dot_escape(node["rule_hash"])
        label = (
            f'{dot_escape(node.get("rule_id", "?"))}: '
            f'{dot_escape(node.get("source_root", "?"))} -> '
            f'{dot_escape(node.get("target_root", "?"))}\\n{rule}'
        )
        lines.append(f'  "{rule}" [label="{label}"];')
    for edge in graph.get("edges", []):
        runtime = edge.get("evidence") == "runtime_observed"
        count = f' x{edge.get("observations", 0)}' if runtime else ""
        attrs = ",color=blue,penwidth=2" if runtime else ""
        label = edge.get("target_path", "?")
        if runtime:
            label = (
                f'{edge.get("relation", "followed_by")}\\n'
                f'{edge.get("scheduler", "unknown")}:'
                f'{edge.get("src_target_path", label)}->'
                f'{edge.get("dst_binding_path", edge.get("dst_source_path", "r"))}{count}'
            )
            if edge.get("producer_outcome"):
                label += f'\\n{edge["producer_outcome"]}'
        lines.append(
            f'  "{dot_escape(edge["src_rule"])}" -> '
            f'"{dot_escape(edge["dst_rule"])}" '
            f'[label="{dot_escape(label)}"{attrs}];'
        )
    for index, item in enumerate(graph.get("unresolved_inputs", [])):
        lines.append(
            f'  input_{index} [shape=box,style=dashed,label="Input @ '
            f'{dot_escape(item["target_path"])}"];'
        )
        lines.append(
            f'  "{dot_escape(item["src_rule"])}" -> input_{index} '
            f'[style=dashed,label="{dot_escape(item["target_path"])}"];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path, help="base rule_graph.json")
    parser.add_argument(
        "traces", type=Path, nargs="+", help="DSL trace files or artifact directories"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dot", type=Path)
    args = parser.parse_args()

    try:
        graph = json.loads(args.graph.read_text(encoding="utf-8"))
        merged = merge_graph(graph, read_trace_inputs(args.traces))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.dot:
            args.dot.parent.mkdir(parents=True, exist_ok=True)
            args.dot.write_text(render_dot(merged), encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
