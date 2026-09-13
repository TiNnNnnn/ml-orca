#!/usr/bin/env python3
"""Merge the runtime native-xform registry with causal E2E evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from typing import Any

from ml_orca.trace.compare_rule_traces import read_records
from ml_orca.common.replacement_contract import validate_replacement_matrix


STATUS_RANK = {
    "unassessed": 0,
    "expressible": 1,
    "verified_direct_replacement": 2,
    "verified_full_scope_replacement": 2,
    "verified_partial_replacement": 2,
    "verified_full_replacement": 3,
}
CANONICAL_STATUS = {
    0: "unassessed",
    1: "expressible",
    2: "verified_partial_replacement",
    3: "verified_full_replacement",
}


def audit_memo_provenance(
    runtime: dict[str, Any],
    records: list[dict[str, Any]],
    allowed_native_origins: set[str] | None = None,
) -> dict[str, Any]:
    allowed = allowed_native_origins or set()
    xforms = {
        entry["name"]: entry
        for entry in runtime.get("xforms", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    alternatives = [
        record for record in records if record.get("kind") == "memo_alternative"
    ]
    sources = Counter(record.get("source", "unknown") for record in alternatives)
    origin_categories: Counter[str] = Counter()
    native_origins: Counter[str] = Counter()

    for record in alternatives:
        source = record.get("source")
        origin = record.get("origin")
        if origin == "input":
            origin_categories["input"] += 1
            continue
        entry = xforms.get(origin)
        if entry is None:
            category = (
                source
                if source in {"dsl", "dphyper", "dsl+dphyper"}
                else "unknown"
            )
        elif entry.get("category") == "implementation_property":
            category = "retained_cascades"
        else:
            category = entry.get("category", "unknown")
        origin_categories[category] += 1
        if category in {"semantic_rewrite", "join_enumeration", "unknown"}:
            native_origins[str(origin)] += 1

    unexpected = native_origins - Counter(
        {origin: native_origins[origin] for origin in allowed}
    )
    return {
        "logical_alternatives": len(alternatives),
        "sources": dict(sorted(sources.items())),
        "origin_categories": dict(sorted(origin_categories.items())),
        "native_logical_origins": dict(sorted(native_origins.items())),
        "unexpected_native_origins": dict(sorted(unexpected.items())),
    }


def read_matrices(expect_dir: Path) -> list[dict[str, Any]]:
    matrices = []
    for path in sorted(expect_dir.glob("*.expect")):
        expectation = json.loads(path.read_text(encoding="utf-8"))
        if "replacement" not in expectation:
            continue
        validate_replacement_matrix(expectation)
        replacement = expectation["replacement"]
        status = replacement.get("status")
        if status not in STATUS_RANK:
            raise ValueError(f"{path}: invalid replacement status: {status!r}")
        rules = replacement.get("dsl_rules", [])
        if (
            not isinstance(rules, list)
            or any(not isinstance(rule_id, int) or rule_id <= 0 for rule_id in rules)
        ):
            raise ValueError(f"{path}: replacement.dsl_rules must be positive ids")
        rule_hashes = replacement.get("dsl_rule_hashes", [])
        if (
            not isinstance(rule_hashes, list)
            or any(
                not isinstance(rule_hash, str)
                or re.fullmatch(r"[0-9a-f]{16}", rule_hash) is None
                for rule_hash in rule_hashes
            )
        ):
            raise ValueError(
                f"{path}: replacement.dsl_rule_hashes must be 16-digit lowercase hex"
            )
        scope = replacement.get("scope")
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError(f"{path}: replacement.scope must be non-empty")
        exclusions = replacement.get("excluded_native_domains", [])
        matrices.append(
            {
                "case": path.stem,
                "xforms": replacement["xforms"],
                "scope": scope,
                "status": status,
                "dsl_rules": rules,
                "dsl_rule_hashes": rule_hashes,
                "excluded_native_domains": exclusions,
            }
        )
    return matrices


def merge_inventory(
    runtime: dict[str, Any], matrices: list[dict[str, Any]], rule_graph: dict | None = None
) -> dict[str, Any]:
    if rule_graph is not None:
        known = {node['rule_hash'] for node in rule_graph['nodes']}
        for matrix in matrices:
            missing = set(matrix.get('dsl_rule_hashes', [])) - known
            if missing:
                raise ValueError(f"{matrix['case']}: replacement evidence references absent rules: "
                                 + ', '.join(sorted(missing)))
    xforms = runtime.get("xforms")
    if not isinstance(xforms, list):
        raise ValueError("runtime audit has no xforms array")
    by_name = {entry.get("name"): entry for entry in xforms}
    if len(by_name) != len(xforms) or None in by_name:
        raise ValueError("runtime audit contains duplicate or unnamed xforms")

    result_xforms = []
    for source in xforms:
        entry = dict(source)
        category = entry.get("category")
        if category == "join_enumeration":
            entry["replacement_status"] = (
                "replaced_by_dphyper"
                if entry.get("replacement_owner") == "dphyper"
                else "retained_native_gap"
            )
        elif category == "implementation_property":
            entry["replacement_status"] = "retained_in_cascades"
        else:
            entry["replacement_status"] = "unassessed"
        entry["evidence"] = []
        result_xforms.append(entry)
    result_by_name = {entry["name"]: entry for entry in result_xforms}

    for matrix in matrices:
        for name in matrix["xforms"]:
            if name not in result_by_name:
                raise ValueError(
                    f"{matrix['case']}: replacement target is not a native "
                    f"exploration xform: {name}"
                )
            entry = result_by_name[name]
            if entry["category"] == "implementation_property":
                raise ValueError(
                    f"{matrix['case']}: cannot attach DSL replacement evidence "
                    f"to {entry['category']} xform {name}"
                )
            evidence = {
                key: matrix.get(key, [] if key.startswith("dsl_rule") else None)
                for key in (
                    "case",
                    "scope",
                    "status",
                    "dsl_rules",
                    "dsl_rule_hashes",
                    "excluded_native_domains",
                )
            }
            entry["evidence"].append(evidence)
            if entry["category"] == "join_enumeration":
                continue
            if STATUS_RANK[matrix["status"]] > STATUS_RANK[
                entry["replacement_status"]
            ]:
                entry["replacement_status"] = CANONICAL_STATUS[
                    STATUS_RANK[matrix["status"]]
                ]

    summary = {
        "native_exploration_xforms": len(result_xforms),
        "semantic_rewrite_xforms": sum(
            entry["category"] == "semantic_rewrite" for entry in result_xforms
        ),
        "join_enumeration_xforms": sum(
            entry["category"] == "join_enumeration" for entry in result_xforms
        ),
        "implementation_property_xforms": sum(
            entry["category"] == "implementation_property"
            for entry in result_xforms
        ),
        "verified_partial_xforms": sum(
            entry["replacement_status"] == "verified_partial_replacement"
            for entry in result_xforms
        ),
        "verified_full_xforms": sum(
            entry["replacement_status"] == "verified_full_replacement"
            for entry in result_xforms
        ),
        "causal_matrix_cases": len(matrices),
    }
    return {
        "schema_version": 1,
        "summary": summary,
        "xform_sets": [matrix for matrix in matrices if len(matrix["xforms"]) > 1],
        "xforms": result_xforms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime_audit", type=Path)
    parser.add_argument("expect_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--allow-native-origin", action="append", default=[])
    parser.add_argument("--strict-provenance", action="store_true")
    parser.add_argument("--rule-graph", type=Path, help="production rule_graph.json for identity validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = json.loads(args.runtime_audit.read_text(encoding="utf-8"))
    graph = json.loads(args.rule_graph.read_text()) if args.rule_graph else None
    inventory = merge_inventory(runtime, read_matrices(args.expect_dir), graph)
    if args.trace is not None:
        inventory["memo_provenance"] = audit_memo_provenance(
            runtime, read_records(args.trace), set(args.allow_native_origin)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inventory["summary"], ensure_ascii=False))
    if (
        args.strict_provenance
        and inventory.get("memo_provenance", {}).get("unexpected_native_origins")
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
