#!/usr/bin/env python3
"""Replay a redacted WeTune manifest through pgORCA, one case at a time."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from ml_orca.trace.compare_rule_traces import compare, read_records


def read_manifest(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if record.get("kind") != "differential_case":
                continue
            if not isinstance(record.get("case_id"), str) or not isinstance(record.get("query"), str):
                raise ValueError(f"{path}:{line_number}: case_id and query are required")
            if record["case_id"] in seen_case_ids:
                raise ValueError(f"{path}:{line_number}: duplicate case_id {record['case_id']}")
            applications = record.get(
                "expected_applications", record.get("reference_applications")
            )
            if not isinstance(applications, list) or not any(
                isinstance(application, dict) and "rule_id" in application
                for application in applications
            ):
                raise ValueError(
                    f"{path}:{line_number}: at least one expected/reference rule_id is required"
                )
            seen_case_ids.add(record["case_id"])
            cases.append(record)
    return cases


def reference_records(case: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    applications = case.get("expected_applications", case.get("reference_applications", []))
    for application in applications:
        if not isinstance(application, dict) or "rule_id" not in application:
            continue
        records.append(
            {
                "kind": "application",
                "engine": "wetune",
                "case_id": case["case_id"],
                "rule_id": application["rule_id"],
                "status": application.get("status", "applied"),
            }
        )
    return records


def rule_ids(path: Path) -> set[int]:
    """Return admitted physical source lines using the loader's conventions."""
    identifiers: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split("\t", 1)
            if len(fields) == 2 and fields[1].strip() not in {"", "EQ"}:
                continue
            identifiers.add(line_number)
    return identifiers


def rule_count(path: Path) -> int:
    return len(rule_ids(path))


def validate_rule_ids(cases: list[dict[str, Any]], rules_path: Path) -> None:
    available_ids = rule_ids(rules_path)
    expected = {
        application.get("rule_id")
        for case in cases
        for application in case.get(
            "expected_applications", case.get("reference_applications", [])
        )
        if isinstance(application, dict)
    }
    invalid = sorted(
        (
            rule_id
            for rule_id in expected
            if not isinstance(rule_id, int)
            or isinstance(rule_id, bool)
            or rule_id not in available_ids
        ),
        key=str,
    )
    if invalid:
        rendered = ", ".join(map(str, invalid))
        raise ValueError(
            f"manifest rule_id absent from supplied rule set "
            f"({len(available_ids)} admitted rules): {rendered}; "
            "check that the manifest and rules file were generated together"
        )


def trace_metrics(records: list[dict[str, Any]]) -> dict[str, int]:
    """Compact cumulative optimizer metrics without double-counting stages."""
    memo_records = [record for record in records if record.get("kind") == "memo_summary"]
    rule_records = [record for record in records if record.get("kind") == "rule_summary"]

    per_rule: dict[int, dict[str, int]] = {}
    rule_fields = (
        "binding_attempts",
        "generated_alternatives",
        "duplicate_alternatives",
        "memo_inserted_alternatives",
        "memo_duplicate_alternatives",
        "memo_cycle_rejected_alternatives",
        "budget_exhausted",
        "budget_skipped",
    )
    for record in rule_records:
        rule_id = record.get("rule_id")
        if not isinstance(rule_id, int) or isinstance(rule_id, bool):
            continue
        counters = per_rule.setdefault(rule_id, {field: 0 for field in rule_fields})
        # CEngine may print one cumulative summary per search stage. Taking the
        # maximum per rule preserves the query total instead of summing prefixes.
        for field in rule_fields:
            value = record.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                counters[field] = max(counters[field], value)

    metrics = {
        "memo_stages": len(memo_records),
        "peak_groups": 0,
        "peak_duplicate_groups": 0,
        "peak_group_expressions": 0,
        "rules_attempted": len(per_rule),
    }
    for record in memo_records:
        for field, target in (
            ("groups", "peak_groups"),
            ("duplicate_groups", "peak_duplicate_groups"),
            ("group_expressions", "peak_group_expressions"),
        ):
            value = record.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                metrics[target] = max(metrics[target], value)
    for field in rule_fields:
        metrics[field] = sum(counters[field] for counters in per_rule.values())
    return metrics


def alignment_summary(
    cases: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    comparable = [case for case in cases if "missing_rewritten" in case]
    subset_aligned = sum(
        not case["missing_rewritten"] and not case["inconclusive_budget"]
        for case in comparable
    )
    capability_subset_aligned = sum(
        not case.get("missing_capable", case["missing_rewritten"])
        for case in comparable
    )
    exact_trigger_set = sum(
        not case["missing_rewritten"]
        and not case["inconclusive_budget"]
        and not case["candidate_extra"]
        for case in comparable
    )
    missing_rules = Counter(
        rule_id for case in comparable for rule_id in case["missing_rewritten"]
    )
    missing_capable_rules = Counter(
        rule_id
        for case in comparable
        for rule_id in case.get("missing_capable", case["missing_rewritten"])
    )
    extra_rules = Counter(
        rule_id for case in comparable for rule_id in case["candidate_extra"]
    )
    application_statuses = Counter(
        record.get("status")
        for record in records
        if record.get("kind") == "application" and isinstance(record.get("status"), str)
    )
    priority_conflicts = Counter(
        (record.get("selected_rule_id"), record.get("rule_id"))
        for record in records
        if record.get("kind") == "application"
        and record.get("status") == "applicable_rbo"
        and isinstance(record.get("selected_rule_id"), int)
        and not isinstance(record.get("selected_rule_id"), bool)
        and isinstance(record.get("rule_id"), int)
        and not isinstance(record.get("rule_id"), bool)
    )
    return {
        "comparable": len(comparable),
        "subset_aligned": subset_aligned,
        "capability_subset_aligned": capability_subset_aligned,
        "exact_trigger_set": exact_trigger_set,
        "missing_rule_distribution": dict(sorted(missing_rules.items())),
        "missing_capability_distribution": dict(
            sorted(missing_capable_rules.items())
        ),
        "extra_rule_distribution": dict(sorted(extra_rules.items())),
        "priority_conflicts": [
            {
                "selected_rule_id": selected,
                "applicable_rule_id": applicable,
                "count": count,
            }
            for (selected, applicable), count in sorted(
                priority_conflicts.items()
            )
        ],
        "application_status_distribution": dict(sorted(application_statuses.items())),
    }


def safe_name(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._") or "case"


def parameter_count(query: str) -> int:
    """Return the highest $n parameter outside SQL strings and comments."""
    parameters: list[int] = []
    index = 0
    while index < len(query):
        if query[index] in {"'", '"'}:
            quote = query[index]
            index += 1
            while index < len(query):
                if query[index] == quote:
                    if index + 1 < len(query) and query[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            continue
        if query.startswith("--", index):
            newline = query.find("\n", index + 2)
            index = len(query) if newline < 0 else newline + 1
            continue
        if query.startswith("/*", index):
            end = query.find("*/", index + 2)
            index = len(query) if end < 0 else end + 2
            continue
        if query[index] == "$" and index + 1 < len(query) and query[index + 1].isdigit():
            end = index + 2
            while end < len(query) and query[end].isdigit():
                end += 1
            parameters.append(int(query[index + 1 : end]))
            index = end
            continue
        index += 1
    return max(parameters, default=0)


def render_trace_query(query: str) -> str:
    query = query.rstrip().rstrip(";")
    if parameter_count(query) == 0:
        return f"EXPLAIN (COSTS OFF)\n{query};\n"
    return f"EXPLAIN (GENERIC_PLAN, COSTS OFF)\n{query};\n"


def tail_text(path: Path, maximum: int = 2000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - maximum * 2))
            return stream.read().decode("utf-8", errors="replace")[-maximum:].strip()
    except OSError:
        return ""


def failed_query_status(detail: str) -> str:
    if "canceling statement due to statement timeout" in detail:
        return "timeout"
    return "query_error"


def orca_fallback_reason(path: Path) -> str | None:
    """Return a reason when EXPLAIN was produced by PostgreSQL, not ORCA."""
    marker = "Falling back to Postgres-based planner because GPORCA does not support"
    used_orca = False
    internal_error: str | None = None
    try:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding="utf-8", errors="replace") as stream:
            for line in stream:
                used_orca = used_orca or "Optimizer: pg_orca" in line
                position = line.find(marker)
                if position >= 0:
                    return line[position:].rstrip('\n\r\",')
                if internal_error is None and ',ERROR,"' in line:
                    internal_error = line.split(',ERROR,"', 1)[1].rstrip('\n\r\",')
    except OSError:
        pass
    if used_orca:
        return None
    return internal_error or "ORCA did not produce the EXPLAIN plan"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("rules", type=Path)
    parser.add_argument("schema", type=Path)
    parser.add_argument("--pg-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        help="query policy; use a rule: '*' BottomUp-RBO policy for full-bank replay",
    )
    parser.add_argument(
        "--strict-trigger-set",
        action="store_true",
        help="also fail when pgORCA rewrites a rule absent from the WeTune trace",
    )
    parser.add_argument(
        "--trace-runner", type=Path, default=script_dir / "trace_corpus.sh"
    )
    parser.add_argument("--base-port", type=int, default=55440)
    parser.add_argument(
        "--statement-timeout",
        type=int,
        default=60000,
        help="per-case PostgreSQL statement timeout in milliseconds",
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=[],
        help="run only the selected case_id; may be specified more than once",
    )
    parser.add_argument(
        "--disable-xform",
        action="append",
        default=[],
        help="disable a native ORCA xform for each replayed case; may be repeated",
    )
    parser.add_argument(
        "--dsl",
        choices=("on", "off"),
        default="on",
        help="enable or disable DSL rewrites during replay (default: on)",
    )
    parser.add_argument(
        "--dphyper",
        choices=("on", "off"),
        default="off",
        help="enable or disable DPHyper join enumeration (default: off)",
    )
    parser.add_argument(
        "--dphyper-shadow",
        choices=("on", "off"),
        default="on",
        help="retain native join enumeration beside DPHyper (default: on)",
    )
    parser.add_argument(
        "--dphyper-pair-budget",
        type=int,
        default=100,
        help="maximum CSG-CMP pairs before graph simplification (default: 100)",
    )
    parser.add_argument(
        "--dphyper-edge-budget",
        type=int,
        default=100000,
        help="maximum generated DPHyper edges",
    )
    parser.add_argument(
        "--max-alternatives",
        type=int,
        default=0,
        help="query-wide DSL alternative budget; zero is unlimited",
    )
    parser.add_argument(
        "--max-alternatives-per-rule",
        type=int,
        default=0,
        help="per-rule DSL alternative budget; zero is unlimited",
    )
    parser.add_argument("--max", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.max < 0
        or args.max_alternatives < 0
        or args.max_alternatives_per_rule < 0
        or not 1 <= args.base_port <= 65535
        or args.statement_timeout <= 0
        or args.dphyper_pair_budget <= 0
        or args.dphyper_edge_budget <= 0
    ):
        print(
            "invalid --max, --max-alternatives, "
            "--max-alternatives-per-rule, --base-port, --statement-timeout, "
            "or DPHyper budget",
            file=sys.stderr,
        )
        return 2
    invalid_xforms = [
        name
        for name in args.disable_xform
        if not re.fullmatch(r"CXform[A-Za-z0-9_]+", name)
    ]
    if invalid_xforms:
        print(f"invalid xform name: {', '.join(invalid_xforms)}", file=sys.stderr)
        return 2
    try:
        cases = read_manifest(args.manifest)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    rules_path = args.rules.resolve()
    try:
        validate_rule_ids(cases, rules_path)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    policy_path = args.policy.resolve() if args.policy is not None else None
    if policy_path is not None and not policy_path.is_file():
        print(f"policy is not a readable file: {policy_path}", file=sys.stderr)
        return 2
    if args.case_ids:
        selected = set(args.case_ids)
        cases = [case for case in cases if case["case_id"] in selected]
        missing = sorted(selected - {case["case_id"] for case in cases})
        if missing:
            print(f"case_id not found: {', '.join(missing)}", file=sys.stderr)
            return 2
    if args.max:
        cases = cases[: args.max]

    schema_path = args.schema.resolve()
    trace_runner = args.trace_runner.resolve()
    pg_config = args.pg_config.resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_jsonl = args.output_dir / "pgorca.jsonl"
    case_reports: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []

    if not cases:
        candidate_jsonl.write_text("", encoding="utf-8")
        report = {
            "kind": "corpus_comparison",
            "manifest": str(args.manifest),
            "rules": str(args.rules),
            "policy": str(policy_path) if policy_path is not None else None,
            "strict_trigger_set": args.strict_trigger_set,
            "dsl": args.dsl,
            "dphyper": args.dphyper,
            "dphyper_shadow": args.dphyper_shadow,
            "dphyper_pair_budget": args.dphyper_pair_budget,
            "dphyper_edge_budget": args.dphyper_edge_budget,
            "disabled_xforms": args.disable_xform,
            "max_alternatives": args.max_alternatives,
            "max_alternatives_per_rule": args.max_alternatives_per_rule,
            "attempted": 0,
            "passed": 0,
            "failed": 0,
            "alignment": alignment_summary([], []),
            "cases": [],
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    stems: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="pgorca-corpus-queries.") as directory:
        query_dir = Path(directory)
        for case in cases:
            case_id = case["case_id"]
            stem = safe_name(case_id)
            if stem in stems:
                print(f"case ids map to the same filename: {stems[stem]}, {case_id}", file=sys.stderr)
                return 2
            stems[stem] = case_id
            (query_dir / f"{stem}.sql").write_text(
                render_trace_query(case["query"]), encoding="utf-8"
            )

        environment = os.environ.copy()
        environment["PG_CONFIG"] = str(pg_config)
        environment["DSL_TRACE_PORT"] = str(args.base_port)
        environment["DSL_TRACE_STATEMENT_TIMEOUT"] = str(args.statement_timeout)
        environment["DSL_TRACE_DISABLE_XFORMS"] = ",".join(args.disable_xform)
        environment["DSL_TRACE_DSL_ENABLED"] = args.dsl
        environment["DSL_TRACE_POLICY_PATH"] = (
            str(policy_path) if policy_path is not None else ""
        )
        environment["DSL_TRACE_DPHYPER_ENABLED"] = args.dphyper
        environment["DSL_TRACE_DPHYPER_SHADOW"] = args.dphyper_shadow
        environment["DSL_TRACE_DPHYPER_PAIR_BUDGET"] = str(
            args.dphyper_pair_budget
        )
        environment["DSL_TRACE_DPHYPER_EDGE_BUDGET"] = str(
            args.dphyper_edge_budget
        )
        environment["DSL_TRACE_MAX_ALTERNATIVES"] = str(args.max_alternatives)
        environment["DSL_TRACE_MAX_ALTERNATIVES_PER_RULE"] = str(
            args.max_alternatives_per_rule
        )
        process = subprocess.run(
            [
                str(trace_runner),
                str(rules_path),
                str(schema_path),
                str(query_dir.resolve()),
                str(args.output_dir.resolve()),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )
    if process.returncode != 0:
        print(process.stderr.strip()[-4000:], file=sys.stderr)
        return 2

    statuses: dict[str, int] = {}
    status_path = args.output_dir / "status.tsv"
    try:
        with status_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 2:
                    raise ValueError(f"{status_path}:{line_number}: invalid status row")
                statuses[fields[0]] = int(fields[1])
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    for case in cases:
        case_id = case["case_id"]
        stem = safe_name(case_id)
        log_path = args.output_dir / "logs" / f"{stem}.log"
        returncode = statuses.get(stem)
        if returncode is None:
            case_reports.append(
                {"case_id": case_id, "status": "trace_error", "detail": "missing status"}
            )
            continue
        if returncode != 0:
            detail = tail_text(log_path)
            try:
                partial_records = read_records(log_path)
            except (OSError, ValueError):
                partial_records = []
            for record in partial_records:
                record["case_id"] = case_id
            candidate_records.extend(partial_records)
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": failed_query_status(detail),
                    "returncode": returncode,
                    "detail": detail,
                    "candidate_application_count": sum(
                        record.get("kind") == "application"
                        for record in partial_records
                    ),
                    "search_metrics": trace_metrics(partial_records),
                }
            )
            continue

        fallback = orca_fallback_reason(log_path)
        if fallback is not None:
            case_reports.append(
                {
                    "case_id": case_id,
                    "status": "orca_fallback",
                    "detail": fallback,
                }
            )
            continue

        try:
            parsed = read_records(log_path)
        except (OSError, ValueError) as error:
            case_reports.append(
                {"case_id": case_id, "status": "trace_error", "detail": str(error)}
            )
            continue
        for record in parsed:
            record["case_id"] = case_id
        candidate_records.extend(parsed)
        comparison = compare(reference_records(case), parsed, "rule_id")
        if comparison["missing_rewritten"]:
            status = "missing_rewrite"
        elif comparison["inconclusive_budget"]:
            status = "budget_limited"
        elif args.strict_trigger_set and comparison["candidate_extra"]:
            status = "trigger_set_mismatch"
        else:
            status = "passed"
        case_reports.append(
            {
                "case_id": case_id,
                "status": status,
                "missing_rewritten": comparison["missing_rewritten"],
                "missing_capable": comparison["missing_capable"],
                "inconclusive_budget": comparison["inconclusive_budget"],
                "candidate_extra": comparison["candidate_extra"],
                "candidate_capable_extra": comparison[
                    "candidate_capable_extra"
                ],
                "candidate_application_count": comparison["candidate_application_count"],
                "search_metrics": trace_metrics(parsed),
            }
        )

    with candidate_jsonl.open("w", encoding="utf-8") as stream:
        for record in candidate_records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    failed = sum(case["status"] not in {"passed", "budget_limited"} for case in case_reports)
    inconclusive = sum(case["status"] == "budget_limited" for case in case_reports)
    passed = sum(case["status"] == "passed" for case in case_reports)
    report = {
        "kind": "corpus_comparison",
        "manifest": str(args.manifest),
        "rules": str(args.rules),
        "policy": str(policy_path) if policy_path is not None else None,
        "strict_trigger_set": args.strict_trigger_set,
        "dsl": args.dsl,
        "dphyper": args.dphyper,
        "dphyper_shadow": args.dphyper_shadow,
        "dphyper_pair_budget": args.dphyper_pair_budget,
        "dphyper_edge_budget": args.dphyper_edge_budget,
        "disabled_xforms": args.disable_xform,
        "max_alternatives": args.max_alternatives,
        "max_alternatives_per_rule": args.max_alternatives_per_rule,
        "attempted": len(case_reports),
        "passed": passed,
        "failed": failed,
        "inconclusive": inconclusive,
        "alignment": alignment_summary(case_reports, candidate_records),
        "cases": case_reports,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
