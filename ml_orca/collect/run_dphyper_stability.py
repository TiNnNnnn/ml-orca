#!/usr/bin/env python3
"""Compare native ORCA and DPHyper replacement over imported WeTune SQL."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import subprocess
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ml_orca.common.paths import TEST_ASSETS
from ml_orca.collect.run_trace_corpus import parameter_count


DPHYPER_RE = re.compile(r"DPHyper: status=([a-z_]+) ([^\n\"]*)")
COST_RE = re.compile(r"cost=[0-9.eE+-]+\.\.([0-9.eE+-]+)")
JOB_PROFILE_RE = re.compile(
    r"Job profile: type=([a-z_]+) calls=([0-9]+) elapsed_us=([0-9]+)"
)
XFORM_PROFILE_RE = re.compile(
    r"(CXform[A-Za-z0-9_]+): ([0-9]+) calls, ([0-9]+) total bindings, "
    r"([0-9]+) alternatives generated, ([0-9]+)ms"
)
MEMO_SUMMARY_RE = re.compile(
    r'DSL_TRACE (\{"kind":"memo_summary"[^\n]*\})'
)
PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run every imported WeTune SQL with native ORCA and "
        "DPHyper replacement"
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=TEST_ASSETS / "corpus"
    )
    parser.add_argument("--pg-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rules", type=Path, default=TEST_ASSETS / "rules" / "framework.rules"
    )
    parser.add_argument(
        "--dataset", action="append", default=[],
        help="run only this dataset; may be repeated",
    )
    parser.add_argument(
        "--case", action="append", default=[],
        help="run only dataset:line-number; may be repeated",
    )
    parser.add_argument("--max-per-dataset", type=int, default=0)
    parser.add_argument("--statement-timeout", type=int, default=60000)
    parser.add_argument("--dphyper-pair-budget", type=int, default=100)
    parser.add_argument("--dphyper-edge-budget", type=int, default=100000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--query-jobs", type=int, default=1,
        help="concurrent queries per dataset server (default: 1)",
    )
    parser.add_argument("--base-port", type=int, default=60400)
    parser.add_argument(
        "--optimization-stats", action="store_true",
        help="collect scheduler job profiles for focused diagnosis",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse completed cases from an existing cases.jsonl",
    )
    return parser.parse_args()


def run_command(
    command: list[str], *, environment: dict[str, str] | None = None,
    input_text: str | None = None, timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=timeout,
        check=False,
    )


def pg_config_value(pg_config: Path, option: str) -> str:
    process = run_command([str(pg_config), option])
    if process.returncode != 0 or not process.stdout.strip():
        raise RuntimeError(process.stderr.strip() or f"pg_config {option} failed")
    return process.stdout.strip()


def imported_cases(dataset: str, path: Path) -> list[tuple[str, str]]:
    cases = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            query = line.strip()
            if query:
                cases.append((f"{dataset}:{line_number}", query))
    return cases


def parse_dphyper_events(stderr: str) -> list[dict[str, Any]]:
    events = []
    for match in DPHYPER_RE.finditer(stderr):
        event: dict[str, Any] = {"status": match.group(1)}
        for field in match.group(2).split():
            if "=" not in field:
                continue
            key, value = field.rstrip(",").split("=", 1)
            if re.fullmatch(r"-?[0-9]+", value):
                event[key] = int(value)
            else:
                event[key] = value
        events.append(event)
    return events


def tail(text: str, maximum: int = 2000) -> str:
    return text[-maximum:].strip()


def explain_sql(
    query: str, *, dphyper: bool, timeout_ms: int, pair_budget: int,
    edge_budget: int, optimization_stats: bool = False,
) -> str:
    explain_options = "GENERIC_PLAN, " if parameter_count(query) else ""
    query = query.rstrip().rstrip(";")
    return f"""
LOAD 'pg_orca';
SET pg_orca.enable_orca=on;
SET pg_orca.trace_fallback=on;
SET pg_orca.enable_dsl_rule=off;
SET pg_orca.enable_dphyper={'on' if dphyper else 'off'};
SET pg_orca.dphyper_shadow={'off' if dphyper else 'on'};
SET pg_orca.dphyper_pair_budget={pair_budget};
SET pg_orca.dphyper_edge_budget={edge_budget};
SET pg_orca.trace_dsl_rule=on;
SET optimizer_print_xform=off;
SET optimizer_print_xform_results=off;
SET optimizer_print_optimization_stats={'on' if optimization_stats else 'off'};
SET client_min_messages=log;
SET statement_timeout='{timeout_ms}ms';
SET optimizer_enable_query_parameter=on;
SET plan_cache_mode=force_generic_plan;
EXPLAIN ({explain_options}COSTS ON)
{query};
"""


def run_explain(
    psql: list[str], query: str, *, dphyper: bool, timeout_ms: int,
    pair_budget: int, edge_budget: int, optimization_stats: bool = False,
) -> dict[str, Any]:
    sql = explain_sql(
        query, dphyper=dphyper, timeout_ms=timeout_ms,
        pair_budget=pair_budget, edge_budget=edge_budget,
        optimization_stats=optimization_stats,
    )
    started = time.monotonic()
    try:
        process = run_command(
            psql, input_text=sql, timeout=timeout_ms / 1000.0 + 10.0
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "client_timeout",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "detail": tail((error.stderr or "") if isinstance(error.stderr, str) else ""),
            "dphyper_events": [],
        }

    elapsed_ms = round((time.monotonic() - started) * 1000, 3)
    combined = process.stderr + "\n" + process.stdout
    events = parse_dphyper_events(process.stderr)
    if process.returncode != 0:
        status = "statement_timeout" if any(
            marker in combined
            for marker in (
                "canceling statement due to statement timeout",
                "canceling statement due to user request",
            )
        ) else "query_error"
    elif "Optimizer: pg_orca" not in process.stdout:
        status = "postgres_fallback"
    else:
        status = "ok"
    cost_match = COST_RE.search(process.stdout)
    result: dict[str, Any] = {
        "status": status,
        "elapsed_ms": elapsed_ms,
        "dphyper_events": events,
    }
    if cost_match:
        result["total_cost"] = float(cost_match.group(1))
    job_profiles = {
        match.group(1): {
            "calls": int(match.group(2)),
            "elapsed_us": int(match.group(3)),
        }
        for match in JOB_PROFILE_RE.finditer(process.stderr)
    }
    if job_profiles:
        result["job_profiles"] = job_profiles
    xform_profiles = {
        match.group(1): {
            "calls": int(match.group(2)),
            "bindings": int(match.group(3)),
            "alternatives": int(match.group(4)),
            "elapsed_ms": int(match.group(5)),
        }
        for match in XFORM_PROFILE_RE.finditer(process.stderr)
    }
    if xform_profiles:
        result["xform_profiles"] = xform_profiles
    memo_summaries = []
    for match in MEMO_SUMMARY_RE.finditer(process.stderr):
        try:
            memo_summaries.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    if memo_summaries:
        result["memo_summary"] = memo_summaries[-1]
    if status != "ok":
        # ORCA reports the actual fallback/exception reason on stderr.  EXPLAIN
        # emits the often-large PostgreSQL fallback plan on stdout afterwards,
        # so tailing the concatenated streams can hide the diagnostic entirely.
        result["detail"] = tail(process.stderr, 8000) or tail(combined)
    return result


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                if "native" in record and "replacement" in record:
                    completed[record["case_id"]] = record
    return completed


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return round(ordered[index], 3)


def summarize(dataset: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    native_status = Counter(record["native"]["status"] for record in records)
    replacement_status = Counter(
        record["replacement"]["status"] for record in records
    )
    replacement_times = [
        record["replacement"]["elapsed_ms"] for record in records
    ]
    applied = 0
    simplified = 0
    greedy_fallback = 0
    dpv2_fallback = 0
    fallback_reasons: Counter[str] = Counter()
    for record in records:
        events = record["replacement"].get("dphyper_events", [])
        if any(event["status"] == "applied" for event in events):
            applied += 1
        if any(
            event["status"] == "applied"
            and event.get("enumeration") == "simplified"
            for event in events
        ):
            simplified += 1
        for event in events:
            if event["status"] == "fallback":
                fallback_reasons[str(event.get("reason", "unknown"))] += 1
                greedy_fallback += event.get("owner") in {
                    "greedy_nary", "greedy_binary"
                }
                dpv2_fallback += event.get("owner") == "native_nary"
    timeout_statuses = {"statement_timeout", "client_timeout"}
    regressions = [
        record["case_id"] for record in records
        if record["native"]["status"] == "ok"
        and record["replacement"]["status"] not in {"ok", *timeout_statuses}
    ]
    replacement_timeouts = [
        record["case_id"] for record in records
        if record["replacement"]["status"] in timeout_statuses
    ]
    ratios = [
        record["replacement"]["elapsed_ms"] / record["native"]["elapsed_ms"]
        for record in records
        if record["native"]["status"] == "ok"
        and record["replacement"]["status"] == "ok"
        and record["native"]["elapsed_ms"] > 0
    ]
    return {
        "dataset": dataset,
        "attempted": len(records),
        "executed_cases": sum("reused_from" not in record for record in records),
        "reused_cases": sum("reused_from" in record for record in records),
        "native_status": dict(sorted(native_status.items())),
        "replacement_status": dict(sorted(replacement_status.items())),
        "replacement_regressions": regressions,
        "replacement_timeouts": replacement_timeouts,
        "dphyper_applied_queries": applied,
        "dphyper_simplified_queries": simplified,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "greedy_fallback_events": greedy_fallback,
        "dpv2_fallback_events": dpv2_fallback,
        "replacement_elapsed_ms": {
            "median": round(statistics.median(replacement_times), 3)
            if replacement_times else 0.0,
            "p95": percentile(replacement_times, 0.95),
            "max": round(max(replacement_times), 3) if replacement_times else 0.0,
        },
        "replacement_native_ratio": {
            "median": round(statistics.median(ratios), 3) if ratios else 0.0,
            "p95": percentile(ratios, 0.95),
            "max": round(max(ratios), 3) if ratios else 0.0,
        },
    }


def run_dataset(
    args: argparse.Namespace, dataset_dir: Path, port: int,
    bindir: Path, selected_cases: set[str],
) -> dict[str, Any]:
    dataset = dataset_dir.name
    dataset_output = args.output_dir / dataset
    dataset_output.mkdir(parents=True, exist_ok=True)
    records_path = dataset_output / "cases.jsonl"
    completed = load_completed(records_path) if args.resume else {}
    query_results = {
        record["query"]: record
        for record in completed.values()
        if isinstance(record.get("query"), str)
    }
    cases = imported_cases(dataset, dataset_dir / "cases.sql")
    if selected_cases:
        cases = [case for case in cases if case[0] in selected_cases]
    if args.max_per_dataset:
        cases = cases[: args.max_per_dataset]

    environment = os.environ.copy()
    environment["MONSOON_DSL_RULES"] = str(args.rules.resolve())
    # PostgreSQL's Unix sockets must live on a filesystem that permits socket
    # nodes. Some build environments point TMPDIR at an artifact filesystem
    # that only supports regular files, so use the system socket location.
    with tempfile.TemporaryDirectory(
        prefix=f"pgorca-dphyper-{dataset}.", dir="/tmp"
    ) as temp:
        root = Path(temp)
        data_dir = root / "data"
        socket_dir = root / "socket"
        socket_dir.mkdir()
        server_log = root / "postgresql.log"
        initdb = run_command([
            str(bindir / "initdb"), "-D", str(data_dir), "--no-locale",
            "--encoding=UTF8", "--auth=trust",
        ])
        if initdb.returncode != 0:
            raise RuntimeError(f"{dataset}: initdb failed: {tail(initdb.stderr)}")
        started = False
        try:
            start = run_command([
                str(bindir / "pg_ctl"), "-D", str(data_dir), "-l",
                str(server_log), "-o",
                f"-c listen_addresses='' -c logging_collector=off "
                f"-k {socket_dir} -p {port}", "start",
            ], environment=environment)
            if start.returncode != 0:
                server_detail = ""
                if server_log.exists():
                    server_detail = tail(
                        server_log.read_text(encoding="utf-8", errors="replace"),
                        4000,
                    )
                raise RuntimeError(
                    f"{dataset}: server start failed: "
                    f"{tail(start.stderr)}\n{server_detail}"
                )
            started = True
            psql_base = [
                str(bindir / "psql"), "-X", "-v", "ON_ERROR_STOP=1",
                "-h", str(socket_dir), "-p", str(port), "-d", "postgres",
                "-qAt",
            ]
            create = run_command(
                [*psql_base, "-c", "CREATE EXTENSION pg_orca;"],
                environment=environment,
            )
            if create.returncode != 0:
                raise RuntimeError(f"{dataset}: extension failed: {tail(create.stderr)}")
            schema = run_command(
                [*psql_base, "-f", str((dataset_dir / "schema.sql").resolve())],
                environment=environment,
                timeout=max(120.0, args.statement_timeout / 1000.0 + 10.0),
            )
            if schema.returncode != 0:
                raise RuntimeError(f"{dataset}: schema failed: {tail(schema.stderr)}")

            mode_psql = [*psql_base, "-f", "-"]
            records: list[dict[str, Any]] = []
            pending: list[tuple[int, str, str]] = []
            stream_mode = "a" if args.resume else "w"
            with records_path.open(stream_mode, encoding="utf-8") as output:
                for index, (case_id, query) in enumerate(cases):
                    if case_id in completed:
                        records.append(completed[case_id])
                        continue
                    if query in query_results:
                        prior = query_results[query]
                        record = {
                            "case_id": case_id,
                            "query": query,
                            "reused_from": prior["case_id"],
                            "native": prior["native"],
                            "replacement": prior["replacement"],
                        }
                        records.append(record)
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                        continue
                    pending.append((index, case_id, query))

                def execute_case(item: tuple[int, str, str]) -> dict[str, Any]:
                    index, case_id, query = item
                    mode_order = ("native", "replacement")
                    if index % 2:
                        mode_order = tuple(reversed(mode_order))
                    results = {}
                    for mode in mode_order:
                        results[mode] = run_explain(
                            mode_psql, query, dphyper=mode == "replacement",
                            timeout_ms=args.statement_timeout,
                            pair_budget=args.dphyper_pair_budget,
                            edge_budget=args.dphyper_edge_budget,
                            optimization_stats=args.optimization_stats,
                        )
                    return {
                        "case_id": case_id,
                        "query": query,
                        "native": results["native"],
                        "replacement": results["replacement"],
                    }

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.query_jobs
                ) as query_executor:
                    futures = {
                        query_executor.submit(execute_case, item): item[1]
                        for item in pending
                    }
                    for future in concurrent.futures.as_completed(futures):
                        record = future.result()
                        case_id = record["case_id"]
                        query = record["query"]
                        results = {
                            "native": record["native"],
                            "replacement": record["replacement"],
                        }
                        records.append(record)
                        query_results[query] = record
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                        if results["native"]["status"] == "ok" and \
                                results["replacement"]["status"] != "ok":
                            with PRINT_LOCK:
                                print(
                                    f"regression {case_id}: "
                                    f"{results['replacement']['status']}", flush=True
                                )
            summary = summarize(dataset, records)
            (dataset_output / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with PRINT_LOCK:
                print(
                    f"done {dataset}: {len(records)} cases, "
                    f"{len(summary['replacement_regressions'])} regressions, "
                    f"{len(summary['replacement_timeouts'])} timeouts, "
                    f"{summary['greedy_fallback_events']} greedy fallbacks, "
                    f"{summary['dpv2_fallback_events']} DPv2 fallbacks",
                    flush=True,
                )
            return summary
        finally:
            if started:
                run_command([
                    str(bindir / "pg_ctl"), "-D", str(data_dir),
                    "stop", "-m", "fast",
                ])


def main() -> int:
    args = parse_args()
    if (
        args.jobs <= 0 or args.query_jobs <= 0 or args.max_per_dataset < 0
        or args.statement_timeout <= 0 or args.dphyper_pair_budget <= 0
        or args.dphyper_edge_budget <= 0
    ):
        print("invalid jobs, max, timeout, or DPHyper budget", file=os.sys.stderr)
        return 2
    args.corpus_dir = args.corpus_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.rules = args.rules.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.rules.is_file():
        print(f"rules file not found: {args.rules}", file=os.sys.stderr)
        return 2
    bindir = Path(pg_config_value(args.pg_config.resolve(), "--bindir"))
    datasets = sorted(
        path.parent for path in args.corpus_dir.glob("*/cases.sql")
        if (path.parent / "schema.sql").is_file()
    )
    if args.dataset:
        selected = set(args.dataset)
        datasets = [path for path in datasets if path.name in selected]
        missing = selected - {path.name for path in datasets}
        if missing:
            print(f"dataset not found: {', '.join(sorted(missing))}", file=os.sys.stderr)
            return 2
    selected_cases = set(args.case)
    if selected_cases:
        available_datasets = {case_id.split(":", 1)[0] for case_id in selected_cases}
        datasets = [path for path in datasets if path.name in available_datasets]
    if args.base_port < 1024 or args.base_port + len(datasets) > 65535:
        print("base port does not leave enough valid dataset ports", file=os.sys.stderr)
        return 2

    summaries = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_dataset = {
            executor.submit(
                run_dataset, args, dataset, args.base_port + index,
                bindir, selected_cases,
            ): dataset.name
            for index, dataset in enumerate(datasets)
        }
        for future in concurrent.futures.as_completed(future_to_dataset):
            dataset = future_to_dataset[future]
            try:
                summaries.append(future.result())
            except Exception as error:  # preserve other datasets and report setup failures
                failures.append({"dataset": dataset, "error": str(error)})
                with PRINT_LOCK:
                    print(f"failed {dataset}: {error}", file=os.sys.stderr, flush=True)
    summaries.sort(key=lambda summary: summary["dataset"])
    report = {
        "kind": "dphyper_stability",
        "statement_timeout_ms": args.statement_timeout,
        "pair_budget": args.dphyper_pair_budget,
        "edge_budget": args.dphyper_edge_budget,
        "datasets": summaries,
        "setup_failures": failures,
        "attempted": sum(summary["attempted"] for summary in summaries),
        "replacement_regressions": sum(
            len(summary["replacement_regressions"]) for summary in summaries
        ),
        "replacement_timeouts": sum(
            len(summary["replacement_timeouts"]) for summary in summaries
        ),
        "greedy_fallback_events": sum(
            summary["greedy_fallback_events"] for summary in summaries
        ),
        "dpv2_fallback_events": sum(
            summary["dpv2_fallback_events"] for summary in summaries
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if (
        failures or report["replacement_regressions"]
        or report["dpv2_fallback_events"]
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
