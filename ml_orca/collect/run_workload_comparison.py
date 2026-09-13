#!/usr/bin/env python3
"""Compare native ORCA with DSL+DPHyper on fixed optimizer workloads."""

from __future__ import annotations

from ml_orca.common.paths import PGORCA_ROOT, TEST_ASSETS, source_file
from ml_orca.common.artifacts import artifact_snapshot, read_snapshot

import argparse
from collections import defaultdict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import difflib
import fnmatch
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
import zlib
from typing import Any

from ml_orca.trace.build_xform_replacement_inventory import merge_inventory, read_matrices
from ml_orca.collect.run_dphyper_stability import parse_dphyper_events


DEFAULT_WORKLOADS = TEST_ASSETS / "workloads"
DEFAULT_POLICY = TEST_ASSETS / "rules/empty_workload_cbo.policy"
# COPY output is already buffered. Do not reject valid large text/aggregate fields;
# set the process-wide CSV limit once, not concurrently in run_mode workers.
csv.field_size_limit(sys.maxsize)
XFORM_RE = re.compile(r"CXform[A-Za-z0-9_]+")
OPTIMIZATION_TIME_RE = re.compile(r"\[OPT\]: Total Optimization Time: (\d+)ms")
SERVER_FAILURE_MARKERS = (
    "server closed the connection unexpectedly",
    "database system is in recovery mode",
    "database system is not yet accepting connections",
    "connection refused",
)
RULE_COUNTER_FIELDS = (
    "binding_attempts",
    "bound_symbols",
    "match_rejected",
    "constraint_rejected",
    "instantiate_rejected",
    "generated_alternatives",
    "duplicate_alternatives",
    "budget_exhausted",
    "budget_skipped",
    "match_us",
    "constraint_us",
    "instantiate_us",
)
CURVE_SEARCH_FIELDS = (
    "memo_groups",
    "memo_group_expressions",
    "groups",
    "group_expressions",
    "bindings_built",
    "candidates_found",
    "rule_candidates",
    "rule_applications",
    "optimizer_cost",
    "optimization_ms",
    "planning_ms",
    "execution_ms",
    "optimizer_memory_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pg-config", required=True)
    parser.add_argument("--audit-bin", required=True)
    parser.add_argument("--workload", action="append", help="workload directory name; defaults to the four benchmark suites")
    parser.add_argument("-t", "--test", action="append", default=[], help="glob of workload/query_stem")
    parser.add_argument("--workload-dir", type=Path, default=DEFAULT_WORKLOADS)
    parser.add_argument("--rule-file", type=Path, default=TEST_ASSETS / "rules/orca_replacements.rules")
    parser.add_argument("--policy-file", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--stats-experiment", type=Path, action="append", default=[],
        help="also run each query with this cardinality experiment (repeatable)",
    )
    parser.add_argument(
        "--profile-rule",
        help="also run causal OFF/RBO/CBO arms for this hash; other rules use engine defaults, not --policy-file",
    )
    parser.add_argument("--profile-cbo-only", action="store_true",
                        help="profile only OFF/CBO; never construct or execute an RBO policy")
    parser.add_argument("--compare-policy", action="append", default=[], metavar="NAME=FILE",
                        help="compare complete policies (repeatable); first is baseline, excludes --profile-rule")
    parser.add_argument("--profile-companion-rule", action="append", default=[],
                        help="toggle another rule with --profile-rule as one OFF/CBO bundle (repeatable)")
    parser.add_argument(
        "--profile-rbo-phase",
        choices=("normalize", "pre_join", "cleanup"),
        default="pre_join",
    )
    parser.add_argument(
        "--profile-effect",
        choices=("changes_join_graph", "preserves_join_graph", "join_reordering"),
        default="preserves_join_graph",
    )
    parser.add_argument("--unbounded", action="store_true")
    parser.add_argument("--output", type=Path, default=PGORCA_ROOT / "output/dsl-workloads")
    parser.add_argument("--port", type=int, default=60460)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=0,
                        help="untraced randomized complete-block OFF/RBO/CBO timing rounds (0 disables)")
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--postgres-oracle", action="store_true",
                        help="also check result bags against PostgreSQL with ORCA disabled")
    parser.add_argument("--setup-sql", type=Path, help="load data after schema, in the isolated test database")
    parser.add_argument("--capture-pre-context", action="store_true",
                        help="capture read-only catalog/context before workload arms; outside measured timings")
    parser.add_argument("--feature-graph", type=Path,
                        help="freeze an existing rule graph before collection; requires --capture-pre-context")
    parser.add_argument("--cardinality-probes", type=Path,
                        help="JSON targets with operator/fingerprint and row-producing SQL to count")
    parser.add_argument("--input-stats-reference", type=Path,
                        help="preload native discovery comparison.json as external input statistics, not optimizer stats")
    parser.add_argument("--dro-contract", type=Path, help="bind a frozen DRO contract before starting PostgreSQL")
    parser.add_argument("--dro-unit", help="preassigned independent unit ID (repeats remain inside this unit)")
    parser.add_argument("--dro-describe", action="store_true",
                        help="write acquisition descriptor only; do not initialize or start PostgreSQL")
    args = parser.parse_args()
    try:
        args.comparison_policies = parse_comparison_policies(args.compare_policy)
    except ValueError as error:
        parser.error(str(error))
    if args.comparison_policies and (args.profile_rule or args.profile_companion_rule or args.profile_cbo_only
                                     or args.dro_contract or args.dro_describe or args.input_stats_reference):
        parser.error('--compare-policy cannot mix with rule profiling or the existing OFF/CBO DRO contract')
    if args.feature_graph and (not args.capture_pre_context or args.dro_contract or args.dro_describe):
        parser.error('--feature-graph requires --capture-pre-context and is outside the legacy DRO contract')
    if args.feature_graph and not args.feature_graph.is_file():
        parser.error('feature graph must be an existing JSON file')
    if any(not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_-]*", name) for name in (args.workload or [])):
        parser.error("workload must be a directory name, not a path")
    if args.profile_cbo_only and not args.profile_rule:
        parser.error("--profile-cbo-only requires --profile-rule")
    if args.timing_repeats < 0 or args.timing_warmups < 0:
        parser.error("timing repeats/warmups must be nonnegative")
    if args.timing_repeats and (not (args.profile_rule or args.comparison_policies) or args.jobs != 1):
        parser.error("timing requires --profile-rule or --compare-policy, and --jobs 1")
    if bool(args.dro_contract) != bool(args.dro_unit) or (args.dro_describe and args.dro_contract):
        parser.error("pair --dro-contract with --dro-unit; --dro-describe is a separate preflight")
    if (args.dro_contract or args.dro_describe) and not (
            args.profile_cbo_only and args.timing_repeats > 0 and args.postgres_oracle):
        parser.error("DRO collection requires --profile-cbo-only, positive --timing-repeats and --postgres-oracle")
    if args.unbounded:
        args.policy_file = None
    missing = [path for path in args.stats_experiment if not path.is_file()]
    if missing:
        parser.error(f"stats experiment not found: {missing[0]}")
    if args.profile_rule and not re.fullmatch(r"[0-9a-fA-F]{16}", args.profile_rule):
        parser.error("profile rule must be a 16-digit canonical hexadecimal identity")
    if args.profile_rule:
        args.profile_rule = args.profile_rule.lower()
    if args.profile_companion_rule:
        try:
            profile_targets(args)
        except ValueError as error:
            parser.error(str(error))
    if args.input_stats_reference and (not args.setup_sql or not args.profile_rule or not args.stats_experiment):
        parser.error("input stats reference requires --setup-sql, --profile-rule and --stats-experiment")
    if args.setup_sql or args.cardinality_probes:
        if not args.workload or len(args.workload) != 1:
            parser.error("data setup/probes require exactly one explicit --workload")
        for path in (args.setup_sql, args.cardinality_probes):
            if path is not None and not path.is_file():
                parser.error(f"fixture not found: {path}")
    return args


def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, **kwargs)


def parse_comparison_policies(specifications: list[str]) -> dict[str, Path]:
    policies = {}
    for specification in specifications:
        name, separator, filename = specification.partition('=')
        if (not separator or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', name)
                or name in {'native', 'replacement'} or name in policies or not Path(filename).is_file()):
            raise ValueError('policy requires a unique safe NAME=existing-file; native/replacement are reserved')
        policies[name] = Path(filename).resolve()
    if len(policies) == 1:
        raise ValueError('complete-policy comparison requires at least two policies')
    return policies


def freeze_feature_graph(source: Path, output: Path) -> dict:
    """Freeze bytes before collection, not a certificate of historical/test independence."""
    if source.resolve().is_relative_to(output.resolve()):
        raise ValueError('feature graph must be outside the new output directory')
    raw = source.read_bytes()
    graph = json.loads(raw)
    if (not isinstance(graph, dict) or graph.get('schema_version') != 2
            or not isinstance(graph.get('nodes'), list) or not isinstance(graph.get('edges'), list)):
        raise ValueError('expected a version 2 rule graph with nodes and edges')
    json.dumps(graph, allow_nan=False)
    hashes = []
    for node in graph['nodes']:
        identity = node.get('rule_hash') if isinstance(node, dict) else None
        if not isinstance(identity, str) or not re.fullmatch(r'[0-9a-f]{16}', identity):
            raise ValueError('graph nodes require canonical rule hashes')
        hashes.append(identity)
    known = set(hashes)
    if len(known) != len(hashes):
        raise ValueError('duplicate rule node in feature graph')
    if any(not isinstance(edge, dict) or not isinstance(edge.get('src_rule'), str)
           or not isinstance(edge.get('dst_rule'), str)
           or edge['src_rule'] not in known or edge['dst_rule'] not in known for edge in graph['edges']):
        raise ValueError('feature graph edge references an unknown rule')
    path = output / 'feature-graph.json'
    with path.open('xb') as stream:
        stream.write(raw)  # Keep all parallel positions, counts and original provenance.
    snapshot = artifact_snapshot({'graph': path})['graph']
    if snapshot.get('size') != len(raw) or snapshot.get('crc32') != f'{zlib.crc32(raw):08x}':
        raise ValueError('feature graph copy changed during capture')
    receipt = {'schema_version': 1, 'capture': 'before_server_start',
               'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'pre_run_graph_copy_not_historical_split_certificate',
               'source': {**snapshot, 'path': str(source.resolve())}, 'snapshot': snapshot,
               'nodes': len(hashes), 'edges': len(graph['edges']),
               'historical_split_verified': False}
    with (output / 'feature-graph-snapshot.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    return receipt


def psql(
    binary: Path,
    socket: Path,
    port: int,
    database: str,
    sql: str,
    timeout: int,
    retry_on_server_failure: bool = True,
) -> tuple[str, str, int, float]:
    start = time.perf_counter()
    command = [
        str(binary), "-X", "-qAt", "-v", "ON_ERROR_STOP=1",
        "-h", str(socket), "-p", str(port), "-d", database,
    ]
    for attempt in range(2 if retry_on_server_failure else 1):
        try:
            result = run(
                command,
                input=f"SET statement_timeout='{timeout * 1000}ms';\n{sql}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout + 5,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
            return stdout, stderr + "\nTIMEOUT", 124, 1000 * (time.perf_counter() - start)
        if retry_on_server_failure and attempt == 0 and server_failure(result.stderr) and wait_ready(
            binary.with_name("pg_isready"), socket, port, database
        ):
            continue
        return result.stdout, result.stderr, result.returncode, 1000 * (time.perf_counter() - start)
    raise AssertionError("unreachable")


def collect_cardinalities(binary: Path, socket: Path, port: int, database: str,
                          probes: list[dict], timeout: int) -> list[dict]:
    identities = set()
    for probe in probes:
        identity = (probe.get("fingerprint"), probe.get("operator"))
        if (not re.fullmatch(r"[0-9a-f]{16}", identity[0] or "")
                or not re.fullmatch(r"CLogical[A-Za-z0-9]+", identity[1] or "")
                or not isinstance(probe.get("sql"), str) or not probe["sql"].strip()
                or identity in identities):
            raise ValueError("cardinality probes require distinct operator/fingerprint and SQL")
        identities.add(identity)
    results = []
    for probe in probes:
        sql = probe["sql"].strip().removesuffix(";")
        stdout, stderr, rc, elapsed = psql(binary, socket, port, database,
            "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;\n"
            "SET LOCAL pg_orca.enable_orca=off;\n"
            f"SELECT COUNT(*) FROM ({sql}) AS cardinality_probe;\nROLLBACK;", timeout)
        valid = rc == 0 and re.fullmatch(r"[0-9]+", stdout.strip()) is not None
        results.append({**probe, "status": "ok" if valid else "error", "returncode": rc,
                        "actual_rows": int(stdout.strip()) if valid else None,
                        "elapsed_ms": elapsed, "error": stderr if not valid else ""})
    return results


def settings(
    mode: str,
    semantic_xforms: list[str],
    policy_file: Path | None,
    stats_experiment: Path | None = None,
) -> str:
    replacement = mode == "replacement"
    disabled = "\n".join(
        f"DO $dsl$ BEGIN PERFORM disable_xform('{name}'); END $dsl$;"
        for name in semantic_xforms
    ) if replacement else ""
    policy = "RESET pg_orca.dsl_rule_policy_path;"
    if replacement and policy_file is not None:
        path = str(policy_file.resolve()).replace("'", "''")
        policy = f"SET pg_orca.dsl_rule_policy_path='{path}';"
    stats = "RESET pg_orca.dsl_stats_experiment_path;"
    if stats_experiment is not None:
        path = str(stats_experiment.resolve()).replace("'", "''")
        stats = f"SET pg_orca.dsl_stats_experiment_path='{path}';"
    return f"""
LOAD 'pg_orca';
SET pg_orca.enable_orca=on;
SET pg_orca.enable_dsl_rule={'on' if replacement else 'off'};
{policy}
{stats}
SET pg_orca.enable_dphyper={'on' if replacement else 'off'};
SET pg_orca.dphyper_shadow=off;
SET pg_orca.enable_assert_maxonerow=off;
SET pg_orca.dsl_only_xforms='';
SET pg_orca.trace_fallback=on;
{disabled}
"""


def collect_policy_context(binary: Path, rules: Path, policies: dict, timeout: int) -> dict:
    """Use the native loader/compiler, preserving failures instead of assuming CBO."""
    snapshots = {}
    for name, path in policies.items():
        command = [str(binary), '--policy-snapshot', str(rules)]
        if path is not None:
            command.append(str(path))
        start = time.perf_counter()
        data, error, rc = None, '', None
        try:
            result = run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            rc = result.returncode
            if rc:
                error = result.stderr
            else:
                data = json.loads(result.stdout)
                if (not isinstance(data, dict) or data.get('schema_version') != 1
                        or data.get('scope') != 'native_policy_snapshot_not_runtime_applicability'
                        or not isinstance(data.get('load'), dict)
                        or not isinstance(data.get('rules'), list)
                        or data['load'].get('admitted') != len(data['rules'])
                        or any(type(data['load'].get(k)) is not int or data['load'][k] < 0
                               for k in ('admitted', 'skipped_non_eq', 'failed'))):
                    raise ValueError('invalid native policy snapshot structure')
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as failure:
            data, error = None, str(failure)
        snapshots[name] = {'status': 'error' if data is None else 'partial' if data['load']['failed'] else 'ok',
                           'returncode': rc, 'elapsed_ms': 1000 * (time.perf_counter() - start),
                           'error': error, 'snapshot': data}
    return snapshots


def collect_catalog_context(binary: Path, socket: Path, port: int, database: str, timeout: int) -> dict:
    """Existing catalog estimates, never actual counts or forced ANALYZE/ORCA derivation."""
    sql = """
LOAD 'pg_orca';
BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL pg_orca.enable_orca=off;
WITH relations AS (
  SELECT c.oid, n.nspname AS schema, c.relname AS name, c.relkind,
         CASE WHEN c.reltuples >= 0 THEN c.reltuples END AS estimated_rows,
         c.relpages AS pages, c.relispartition AS is_partition
  FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
  WHERE c.relkind IN ('r','p','m','f') AND n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
), columns AS (
  SELECT a.attrelid AS relation_oid, a.attnum AS position, a.attname AS name,
         pg_catalog.format_type(a.atttypid,a.atttypmod) AS type,
         a.attnotnull AS not_null, a.attstattarget AS statistics_target,
         a.attcollation AS collation_oid
  FROM pg_catalog.pg_attribute a JOIN relations r ON r.oid=a.attrelid
  WHERE a.attnum > 0 AND NOT a.attisdropped
), statistics AS (
  SELECT r.oid AS relation_oid, a.attnum AS position, s.inherited,
         s.null_frac, s.avg_width, s.n_distinct, s.correlation, s.most_common_freqs
  FROM relations r JOIN pg_catalog.pg_stats s ON s.schemaname=r.schema AND s.tablename=r.name
  JOIN pg_catalog.pg_attribute a ON a.attrelid=r.oid AND a.attname=s.attname
), constraints AS (
  SELECT c.conrelid AS relation_oid, c.conname AS name, c.contype AS kind,
         c.conkey AS column_positions, c.confrelid AS referenced_relation_oid,
         c.confkey AS referenced_column_positions, c.convalidated AS validated,
         c.condeferrable AS deferrable, c.condeferred AS initially_deferred,
         pg_catalog.pg_get_constraintdef(c.oid) AS definition
  FROM pg_catalog.pg_constraint c JOIN relations r ON r.oid=c.conrelid
), indexes AS (
  SELECT i.indrelid AS relation_oid, i.indexrelid AS index_oid,
         i.indisunique AS is_unique, i.indisvalid AS is_valid, i.indisready AS is_ready,
         pg_catalog.pg_get_indexdef(i.indexrelid) AS definition
  FROM pg_catalog.pg_index i JOIN relations r ON r.oid=i.indrelid
)
SELECT json_build_object(
  'captured_at', statement_timestamp(), 'database', current_database(),
  'server_version', current_setting('server_version'),
  'relations', COALESCE((SELECT json_agg(r ORDER BY r.oid) FROM relations r), '[]'::json),
  'columns', COALESCE((SELECT json_agg(c ORDER BY c.relation_oid,c.position) FROM columns c), '[]'::json),
  'statistics', COALESCE((SELECT json_agg(s ORDER BY s.relation_oid,s.position,s.inherited) FROM statistics s), '[]'::json),
  'constraints', COALESCE((SELECT json_agg(c ORDER BY c.relation_oid,c.name) FROM constraints c), '[]'::json),
  'indexes', COALESCE((SELECT json_agg(i ORDER BY i.relation_oid,i.index_oid) FROM indexes i), '[]'::json),
  'settings', (SELECT json_object_agg(name,setting) FROM pg_catalog.pg_settings
    WHERE name LIKE 'pg_orca.%' OR category LIKE 'Query Tuning%' OR
      name IN ('work_mem','effective_cache_size','max_parallel_workers','search_path')));
ROLLBACK;
"""
    stdout, stderr, rc, elapsed = psql(binary, socket, port, database, sql, timeout,
                                     retry_on_server_failure=False)
    error, data = stderr if rc else "", None
    if not rc:
        try:
            data = json.loads(stdout)
            if (not isinstance(data, dict) or not data.get('captured_at')
                    or not isinstance(data.get('settings'), dict)
                    or any(not isinstance(data.get(k), list) for k in
                           ('relations', 'columns', 'statistics', 'constraints', 'indexes'))):
                raise ValueError('invalid catalog snapshot structure')
        except (ValueError, TypeError) as failure:
            data, error = None, str(failure)
    return {'schema_version': 1, 'status': 'ok' if data is not None else 'error',
            'returncode': rc, 'elapsed_ms': elapsed, 'error': error, 'catalog': data,
            'capture': 'after_fixture_before_workload_arms',
            'scope': 'existing_catalog_estimates_not_candidate_group_statistics',
            'settings_scope': 'collector_backend_with_orca_disabled_not_arm_effective_settings',
            'ndistinct_encoding': 'pg_stats_raw_negative_fraction_of_relation_rows',
            'not_collected': ['histogram_bounds', 'most_common_values', 'extended_statistics',
                              'candidate_group_estimates', 'continuous_catalog_change_monitoring']}


def trace_settings(
    mode: str,
    semantic_xforms: list[str],
    policy_file: Path | None,
    stats_experiment: Path | None = None,
) -> str:
    native_xform_trace = "on" if mode == "native" else "off"
    return settings(mode, semantic_xforms, policy_file, stats_experiment) + """
SET optimizer_print_xform={xform_trace};
SET optimizer_print_xform_results={xform_trace};
SET optimizer_print_optimization_stats=on;
SET pg_orca.trace_dsl_rule=on;
SET client_min_messages=log;
""".format(xform_trace=native_xform_trace)


def trace_records(text: str) -> list[dict[str, Any]]:
    records = []
    contexts = {}
    for line in text.splitlines():
        if "DSL_TRACE {" not in line:
            continue
        payload = line.split("DSL_TRACE ", 1)[1]
        try:
            record = json.loads(payload[: payload.rfind("}") + 1])
        except (ValueError, json.JSONDecodeError):
            continue
        if record.get("kind") == "candidate_context":
            key = record.get("field"), record.get("context_id")
            if key[0] not in ("input_context", "binding_context") or type(key[1]) is not int or key[1] < 1:
                raise ValueError("invalid candidate context definition")
            if not isinstance(record.get("value"), dict) or (key in contexts and contexts[key] != record["value"]):
                raise ValueError("invalid or conflicting candidate context definition")
            contexts[key] = record["value"]
        if record.get("kind") == "rule_candidate":
            for field in ("input_context", "binding_context"):
                if field + "_id" not in record:
                    continue
                value = contexts.get((field, record[field + "_id"]))
                if value is None or (field in record and record[field] != value):
                    record.setdefault("context_resolution_errors", []).append(field)
                else:
                    record[field] = deepcopy(value)
        records.append(record)
        if record.get("kind") == "experiment_outcome":
            contexts.clear()  # Context IDs are query-local; support concatenated completed runs.
    return records


def error_summary(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if "ERROR:" in line
        or "FATAL:" in line
        or ',ERROR,"' in line
        or line.startswith("INFO:  pg_orca: falling back")
        or (line.startswith("DETAIL:") and "Failed assertion" in line)
        or any(marker in line for marker in SERVER_FAILURE_MARKERS)
        or line == "TIMEOUT"
    )


def server_failure(text: str) -> bool:
    return any(marker in text for marker in SERVER_FAILURE_MARKERS)


def wait_ready(binary: Path, socket: Path, port: int, database: str) -> bool:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = run(
            [str(binary), "-q", "-h", str(socket), "-p", str(port), "-d", database],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True
        time.sleep(0.1)
    return False


def produced_xforms(text: str) -> list[str]:
    result = []
    current = None
    alternatives = False
    for line in text.splitlines():
        if 'TRACE,"Xform: ' in line:
            current = line.split('TRACE,"Xform: ', 1)[1].strip()
            alternatives = False
        elif current and line.startswith("Alternatives:"):
            alternatives = True
        elif current and alternatives and line.startswith("0:"):
            result.append(current)
            current = None
    return result


def plan_tree(stdout: str) -> dict[str, Any] | None:
    try:
        value = json.loads(stdout)
        return value[0]["Plan"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None


def optimizer_name(stdout: str) -> str | None:
    try:
        value = json.loads(stdout)
        return value[0].get("Optimizer", "postgres")
    except (IndexError, TypeError, json.JSONDecodeError):
        return None


def explain_times(stdout: str) -> tuple[float | None, float | None]:
    try:
        value = json.loads(stdout)[0]
        return value.get("Planning Time"), value.get("Execution Time")
    except (IndexError, TypeError, json.JSONDecodeError):
        return None, None


def optimization_time(trace: str) -> int | None:
    matches = OPTIMIZATION_TIME_RE.findall(trace)
    return int(matches[-1]) if matches else None


def walk_plan(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    if plan is None:
        return []
    return [plan, *(node for child in plan.get("Plans", []) for node in walk_plan(child))]


def plan_difference(left: dict[str, Any] | None, right: dict[str, Any] | None) -> str:
    if left is None or right is None:
        return "plan_error"
    # EXPLAIN ANALYZE's execution counters are not optimizer plan properties.
    left, right = timing_plan(left), timing_plan(right)
    if left == right:
        return "identical"
    left_nodes = [node.get("Node Type") for node in walk_plan(left)]
    right_nodes = [node.get("Node Type") for node in walk_plan(right)]
    left_relations = [
        (node.get("Relation Name"), node.get("Alias"))
        for node in walk_plan(left)
        if node.get("Relation Name")
    ]
    right_relations = [
        (node.get("Relation Name"), node.get("Alias"))
        for node in walk_plan(right)
        if node.get("Relation Name")
    ]
    relation_key = lambda item: (item[0] or "", item[1] or "")
    if left_relations != right_relations and sorted(
        left_relations, key=relation_key
    ) == sorted(right_relations, key=relation_key):
        return "join_order"
    if left_nodes == right_nodes:
        return "expression_or_property"
    return "physical_shape"


def compact(sequence: list[str]) -> list[str]:
    return [item for index, item in enumerate(sequence) if index == 0 or sequence[index - 1] != item]


def timing_summary(results: list[dict[str, Any]], mode: str, field: str) -> dict[str, Any]:
    values = sorted(
        result["modes"][mode][field]
        for result in results
        if result["modes"][mode][field] is not None
    )
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "total_ms": round(sum(values), 3),
        "mean_ms": round(sum(values) / len(values), 3),
        "p50_ms": values[(50 * len(values) - 1) // 100],
        "p95_ms": values[min(len(values) - 1, (95 * len(values) - 1) // 100)],
        "max_ms": values[-1],
    }


def distribution(values: list[float | int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 6),
        "p50": ordered[(50 * len(ordered) - 1) // 100],
        "p95": ordered[min(len(ordered) - 1, (95 * len(ordered) - 1) // 100)],
        "max": ordered[-1],
    }


def dsl_observability(records: list[dict[str, Any]]) -> dict[str, Any]:
    rules: dict[str, dict[str, Any]] = {}
    registrations: dict[tuple[str, Any], dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    memo = pipeline = None
    for record in records:
        kind = record.get("kind")
        if kind == "memo_summary":
            memo = record
        elif kind == "pipeline_summary":
            pipeline = record
        elif kind == "rule_summary":
            key = str(record.get("rule_hash", record.get("rule_id")))
            # Summaries are cumulative across stages. Keep the last snapshot
            # per registration, then combine aliases of one canonical rule.
            registrations[key, record.get("rule_id")] = record
        elif kind == "rule_candidate":
            key = str(record.get("rule_hash"))
            candidate = candidates.setdefault(key, {"statuses": {}, "timing_us": {}})
            status = str(record.get("status", "unknown"))
            candidate["statuses"][status] = candidate["statuses"].get(status, 0) + 1
            for field in ("match_us", "constraint_us", "instantiate_us"):
                candidate["timing_us"][field] = candidate["timing_us"].get(field, 0) + int(record.get(field, 0))
        elif kind == "rule_candidate_outcome":
            key = str(record.get("rule_hash"))
            outcome = outcomes.setdefault(key, {"statuses": {}, "memo_version_deltas": []})
            status = str(record.get("status", "unknown"))
            outcome["statuses"][status] = outcome["statuses"].get(status, 0) + 1
            before = record.get("memo_version_before")
            after = record.get("memo_version_after")
            if isinstance(before, int) and isinstance(after, int):
                outcome["memo_version_deltas"].append(after - before)
    counter_fields = (
        "binding_attempts", "bound_symbols", "match_us", "constraint_us", "instantiate_us",
        "match_rejected", "constraint_rejected", "instantiate_rejected", "generated_alternatives",
        "duplicate_alternatives", "budget_exhausted", "budget_skipped",
    )
    for (key, _), record in registrations.items():
        if key not in rules:
            rules[key] = dict(record)
            continue
        total = rules[key]
        total["rule_ids"] = sorted(set(total.get("rule_ids", [total.get("rule_id")])) | {record["rule_id"]})
        total.pop("rule_id", None)
        for field in counter_fields:
            if type(total.get(field)) is int and type(record.get(field)) is int:
                total[field] += record[field]
            else:
                total.pop(field, None)  # Missing counters must still fail the trace audit.
    return {
        "memo": memo,
        "pipeline": pipeline,
        "rules": rules,
        "candidates": candidates,
        "candidate_outcomes": outcomes,
    }


def experiment_run_status(result: dict[str, Any], expect_outcome: bool) -> str:
    """Classify completed runs without treating fallback/partial traces as ORCA success."""
    if result.get("server_failure"):
        return "server_failure"
    error = result.get("error", "").lower()
    timed_out = "timeout" in error
    for stage in ("plan", "rows"):
        rc = result.get(f"{stage}_rc")
        if rc is None:
            return "unknown"
        if rc != 0:
            return f"{stage}_timeout" if rc == 124 or timed_out else f"{stage}_error"
    if result.get("fallback") or result.get("optimizer") == "postgres":
        return "fallback"
    if result.get("optimizer") != "pg_orca":
        return "unknown_optimizer"
    if expect_outcome:
        outcomes = result.get("experiment_outcomes", [])
        if len(outcomes) != 1 or not isinstance(outcomes[0].get("stats_targets"), list):
            return "incomplete_trace"
        for target in outcomes[0]["stats_targets"]:
            if any(target.get(key) is None for key in (
                "selector_kind", "selector", "operator", "fingerprint", "requested_rows", "consumed",
            )):
                return "incomplete_trace"
            if target["consumed"] is not True:
                return "injection_unconsumed"
    return "ok"


@lru_cache(maxsize=256)
def result_semantics(query: str) -> str:
    """Only opt into bag comparison when the SQL parser establishes no outer order."""
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return "ordered"
    try:
        parsed = sqlglot.parse(query, read="postgres")
        if len(parsed) == 1 and isinstance(parsed[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            return "ordered" if parsed[0].args.get("order") is not None else "bag"
    except (ValueError, sqlglot.errors.ParseError, sqlglot.errors.TokenError):
        pass
    return "ordered"  # Legacy/unparsed statements retain conservative byte equality.


def result_hash(run: dict) -> str | None:
    return run.get("rows_bag_hash" if run.get("result_semantics") == "bag" else "rows_hash")


def results_equal(reference: dict, candidate: dict) -> bool:
    return (reference.get("rows_rc") == candidate.get("rows_rc") == 0
            and reference.get("result_semantics", "ordered") == candidate.get("result_semantics", "ordered")
            and result_hash(reference) is not None and result_hash(reference) == result_hash(candidate))


def profile_comparability(
    reference: dict[str, Any], candidate: dict[str, Any], expect_outcome: bool,
    reference_label: str = "off",
) -> dict[str, Any]:
    # Necessary result checks, not a proof of equal statistics interventions.
    reasons = [
        f"{arm}:{status}"
        for arm, result in ((reference_label, reference), ("arm", candidate))
        if (status := experiment_run_status(result, expect_outcome)) != "ok"
    ]
    if not reasons:
        if result_hash(reference) is None or result_hash(candidate) is None:
            reasons.append("missing_rows_hash")
        elif not results_equal(reference, candidate):
            reasons.append("rows_mismatch")
        if expect_outcome:
            # A matching config alone does not establish the same resolved boundary.
            fields = ("selector_kind", "selector", "operator", "fingerprint", "requested_rows")
            signatures = [
                sorted(tuple(target[key] for key in fields)
                       for target in result["experiment_outcomes"][0]["stats_targets"])
                for result in (reference, candidate)
            ]
            if signatures[0] != signatures[1]:
                reasons.append("injection_target_mismatch")
    return {"result_comparable": not reasons, "exclusions": reasons}


def load_input_stats_reference(path: Path, workload: str, query: Path, schema: Path, setup: Path) -> dict:
    """Freeze an independent native estimate source before starting the new database/query."""
    raw = path.read_bytes()
    source = json.loads(raw)
    fixture = json.loads((path.parents[2] / "summary.json").read_text()).get("fixture") or {}
    crc = lambda data: f"{zlib.crc32(data):08x}"
    if (source.get("workload") != workload or fixture.get("workload") != workload
            or source.get("query_crc32") != crc(query.read_bytes())
            or fixture.get("schema_crc32") != crc(schema.read_bytes())
            or (fixture.get("setup") or {}).get("crc32") != crc(setup.read_bytes())):
        raise ValueError("input statistics reference SQL/schema/setup identity mismatch")
    candidates = []
    for experiment in source.get("stats_experiments", []):
        run = experiment.get("modes", {}).get("native", {})
        outcomes = run.get("experiment_outcomes", [])
        events = run.get("stats_events", [])
        if (experiment_run_status(run, True) == "ok" and outcomes[0]["stats_targets"] == []
                and run.get("rows_hash") and any(e.get("kind") == "stats_observation" for e in events)
                and not any(e.get("kind") == "stats_injection" for e in events)):
            candidates.append(run)
    if len(candidates) != 1:
        raise ValueError("require exactly one successful native observation-only reference")
    run = candidates[0]
    experiment = run["experiment_outcomes"][0].get("experiment")
    observed = defaultdict(set)
    for event in run["stats_events"]:
        if event.get("kind") != "stats_observation":
            continue
        key, operator, rows = (event.get(field) for field in ("fingerprint", "operator", "native_rows"))
        if (not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{16}", key)
                or not isinstance(operator, str) or not re.fullmatch(r"CLogical[A-Za-z0-9]+", operator)
                or type(rows) not in (int, float) or not math.isfinite(rows) or rows < 0
                or experiment is None or event.get("experiment") != experiment):
            raise ValueError("invalid native statistics reference observation")
        observed[key, operator].add(rows)
    return {"source": str(path.resolve()), "source_crc32": crc(raw), "fixture": fixture,
            "workload": workload, "query_crc32": source["query_crc32"], "rows_hash": run["rows_hash"],
            "rows_bag_hash": run.get("rows_bag_hash"), "result_semantics": run.get("result_semantics", "ordered"),
            "kind": "frozen_native_estimate", "loaded_before_query": True, "optimizer_consumed": False,
            "attachment": "post_run_analysis",
            "targets": [{"reference_key": key, "operator": operator,
                         "status": "available" if len(rows) == 1 else "ambiguous",
                         "rows": next(iter(rows)) if len(rows) == 1 else None}
                        for (key, operator), rows in sorted(observed.items())]}


def annotate_input_stats_reference(result: dict, reference: dict) -> None:
    lookup = {(t["reference_key"], t["operator"]): t for t in reference["targets"]}
    result["input_stats_reference"] = reference
    for run in all_mode_results(result):
        same_result = results_equal({**reference, "rows_rc": 0}, run)
        for observation in [*run.get("rule_input_observations", []), *run.get("candidate_events", [])]:
            context = observation.get("input_context")
            if context is None:
                continue
            for node in [context["root"], *(child["node"] for child in context["children"])]:
                target = lookup.get((node.get("reference_key"), node["operator"])) if same_result else None
                node["external_reference"] = {
                    "status": target["status"] if target else "unmatched" if same_result else "result_not_verified",
                    "rows": target["rows"] if target else None,
                }


def inventory(
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, set[str]], set[str], set[str]]:
    audit_dir = args.output / "audit"
    result = run(
        [str(Path(args.audit_bin).resolve()), str((TEST_ASSETS / "rules").resolve()), str(audit_dir.resolve())],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(result.stdout)
    runtime = json.loads((audit_dir / "coverage.json").read_text(encoding="utf-8"))
    graph = json.loads((audit_dir / 'rule_graph.json').read_text())
    merged = merge_inventory(runtime, read_matrices(TEST_ASSETS / "e2e/expect"), graph)
    semantic = sorted(entry["name"] for entry in merged["xforms"] if entry["category"] == "semantic_rewrite")
    semantic_set = set(semantic)
    joins = {
        entry["name"]
        for entry in merged["xforms"]
        if entry["category"] == "join_enumeration"
    }
    hashes: dict[str, set[str]] = {}
    for entry in merged["xforms"]:
        for evidence in entry["evidence"]:
            for rule_hash in evidence["dsl_rule_hashes"]:
                hashes.setdefault(rule_hash, set()).add(entry["name"])
    return semantic, hashes, semantic_set, joins


def copy_result_summary(output: str) -> dict[str, Any]:
    """Diagnostic multiset hash, not an ORDER BY/LIMIT equivalence verdict."""
    lines = io.StringIO(output, newline="").readlines()
    reader = csv.reader(lines, strict=True)
    records, start = [], 0
    for _ in reader:
        # Retain raw quoting: COPY CSV NULL and empty text decode to the same
        # Python string, but are different SQL values. Embedded newlines stay
        # inside their record; duplicate records retain their multiplicity.
        records.append("".join(lines[start:reader.line_num]))
        start = reader.line_num
    # ponytail: sort buffered records; use external sorting if results outgrow memory.
    return {"rows_count": len(records),
            "rows_bag_hash": hashlib.sha256("".join(sorted(records)).encode()).hexdigest()}


def collect_postgres_oracle(args, binary, socket, database, query, artifact, modes):
    stdout, stderr, rc, elapsed = psql(binary, socket, args.port, database,
        "LOAD 'pg_orca';\nSET pg_orca.enable_orca=off;\n"
        f"COPY ({query}) TO STDOUT WITH (FORMAT csv);", args.timeout)
    (artifact / 'postgres.rows.csv').write_text(stdout)
    (artifact / 'postgres.rows.stderr').write_text(stderr)
    bag = (copy_result_summary(stdout) if rc == 0 else {'rows_count': None, 'rows_bag_hash': None})
    equal = {name: rc == 0 and mode['rows_rc'] == 0 and
             bag['rows_bag_hash'] == mode.get('rows_bag_hash') for name, mode in modes.items()}
    oracle = {'rows_rc': rc, 'rows_hash': hashlib.sha256(stdout.encode()).hexdigest(),
              'result_semantics': result_semantics(query), **bag}
    comparable = {name: results_equal(oracle, {**mode, 'result_semantics': oracle['result_semantics']})
                  for name, mode in modes.items()}
    return {**oracle, 'rows_ms': elapsed, 'mode_bag_equal': equal, 'mode_result_equal': comparable,
            'valid': bool(comparable) and all(comparable.values()), 'error': error_summary(stderr)}


def run_mode(
    args: argparse.Namespace,
    psql_bin: Path,
    socket: Path,
    database: str,
    query: str,
    artifact: Path,
    artifact_name: str,
    mode: str,
    semantic_xforms: list[str],
    policy_file: Path | None,
    stats_experiment: Path | None = None,
) -> dict[str, Any]:
    base = settings(mode, semantic_xforms, policy_file, stats_experiment)
    # Experimental failures are observations, not disposable retry attempts.
    retry_failures = not (stats_experiment or getattr(args, "stats_experiment", None)
                         or getattr(args, "profile_rule", None) or getattr(args, 'comparison_policies', None))
    plan_out, plan_err, plan_rc, plan_ms = psql(
        psql_bin, socket, args.port, database,
        trace_settings(mode, semantic_xforms, policy_file, stats_experiment)
        + f"\nEXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, FORMAT JSON) {query};",
        args.timeout,
        retry_on_server_failure=retry_failures,
    )
    if plan_rc == 0:
        rows_out, rows_err, rows_rc, rows_ms = psql(
            psql_bin, socket, args.port, database,
            base + f"\nSET client_min_messages=warning;\nCOPY ({query}) TO STDOUT WITH (FORMAT csv);",
            args.timeout,
            retry_on_server_failure=retry_failures,
        )
    else:
        rows_out, rows_err, rows_rc, rows_ms = "", "", plan_rc, 0.0
    (artifact / f"{artifact_name}.plan.json").write_text(plan_out, encoding="utf-8")
    (artifact / f"{artifact_name}.trace").write_text(plan_err, encoding="utf-8")
    # Preserve COPY output and errors, including partial failures. The recorded
    # rows_rc/rows_skipped distinguish completed results from incomplete output.
    (artifact / f"{artifact_name}.rows.csv").write_text(rows_out, encoding="utf-8")
    (artifact / f"{artifact_name}.rows.stderr").write_text(rows_err, encoding="utf-8")
    records = trace_records(plan_err)
    optimizer = optimizer_name(plan_out)
    planning_ms, execution_ms = explain_times(plan_out)
    return {
        "plan_rc": plan_rc,
        "rows_rc": rows_rc,
        "rows_skipped": plan_rc != 0,
        "server_failure": server_failure(plan_err + rows_err),
        "plan_ms": round(plan_ms, 3),
        "rows_ms": round(rows_ms, 3),
        "optimization_ms": optimization_time(plan_err),
        "planning_ms": planning_ms,
        "execution_ms": execution_ms,
        "rows_hash": hashlib.sha256(rows_out.encode()).hexdigest(),
        "result_semantics": result_semantics(query),
        **(copy_result_summary(rows_out) if plan_rc == rows_rc == 0
           else {"rows_count": None, "rows_bag_hash": None}),
        "error": "\n".join(
            summary
            for summary in (
                error_summary(plan_err),
                error_summary(rows_err) if rows_rc else "",
            )
            if summary
        ),
        "optimizer": optimizer,
        "fallback": plan_rc == 0 and optimizer == "postgres",
        "timeout_ms": args.timeout * 1000,
        "timing_source": "traced_explain_analyze",
        "plan": plan_tree(plan_out),
        "produced_xforms": produced_xforms(plan_err),
        "applied_rule_hashes": [
            record["rule_hash"] for record in records
            if record.get("kind") == "application"
            and record.get("status") in {"applied", "applied_rbo"}
            and "rule_hash" in record
        ],
        "profile_rule_statuses": [
            record.get("status") for record in records
            if record.get("kind") == "application"
            and record.get("rule_hash") == getattr(args, "profile_rule", None)
        ],
        "rule_input_observations": [
            {key: record.get(key) for key in (
                "rule_hash", "placement", "status", "input_sampling", "input_context")}
            for record in records
            if record.get("kind") == "application" and "input_context" in record
        ],
        "stats_events": [
            record for record in records
            if record.get("kind") in {"stats_injection", "stats_observation"}
        ],
        "candidate_events": [record for record in records
                             if record.get("kind") in {"rule_candidate", "rule_candidate_outcome"}],
        "cost_events": [record for record in records if record.get("kind") == "cost_candidate"],
        "stats_lifecycle_events": [record for record in records if record.get("kind") == "group_stats_lifecycle"],
        "cost_lifecycle_events": [record for record in records if record.get("kind") == "cost_lifecycle"],
        "search_checks": [record for record in records if record.get("kind") == "search_check"],
        "rule_edges": [record for record in records if record.get("kind") == "rule_edge"],
        "experiment_outcomes": [
            record for record in records
            if record.get("kind") == "experiment_outcome"
        ],
        "dsl_observability": dsl_observability(records),
        "dphyper_events": parse_dphyper_events(plan_err),
        "native_memo_origins": sorted({
            str(record.get("origin")) for record in records
            if record.get("kind") == "memo_alternative" and record.get("source") == "native"
        }),
    }


def timing_plan(plan: Any) -> Any:
    """Remove EXPLAIN ANALYZE measurements, retaining plan fields (raw JSON is also saved)."""
    if isinstance(plan, list):
        return [timing_plan(child) for child in plan]
    if not isinstance(plan, dict):
        return plan
    return {key: timing_plan(value) for key, value in plan.items() if not (
        key.startswith(("Actual ", "Rows Removed ", "Shared ", "Local ", "Temp ", "I/O "))
        or key in {"Sort Method", "Sort Space Used", "Sort Space Type", "Heap Fetches",
                   "Hash Buckets", "Original Hash Buckets", "Hash Batches", "Original Hash Batches",
                   "HashAgg Batches", "Peak Memory Usage", "Disk Usage", "Workers Launched"}
    )}


def timing_schedule(scenarios: int, repeats: int, warmups: int, seed: int,
                    arms=("off", "rbo", "cbo")):
    if (len(arms) < 2 or len(set(arms)) != len(arms)
            or any(not isinstance(arm, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', arm) for arm in arms)):
        raise ValueError("timing requires at least two distinct safe arm names")
    rng = random.Random(seed)
    for phase, rounds in (("warmup", warmups), ("measurement", repeats)):
        for block in range(rounds):
            cells = [(index, arm) for index in range(scenarios) for arm in arms]
            rng.shuffle(cells)
            for index, arm in cells:
                yield phase, block, index, arm


def run_profile_timing(args: argparse.Namespace, binary: Path, socket: Path, database: str,
                       query: str, artifact: Path, semantic_xforms: list[str], scenarios: list[dict]) -> dict:
    samples = []
    baseline = next(iter(args.profile_policies))
    # New backends per sample: warm shared buffers, not a persistent-session cache.
    with (artifact / "timing-samples.jsonl").open("w", encoding="utf-8") as log:
        for sequence, (phase, block, index, arm) in enumerate(timing_schedule(
                len(scenarios), args.timing_repeats, args.timing_warmups, args.timing_seed,
                tuple(args.profile_policies))):
            scenario = scenarios[index]
            stats_path = scenario["stats_experiment"]
            diagnostic = scenario["arms"][arm]
            sql = settings("replacement", semantic_xforms, args.profile_policies[arm],
                           Path(stats_path) if stats_path else None) + """
SET optimizer_print_xform=off;
SET optimizer_print_xform_results=off;
SET optimizer_print_optimization_stats=off;
SET pg_orca.trace_dsl_rule=off;
SET client_min_messages=warning;
""" + f"EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, FORMAT JSON) {query};"
            stdout, stderr, rc, elapsed = psql(binary, socket, args.port, database, sql,
                                               args.timeout, retry_on_server_failure=False)
            stem = f"timing-{sequence:04d}.{arm}"
            (artifact / f"{stem}.plan.json").write_text(stdout, encoding="utf-8")
            (artifact / f"{stem}.stderr").write_text(stderr, encoding="utf-8")
            planning, execution = explain_times(stdout)
            optimizer = optimizer_name(stdout)
            plan = plan_tree(stdout)
            if server_failure(stderr):
                status = "server_failure"
            elif rc:
                status = "plan_timeout" if rc == 124 or "timeout" in stderr.lower() else "plan_error"
            elif optimizer == "postgres":
                status = "fallback"
            elif optimizer != "pg_orca":
                status = "unknown_optimizer"
            elif plan is None or not all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0
                                         for t in (planning, execution)):
                status = "incomplete_timing"
            elif trace_records(stderr) or optimization_time(stderr) is not None:
                status = "trace_enabled"
            else:
                status = "ok"
            comparable = profile_comparability(scenario["arms"][baseline], diagnostic, stats_path is not None, baseline)
            reasons = list(comparable["exclusions"])
            if status != "ok":
                reasons.append(status)
            plan_matches = plan is not None and timing_plan(plan) == timing_plan(diagnostic["plan"])
            if not plan_matches:
                reasons.append("diagnostic_plan_mismatch")
            sample = {
                "sequence": sequence, "phase": phase, "block": block, "scenario": index, "arm": arm,
                "stats_experiment": stats_path, "artifact": stem, "status": status, "returncode": rc,
                "timeout_ms": args.timeout * 1000, "optimizer": optimizer,
                "planning_ms": planning, "execution_ms": execution, "client_wall_ms": elapsed,
                "diagnostic_plan_matches": plan_matches, "comparison_exclusions": reasons,
                "diagnostic_rows_hash": diagnostic.get("rows_hash"),
                "diagnostic_rows_count": diagnostic.get("rows_count"),
                "error": stderr if status != "ok" else "",
            }
            samples.append(sample)
            log.write(json.dumps(sample, sort_keys=True) + "\n")
            log.flush()  # Keep every attempt, including failed warmups; never retry timed samples.
    return {"source": "untraced_explain_analyze", "seed": args.timing_seed,
            "arms": list(args.profile_policies),
            "repeats": args.timing_repeats, "warmups": args.timing_warmups,
            "design": "randomized_complete_blocks", "connection": "new_backend_per_sample",
            "result_validation": "separate_diagnostic_run", "samples": samples}


def compare_query(
    args: argparse.Namespace,
    psql_bin: Path,
    socket: Path,
    database: str,
    workload: str,
    query_path: Path,
    semantic_xforms: list[str],
    hash_xforms: dict[str, set[str]],
    semantic_xform_set: set[str],
    join_xforms: set[str],
) -> dict[str, Any]:
    query_bytes = query_path.read_bytes()
    if getattr(args, 'dro_receipt', None) is not None and artifact_snapshot(args.dro_input_paths) != args.dro_input_start:
        raise ValueError('DRO input files changed after contract binding and before query execution')
    reference = getattr(args, "input_stats_reference_snapshot", None)
    if reference and (reference["workload"] != workload
                      or reference["query_crc32"] != f"{zlib.crc32(query_bytes):08x}"):
        raise ValueError("preloaded input statistics reference query mismatch")
    query = query_bytes.decode("utf-8").strip().removesuffix(";")
    artifact = args.output / workload / query_path.stem
    artifact.mkdir(parents=True, exist_ok=True)
    modes: dict[str, dict[str, Any]] = {}
    for mode in ("native", "replacement"):
        modes[mode] = run_mode(
            args, psql_bin, socket, database, query, artifact, mode, mode,
            semantic_xforms, args.policy_file,
        )

    stats_experiments = []
    for index, path in enumerate(args.stats_experiment, 1):
        experiment_modes = {
            mode: run_mode(
                args, psql_bin, socket, database, query, artifact,
                f"stats-{index}.{mode}", mode, semantic_xforms,
                args.policy_file, path,
            )
            for mode in ("native", "replacement")
        }
        stats_experiments.append({
            "path": str(path.resolve()),
            "modes": experiment_modes,
            "comparisons": {
                mode: {
                    "outcome_equal": all(
                        experiment_modes[mode][key] == modes[mode][key]
                        for key in ("plan_rc", "rows_rc")
                    ),
                    "rows_equal": results_equal(modes[mode], experiment_modes[mode]),
                    "plan_comparison": plan_difference(
                        modes[mode]["plan"], experiment_modes[mode]["plan"]
                    ),
                }
                for mode in ("native", "replacement")
            },
        })

    rule_profile = None
    if args.profile_rule or getattr(args, 'comparison_policies', None):
        baseline = next(iter(args.profile_policies))
        scenarios = []
        for index, stats_path in enumerate([None, *args.stats_experiment]):
            arms = {
                arm: run_mode(
                    args, psql_bin, socket, database, query, artifact,
                    f"profile-{index}.{arm}", "replacement", semantic_xforms,
                    args.profile_policies[arm], stats_path,
                )
                for arm in args.profile_policies
            }
            scenarios.append({
                "stats_experiment": (
                    str(stats_path.resolve()) if stats_path is not None else None
                ),
                "arms": arms,
                "comparisons": {
                    arm: {
                        **profile_comparability(arms[baseline], arms[arm], stats_path is not None, baseline),
                        "outcome_equal": all(
                            arms[arm][key] == arms[baseline][key]
                            for key in ("plan_rc", "rows_rc")
                        ),
                        "rows_equal": results_equal(arms[baseline], arms[arm]),
                        "plan_comparison": plan_difference(
                            arms[baseline]["plan"], arms[arm]["plan"]
                        ),
                        **({"target_statuses": arms[arm]["profile_rule_statuses"]} if args.profile_rule else {}),
                    }
                    for arm in arms if arm != baseline
                },
            })
        rule_profile = {
            "rule_hash": args.profile_rule,
            "arms": list(args.profile_policies),
            "placement_baseline": "cbo",
            "background_policy": "engine_defaults_not_policy_file",
            "policy_snapshots": {arm: path.read_text(encoding="utf-8")
                                 for arm, path in args.profile_policies.items()},
            "rbo_phase": args.profile_rbo_phase if "rbo" in args.profile_policies else None,
            "effect": args.profile_effect,
            "scenarios": scenarios,
        }
        if getattr(args, "profile_companion_rule", []):
            rule_profile["intervention_rule_hashes"] = profile_targets(args)
            rule_profile["intervention_scope"] = "joint_bundle_not_individual_rule_effect"
        if args.timing_repeats:
            rule_profile["timing"] = run_profile_timing(
                args, psql_bin, socket, database, query, artifact, semantic_xforms, scenarios)
        if getattr(args, 'comparison_policies', None):
            rule_profile = {key: value for key, value in rule_profile.items()
                            if key in {'arms', 'policy_snapshots', 'scenarios', 'timing'}}
            rule_profile.update(scope='complete_policy_not_individual_rule_effect', baseline=baseline)

    produced = modes["native"]["produced_xforms"]
    native_sequence = [name for name in produced if name in semantic_xform_set]
    native_join_sequence = [name for name in produced if name in join_xforms]
    applied_hashes = modes["replacement"]["applied_rule_hashes"]
    mapped_sequence = [
        "|".join(sorted(hash_xforms[rule_hash]))
        for rule_hash in applied_hashes if rule_hash in hash_xforms
    ]
    native_set = set(native_sequence)
    mapped_set = {name for rule_hash in applied_hashes for name in hash_xforms.get(rule_hash, ())}
    replacement_origins = set(modes["replacement"]["native_memo_origins"])
    forbidden_origins = sorted(
        replacement_origins & (set(semantic_xforms) | join_xforms)
    )
    plan_kind = plan_difference(modes["native"]["plan"], modes["replacement"]["plan"])
    outcome_equal = all(
        modes["replacement"][key] == modes["native"][key]
        for key in ("plan_rc", "rows_rc")
    )
    rows_equal = results_equal(modes["native"], modes["replacement"])
    native_orca = modes["native"]["optimizer"] == "pg_orca"
    replacement_orca = modes["replacement"]["optimizer"] == "pg_orca"
    if native_orca and replacement_orca:
        coverage_status = "comparable"
    elif native_orca:
        coverage_status = "replacement_failed"
    elif replacement_orca:
        coverage_status = "replacement_only"
    else:
        coverage_status = "native_unsupported"
    trigger_set_equal = native_set == mapped_set
    trigger_order_equal = compact(native_sequence) == compact(mapped_sequence)
    dphyper_applied = any(
        event.get("status") == "applied"
        for event in modes["replacement"]["dphyper_events"]
    )
    join_enumeration_replaced = not (replacement_origins & join_xforms)
    explanations = []
    if not trigger_set_equal:
        explanations.append("different_semantic_search_path")
    elif not trigger_order_equal:
        explanations.append("atomic_DSL_rules_change_Cascades_order")
    if dphyper_applied:
        explanations.append("DPHyper_join_enumeration")
    elif native_join_sequence:
        explanations.append("join_enumeration_bypassed_on_replacement_path")
    if plan_kind != "identical":
        explanations.append(f"final_plan_{plan_kind}")
    if forbidden_origins:
        explanations.append("disabled_native_semantic_origin_leaked")
    result = {
        "workload": workload,
        "query": query_path.name,
        "query_crc32": f"{zlib.crc32(query_bytes):08x}",
        "outcome_equal": outcome_equal,
        "rows_equal": rows_equal,
        # Supplemental diagnosis: ordered queries still require ordered equality.
        "rows_bag_equal": (
            modes["native"].get("rows_bag_hash") == modes["replacement"].get("rows_bag_hash")
            if all(m.get("rows_bag_hash") is not None for m in modes.values()) else None
        ),
        "coverage_status": coverage_status,
        "trigger_set_equal": trigger_set_equal,
        "trigger_order_equal": trigger_order_equal,
        "join_enumeration_replaced": join_enumeration_replaced,
        "dphyper_applied": dphyper_applied,
        "plan_comparison": plan_kind,
        "native_trigger_sequence": native_sequence,
        "native_join_trigger_sequence": native_join_sequence,
        "replacement_rule_sequence": applied_hashes,
        "replacement_mapped_sequence": mapped_sequence,
        "missing_native_triggers": sorted(native_set - mapped_set),
        "extra_mapped_triggers": sorted(mapped_set - native_set),
        "forbidden_native_origins": forbidden_origins,
        "explanations": explanations,
        "modes": modes,
        "stats_experiments": stats_experiments,
        "rule_profile": rule_profile,
    }
    if getattr(args, 'comparison_policies', None):
        result['policy_comparison'] = result['rule_profile']
        result['rule_profile'] = None
    if reference:
        annotate_input_stats_reference(result, reference)
    if getattr(args, 'capture_pre_context', False):
        result['pre_workload_context'] = {
            'snapshot': args.pre_context_path,
            'input_endpoints_equal': artifact_snapshot(args.pre_context_inputs) == args.pre_context_start,
            'scope': 'file_endpoints_not_catalog_lock_or_resolved_policy_validation'}
    if getattr(args, 'feature_graph_snapshot', None) is not None:
        result['feature_graph'] = args.feature_graph_snapshot
    if getattr(args, 'postgres_oracle', False):
        oracle_modes = dict(modes)
        if result.get('policy_comparison'):
            oracle_modes.update({f'policy:{i}:{name}': mode
                for i, scenario in enumerate(rule_profile['scenarios']) for name, mode in scenario['arms'].items()})
        result['postgres_oracle'] = collect_postgres_oracle(
            args, psql_bin, socket, database, query, artifact, oracle_modes)
    if getattr(args, "artifact_start", None) is not None:
        end = artifact_snapshot(args.artifact_paths)
        result["artifact_provenance"] = {
            "before_server_start": args.artifact_start, "after_query": end,
            "endpoints_equal": end == args.artifact_start,
            "scope": "file_contents_at_endpoints_not_continuous_install_monitoring",
        }
    if getattr(args, 'dro_receipt', None) is not None:
        result['dro_collection'] = {**args.dro_receipt,
            'input_endpoints_equal': artifact_snapshot(args.dro_input_paths) == args.dro_input_start}
    for mode in [*modes.values(), *(
        mode
        for experiment in stats_experiments
        for mode in experiment["modes"].values()
    ), *(
        mode
        for scenario in ([] if rule_profile is None else rule_profile["scenarios"])
        for mode in scenario["arms"].values()
    )]:
        mode.pop("plan")
    (artifact / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def selected(workload: str, query: Path, patterns: list[str]) -> bool:
    key = f"{workload}/{query.stem}"
    return not patterns or any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns)


def select_workload_queries(root: Path, workloads: list[str], patterns: list[str]) -> dict:
    queries = {workload: [q for q in sorted((root / workload / "sql").glob("*.sql"))
                         if selected(workload, q, patterns)] for workload in workloads}
    if not any(queries.values()):
        raise ValueError(f"no queries selected: {patterns}; -t expects workload/query_stem")
    return queries


def policy_experiment(result: dict) -> dict:
    return result.get('rule_profile') or result.get('policy_comparison') or {}


def all_mode_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *result["modes"].values(),
        *(
            mode
            for experiment in result["stats_experiments"]
            for mode in experiment["modes"].values()
        ),
        *(
            mode
            for scenario in (
                policy_experiment(result).get('scenarios', [])
            )
            for mode in scenario["arms"].values()
        ),
    ]


def experiment_observations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations = []
    for result in results:
        runs = []
        for experiment in result["stats_experiments"]:
            for mode, run_result in experiment["modes"].items():
                runs.append((experiment["path"], mode, None, None, run_result, None))
        profile = policy_experiment(result)
        for scenario in profile["scenarios"] if profile else []:
            for arm, run_result in scenario["arms"].items():
                runs.append((scenario["stats_experiment"], "replacement", arm,
                    profile.get("rule_hash"), run_result, profile_comparability(
                        scenario["arms"][profile.get('baseline', 'off')], run_result,
                        scenario["stats_experiment"] is not None, profile.get('baseline', 'off'))))
        for path, mode, arm, rule, run_result, comparison in runs:
            outcomes = run_result["experiment_outcomes"]
            outcome = outcomes[0] if len(outcomes) == 1 else {}
            status = experiment_run_status(run_result, path is not None)
            observations.append({
                "workload": result["workload"],
                "query": result["query"],
                "experiment_path": path,
                "experiment": outcome.get("experiment"),
                "mode": mode,
                "arm": arm,
                "profile_rule_hash": rule,
                **({'policy_scope': profile['scope']} if arm is not None and 'scope' in profile else {}),
                **({"intervention_rule_hashes": profile["intervention_rule_hashes"]}
                   if arm is not None and profile.get("intervention_rule_hashes") else {}),
                "status": status,
                "complete": status == "ok",
                "comparison": comparison,
                "execution": {field: run_result.get(field) for field in (
                    "plan_rc", "rows_rc", "rows_skipped", "optimizer", "fallback",
                    "server_failure", "error", "rows_hash", "timeout_ms",
                    "plan_ms", "rows_ms", "timing_source",
                )},
                "stats": run_result["stats_events"],
                "rule_inputs": run_result.get("rule_input_observations", []),
                "candidate_events": run_result.get("candidate_events", []),
                "input_stats_reference": result.get("input_stats_reference"),
                "search": {
                    **(run_result["dsl_observability"]["memo"] or {}),
                    **(run_result["dsl_observability"]["pipeline"] or {}),
                    **outcome,
                    **{field: run_result[field] for field in (
                        "optimization_ms", "planning_ms", "execution_ms",
                    )},
                },
                "rules": run_result["dsl_observability"]["rules"],
                "candidates": run_result["dsl_observability"]["candidates"],
                "candidate_outcomes": run_result["dsl_observability"]["candidate_outcomes"],
            })
    return observations


def experiment_distribution_summary(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[(observation["experiment_path"], observation["mode"],
                 observation["arm"], observation["profile_rule_hash"],
                 tuple(sorted(observation.get("intervention_rule_hashes", []))))].append(observation)

    summaries = []
    for (path, mode, arm, profile_rule, bundle), all_runs in sorted(grouped.items(), key=lambda item: str(item[0])):
        ids = {run["experiment"] for run in all_runs if run["experiment"] is not None}
        experiment = next(iter(ids)) if len(ids) == 1 else None
        run_statuses: dict[str, int] = defaultdict(int)
        for run in all_runs:
            run_statuses[run["status"]] += 1
        # Partial counters are retained in observations, not zero-filled into
        # successful-run distributions or interpreted as completed search work.
        runs = [run for run in all_runs if run["complete"]]
        search_fields = sorted({
            field
            for run in runs
            for field, value in run["search"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        })
        rule_hashes = sorted({rule for run in runs for rule in run["rules"]})
        trigger_distributions = {}
        for rule in rule_hashes:
            trigger_distributions[rule] = {
                field: distribution([
                    run["rules"].get(rule, {}).get(field, 0) for run in runs
                ])
                for field in RULE_COUNTER_FIELDS
            }
            trigger_distributions[rule]["run_presence"] = sum(
                rule in run["rules"] for run in runs
            )

        candidate_rules = sorted({
            rule
            for run in runs
            for rule in run["candidates"]
        })
        candidate_triggers = {}
        for rule in candidate_rules:
            statuses = sorted({
                status
                for run in runs
                for status in run["candidates"].get(rule, {}).get("statuses", {})
            })
            timings = sorted({
                field
                for run in runs
                for field in run["candidates"].get(rule, {}).get("timing_us", {})
            })
            candidate_triggers[rule] = {
                "run_presence": sum(rule in run["candidates"] for run in runs),
                "statuses": {
                    status: distribution([
                        run["candidates"].get(rule, {}).get("statuses", {}).get(status, 0)
                        for run in runs
                    ])
                    for status in statuses
                },
                "timing_us": {
                    field: distribution([
                        run["candidates"].get(rule, {}).get("timing_us", {}).get(field, 0)
                        for run in runs
                    ])
                    for field in timings
                },
            }

        candidate_effects = {}
        outcome_rules = sorted({
            rule
            for run in runs
            for rule in run["candidate_outcomes"]
        })
        for rule in outcome_rules:
            deltas = [
                delta
                for run in runs
                for delta in run["candidate_outcomes"].get(rule, {}).get(
                    "memo_version_deltas", []
                )
            ]
            statuses: dict[str, int] = defaultdict(int)
            for run in runs:
                for status, count in run["candidate_outcomes"].get(
                    rule, {}
                ).get("statuses", {}).items():
                    statuses[status] += count
            candidate_effects[rule] = {
                "memo_version_delta": distribution(deltas),
                "statuses": dict(sorted(statuses.items())),
            }

        statistics: dict[str, dict[str, list[float]]] = {}
        for run in runs:
            for event in run["stats"]:
                selector = str(
                    event.get("fingerprint")
                    or event.get("relations")
                    or event.get("operator")
                    or "unknown"
                )
                item = statistics.setdefault(
                    selector, {"native_rows": [], "effective_rows": [], "scale": []}
                )
                native = event.get("native_rows")
                effective = event.get("rows", native)
                if isinstance(native, (int, float)):
                    item["native_rows"].append(native)
                if isinstance(effective, (int, float)):
                    item["effective_rows"].append(effective)
                if isinstance(native, (int, float)) and native and isinstance(effective, (int, float)):
                    item["scale"].append(effective / native)

        summaries.append({
            "experiment": experiment,
            "experiment_path": path,
            "mode": mode,
            "arm": arm,
            "profile_rule_hash": profile_rule,
            **({"intervention_rule_hashes": list(bundle)} if bundle else {}),
            "runs": len(all_runs),
            "complete_runs": len(runs),
            "status_counts": dict(sorted(run_statuses.items())),
            "result_comparable_runs": sum(
                bool(run["comparison"] and run["comparison"]["result_comparable"])
                for run in all_runs
            ),
            "statistics": {
                selector: {
                    field: distribution(values) for field, values in fields.items()
                }
                for selector, fields in sorted(statistics.items())
            },
            "search_space": {
                field: distribution([
                    value
                    for run in runs
                    if isinstance((value := run["search"].get(field)), (int, float))
                    and not isinstance(value, bool)
                ])
                for field in search_fields
            },
            "rule_triggers": trigger_distributions,
            "candidate_triggers": candidate_triggers,
            "candidate_effects": candidate_effects,
        })
    curves: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in observations:
        for event in run["stats"]:
            selector = str(
                event.get("fingerprint")
                or event.get("relations")
                or event.get("operator")
                or "unknown"
            )
            native = event.get("native_rows")
            effective = event.get("rows", native)
            point = {
                "workload": run["workload"],
                "query": run["query"],
                "experiment": run["experiment"],
                "experiment_path": run["experiment_path"],
                "status": run["status"],
                "complete": run["complete"],
                "comparison": run["comparison"],
                "intervened": event.get("kind") == "stats_injection",
                # Keep the joint intervention when projecting one axis; this
                # is not an isolated response curve if other targets also vary.
                "intervention_targets": run["search"].get("stats_targets"),
                "site": event.get("site"),
                "scale_basis": "stats_at_application",
                "native_rows": native,
                "effective_rows": effective,
                "scale": (
                    effective / native
                    if isinstance(native, (int, float))
                    and native
                    and isinstance(effective, (int, float))
                    else None
                ),
                "search": {
                    field: run["search"].get(field)
                    for field in CURVE_SEARCH_FIELDS
                    if run["search"].get(field) is not None
                },
                "rule_triggers": {
                    rule: {
                        field: int(counters.get(field, 0))
                        for field in RULE_COUNTER_FIELDS
                        if counters.get(field, 0)
                    }
                    for rule, counters in run["rules"].items()
                    if any(counters.get(field, 0) for field in RULE_COUNTER_FIELDS)
                },
                "candidate_triggers": run["candidates"],
            }
            curves[(selector, run["mode"], run["arm"], run["profile_rule_hash"],
                    tuple(sorted(run.get("intervention_rule_hashes", []))))].append(point)

    return {
        "groups": summaries,
        "factor_curves": [
            {
                "selector": selector,
                "mode": mode,
                "arm": arm,
                "profile_rule_hash": rule,
                **({"intervention_rule_hashes": list(bundle)} if bundle else {}),
                "points": sorted(
                    points,
                    key=lambda point: (
                        point["scale"] is None,
                        point["scale"] or 0,
                        point["workload"],
                        point["query"],
                    ),
                ),
            }
            for (selector, mode, arm, rule, bundle), points in sorted(
                curves.items(), key=lambda item: str(item[0])
            )
        ],
    }


def profile_targets(args: argparse.Namespace) -> list[str]:
    companions = getattr(args, "profile_companion_rule", [])
    if companions and not (args.profile_rule and getattr(args, "profile_cbo_only", False)):
        raise ValueError("companion rules require --profile-rule and --profile-cbo-only")
    hashes = [args.profile_rule, *companions]
    if any(not isinstance(h, str) or re.fullmatch(r"[0-9a-fA-F]{16}", h) is None for h in hashes):
        raise ValueError("profile targets must be 16-digit hexadecimal identities")
    hashes = [h.lower() for h in hashes]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate profile target")
    return [hashes[0], *sorted(hashes[1:])]


def write_profile_policies(root: Path, args: argparse.Namespace) -> dict[str, Path]:
    policies = {}
    for name, source in getattr(args, 'comparison_policies', {}).items():
        path = root / f'profile-{name}.policy'
        path.write_bytes(source.read_bytes())
        policies[name] = path
    if policies:
        return policies  # Freeze external documents before any backend reads them.
    targets = profile_targets(args)
    entries = {
        "off": "  enabled: false\n",
        "rbo": (
            "  enabled: true\n"
            "  placement: rbo\n"
            f"  phase: {args.profile_rbo_phase}\n"
            f"  effect: {args.profile_effect}\n"
            "  priority: 0\n"
            "  order: bottom_up\n"
            "  fixpoint: true\n"
        ),
        "cbo": (
            "  enabled: true\n"
            "  placement: cbo\n"
            "  phase: explore\n"
            f"  effect: {args.profile_effect}\n"
        ),
    }
    for arm, fields in entries.items():
        if arm == "rbo" and getattr(args, "profile_cbo_only", False):
            continue
        path = root / f"profile-{arm}.policy"
        path.write_text("".join(f"- rule: {target}\n{fields}" for target in targets), encoding="utf-8")
        policies[arm] = path
    return policies


def collection_descriptor(args, query_selection, semantic_xforms):
    """Describe actual files/settings; one selected SQL per independently assigned unit."""
    from ml_orca.experiments.calibrate_rule_dro import contract_identity

    selected = [(workload, query) for workload, queries in query_selection.items() for query in queries]
    if len(selected) != 1:
        raise ValueError('DRO collection requires exactly one selected query per unit')
    workload, query = selected[0]
    paths = {'query': query, 'schema': args.workload_dir / workload / 'schema.sql',
             **{f'stats_{i}': path for i, path in enumerate(args.stats_experiment)}}
    paths.update({key: path for key in ('setup_sql', 'policy_file', 'input_stats_reference', 'cardinality_probes')
                  if (path := getattr(args, key, None)) is not None})
    paths.update({f'profile_{arm}': path for arm, path in args.profile_policies.items()})
    args.dro_input_paths = paths
    args.dro_input_start = artifact_snapshot(paths)
    if any('error' in item for item in args.dro_input_start.values()):
        raise ValueError('cannot fingerprint DRO input files')
    content = lambda snapshot: {key: {field: item[field] for field in ('crc32', 'size')}
                                for key, item in snapshot.items()}
    return {'environment_identity': contract_identity(content(args.artifact_start)),
            'policies': {arm: contract_identity({'policy': path.read_text(encoding='utf-8'),
                                                'background': 'engine_defaults_not_policy_file'})
                         for arm, path in args.profile_policies.items()},
            'inputs': content(args.dro_input_start), 'workload': workload,
            'protocol': {**{key: getattr(args, key) for key in ('timeout', 'jobs', 'timing_repeats',
                         'timing_warmups', 'timing_seed', 'profile_rule', 'profile_effect', 'postgres_oracle')},
                         **({'capture_pre_context': True} if getattr(args, 'capture_pre_context', False) else {})},
            'semantic_xforms': semantic_xforms,
            'scope': 'actual_runner_acquisition_descriptor_not_sampling_justification'}


def main() -> int:
    args = parse_args()
    workloads = args.workload or ["tpch", "tpcds", "job", "sqlstorm"]
    query_selection = select_workload_queries(args.workload_dir, workloads, args.test)
    args.output.mkdir(parents=True, exist_ok=True)
    pg_bindir = Path(subprocess.check_output([args.pg_config, "--bindir"], text=True).strip())
    pg_libdir = Path(subprocess.check_output([args.pg_config, "--pkglibdir"], text=True).strip())
    args.artifact_paths = {"postgres": pg_bindir / "postgres", "pg_orca": pg_libdir / "pg_orca.so",
                           "rule_audit": Path(args.audit_bin), "rules": args.rule_file,
                           "runner": Path(__file__)}
    args.feature_graph_snapshot = None
    if args.feature_graph:
        args.feature_graph_snapshot = freeze_feature_graph(args.feature_graph, args.output)
        args.artifact_paths.update(feature_graph=args.output / 'feature-graph.json',
                                   feature_graph_receipt=args.output / 'feature-graph-snapshot.json')
    if args.dro_contract or args.dro_describe:
        args.artifact_paths.update(dro_calibration=source_file('calibrate_rule_dro.py'),
                                   dro_math=source_file('rule_dro.py'))
    args.artifact_start = artifact_snapshot(args.artifact_paths)
    if any("error" in item for item in args.artifact_start.values()):
        raise ValueError(f"cannot fingerprint experiment artifacts: {args.artifact_start}")
    semantic_xforms, hash_xforms, semantic_xform_set, join_xforms = inventory(args)
    args.input_stats_reference_snapshot = None
    if args.input_stats_reference:
        workload = workloads[0]
        queries = query_selection[workload]
        if len(queries) != 1:
            raise ValueError("input statistics reference requires exactly one selected query")
        if args.input_stats_reference.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("reference must be outside the new output directory")
        args.input_stats_reference_snapshot = load_input_stats_reference(
            args.input_stats_reference, workload, queries[0],
            args.workload_dir / workload / "schema.sql", args.setup_sql)
    with tempfile.TemporaryDirectory(prefix="pgorca-workloads.", dir="/tmp") as temporary:
        root = Path(temporary)
        data = root / "data"
        socket = root / "socket"
        socket.mkdir()
        args.profile_policies = (
            write_profile_policies(root, args) if args.profile_rule or args.comparison_policies else {}
        )
        if args.dro_contract or args.dro_describe:
            from ml_orca.experiments.calibrate_rule_dro import bind_collection
            descriptor = collection_descriptor(args, query_selection, semantic_xforms)
            if args.dro_describe:
                with (args.output / 'dro-acquisition.json').open('x') as stream:
                    stream.write(json.dumps(descriptor, indent=2) + '\n')
                return 0
            contract = json.loads(args.dro_contract.read_text())
            args.dro_receipt = bind_collection(contract, args.dro_unit, descriptor)
            with (args.output / 'dro-collection.json').open('x') as stream:
                stream.write(json.dumps({'contract': contract, 'binding': args.dro_receipt}, indent=2) + '\n')
        init = run([str(pg_bindir / "initdb"), "-D", str(data), "--no-locale", "--encoding=UTF8"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if init.returncode:
            raise RuntimeError(init.stdout)
        env = {**os.environ, "MONSOON_DSL_RULES": str(args.rule_file.resolve())}
        log = args.output / "postgresql.log"
        start = run([
            str(pg_bindir / "pg_ctl"), "-D", str(data), "-l", str(log),
            "-o", f"-c listen_addresses='' -c logging_collector=off -k {socket} -p {args.port}", "start",
        ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if start.returncode:
            raise RuntimeError(start.stdout)
        results = []
        fixture_context = {}
        try:
            for workload in workloads:
                queries = query_selection[workload]
                if not queries:
                    continue
                database = f"dsl_{workload}"
                created = run([str(pg_bindir / "createdb"), "-h", str(socket), "-p", str(args.port), database], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if created.returncode:
                    raise RuntimeError(created.stdout)
                schema = args.workload_dir / workload / "schema.sql"
                loaded = run([
                    str(pg_bindir / "psql"), "-X", "-v", "ON_ERROR_STOP=1",
                    "-h", str(socket), "-p", str(args.port), "-d", database,
                    "-c", "CREATE EXTENSION pg_orca;", "-f", str(schema),
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if loaded.returncode:
                    raise RuntimeError(f"{workload} schema failed:\n{loaded.stdout}")
                fixture_context = {"workload": workload,
                    "schema_crc32": f"{zlib.crc32(schema.read_bytes()):08x}", "setup": None}
                if args.setup_sql:
                    setup = args.setup_sql.read_text(encoding="utf-8")
                    _, error, rc, _ = psql(pg_bindir / "psql", socket, args.port, database,
                                          "SET pg_orca.enable_orca=off;\n" + setup, args.timeout)
                    if rc:
                        raise RuntimeError(f"{workload} data setup failed:\n{error}")
                    fixture_context["setup"] = {"path": str(args.setup_sql.resolve()),
                        "crc32": f"{zlib.crc32(setup.encode()):08x}"}
                reference = args.input_stats_reference_snapshot
                if reference and (
                        fixture_context["schema_crc32"] != reference["fixture"]["schema_crc32"]
                        or fixture_context["setup"]["crc32"] != reference["fixture"]["setup"]["crc32"]):
                    raise ValueError("fixture changed since input statistics reference was loaded")
                if args.capture_pre_context:
                    args.pre_context_inputs = {
                        'schema': schema, 'rules': args.rule_file,
                        **{f'query:{q.name}': q for q in queries},
                        **{f'profile:{arm}': path for arm, path in args.profile_policies.items()},
                        **{f'stats:{i}': path for i, path in enumerate(args.stats_experiment)}}
                    args.pre_context_inputs.update({key: path for key in ('policy_file', 'setup_sql')
                                                   if (path := getattr(args, key)) is not None})
                    args.pre_context_start = artifact_snapshot(args.pre_context_inputs)
                    policies = {'replacement': args.policy_file, **args.profile_policies}
                    policy_text = {name: path.read_text() if path is not None else None
                                   for name, path in policies.items()}
                    context = collect_catalog_context(pg_bindir / 'psql', socket, args.port, database, args.timeout)
                    context.update(fixture=fixture_context, input_files=args.pre_context_start,
                        policy_documents=policy_text,
                        policy_resolution='native_loader_snapshot_in_audit_binary_not_live_backend',
                        resolved_policies=collect_policy_context(Path(args.audit_bin), args.rule_file, policies, args.timeout),
                        settings_sql={'native': settings('native', semantic_xforms, None),
                            **{name: settings('replacement', semantic_xforms, path) for name, path in policies.items()}},
                        stats_experiment_documents={str(i): path.read_text() for i, path in enumerate(args.stats_experiment)},
                        capture_input_endpoints_equal=artifact_snapshot(args.pre_context_inputs) == args.pre_context_start)
                    if args.feature_graph_snapshot is not None:
                        context['feature_graph'] = args.feature_graph_snapshot
                    directory = args.output / workload
                    directory.mkdir(parents=True, exist_ok=True)
                    snapshot_path = directory / 'pre-workload-context.json'
                    with snapshot_path.open('x') as stream:
                        json.dump(context, stream, indent=2)
                        stream.write('\n')
                    args.pre_context_path = artifact_snapshot({'catalog_context': snapshot_path})['catalog_context']
                if args.cardinality_probes:
                    probes = json.loads(args.cardinality_probes.read_text(encoding="utf-8"))
                    measured = collect_cardinalities(pg_bindir / "psql", socket, args.port,
                                                     database, probes, args.timeout)
                    (args.output / "actual-cardinalities.json").write_text(json.dumps({
                        "schema_version": 1, "baseline_kind": "measured_actual_rows",
                        "fixture": fixture_context, "mapping": "explicit_sql_to_logical_target",
                        "targets": measured,
                    }, indent=2) + "\n", encoding="utf-8")
                    if any(probe["status"] != "ok" for probe in measured):
                        raise RuntimeError("cardinality probes failed; see actual-cardinalities.json")
                def compare(query: Path) -> dict[str, Any]:
                    return compare_query(
                        args, pg_bindir / "psql", socket, database, workload, query,
                        semantic_xforms, hash_xforms, semantic_xform_set,
                        join_xforms,
                    )

                with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                    workload_results = list(executor.map(compare, queries))
                for result in workload_results:
                    print(
                        f"{workload}/{Path(result['query']).stem}: outcome={result['outcome_equal']} "
                        f"triggers={result['trigger_set_equal']}/{result['trigger_order_equal']} "
                        f"plan={result['plan_comparison']} "
                        f"coverage={result['coverage_status']}",
                        flush=True,
                    )
                query_by_name = {query.name: query for query in queries}
                for index, result in enumerate(workload_results):
                    if args.timing_repeats or args.stats_experiment or args.profile_rule or args.comparison_policies:
                        continue  # Preserve experimental failures, including the first crash.
                    if not any(mode["server_failure"] for mode in all_mode_results(result)):
                        continue
                    if not wait_ready(pg_bindir / "pg_isready", socket, args.port, database):
                        raise RuntimeError("PostgreSQL did not recover after a backend crash")
                    print(f"{workload}/{Path(result['query']).stem}: retry after server recovery", flush=True)
                    workload_results[index] = compare(query_by_name[result["query"]])
                results.extend(workload_results)
        finally:
            run([str(pg_bindir / "pg_ctl"), "-D", str(data), "stop", "-m", "fast"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    observations = experiment_observations(results)
    summary = {
        "artifact_provenance": {"before_server_start": args.artifact_start,
            "all_query_endpoints_equal": all(result["artifact_provenance"]["endpoints_equal"] for result in results)},
        "fixture": fixture_context if args.setup_sql or args.cardinality_probes else None,
        "queries": len(results),
        "outcome_equal": sum(result["outcome_equal"] for result in results),
        "rows_equal": sum(result["rows_equal"] for result in results),
        "trigger_set_equal": sum(result["trigger_set_equal"] for result in results),
        "trigger_order_equal": sum(result["trigger_order_equal"] for result in results),
        "join_enumeration_replaced": sum(
            result["join_enumeration_replaced"] for result in results
        ),
        "dphyper_applied": sum(result["dphyper_applied"] for result in results),
        "plan_identical": sum(result["plan_comparison"] == "identical" for result in results),
        "coverage_status": {
            status: sum(result["coverage_status"] == status for result in results)
            for status in (
                "comparable",
                "replacement_failed",
                "replacement_only",
                "native_unsupported",
            )
        },
        "comparable": {
            "queries": sum(
                result["coverage_status"] == "comparable" for result in results
            ),
            "trigger_set_equal": sum(
                result["coverage_status"] == "comparable"
                and result["trigger_set_equal"]
                for result in results
            ),
            "trigger_order_equal": sum(
                result["coverage_status"] == "comparable"
                and result["trigger_order_equal"]
                for result in results
            ),
            "plan_identical": sum(
                result["coverage_status"] == "comparable"
                and result["plan_comparison"] == "identical"
                for result in results
            ),
        },
        "plan_difference_reasons": {
            reason: sum(result["plan_comparison"] == reason for result in results)
            for reason in ("join_order", "expression_or_property", "physical_shape", "plan_error")
        },
        "timings": {
            mode: {
                field: timing_summary(results, mode, field)
                for field in ("optimization_ms", "planning_ms", "execution_ms", "plan_ms", "rows_ms")
            }
            for mode in ("native", "replacement")
        },
        "stats_experiments": {
            "runs": sum(len(result["stats_experiments"]) for result in results),
            "outcome_equal": sum(
                comparison["outcome_equal"]
                for result in results
                for experiment in result["stats_experiments"]
                for comparison in experiment["comparisons"].values()
            ),
            "rows_equal": sum(
                comparison["rows_equal"]
                for result in results
                for experiment in result["stats_experiments"]
                for comparison in experiment["comparisons"].values()
            ),
            "events": sum(
                len(mode["stats_events"])
                for result in results
                for experiment in result["stats_experiments"]
                for mode in experiment["modes"].values()
            ),
            "outcomes": sum(
                len(mode["experiment_outcomes"])
                for result in results
                for experiment in result["stats_experiments"]
                for mode in experiment["modes"].values()
            ),
        },
        "rule_profile": {
            "rule_hash": args.profile_rule,
            "scenarios": sum(
                len(result["rule_profile"]["scenarios"])
                for result in results if result["rule_profile"] is not None
            ),
            "result_comparable": sum(
                comparison["result_comparable"]
                for result in results if result["rule_profile"] is not None
                for scenario in result["rule_profile"]["scenarios"]
                for comparison in scenario["comparisons"].values()
            ),
            "outcome_equal": sum(
                comparison["outcome_equal"]
                for result in results if result["rule_profile"] is not None
                for scenario in result["rule_profile"]["scenarios"]
                for comparison in scenario["comparisons"].values()
            ),
            "rows_equal": sum(
                comparison["rows_equal"]
                for result in results if result["rule_profile"] is not None
                for scenario in result["rule_profile"]["scenarios"]
                for comparison in scenario["comparisons"].values()
            ),
        },
        "experiment_observability": experiment_distribution_summary(
            observations
        ),
    }
    if args.postgres_oracle:
        summary['postgres_oracle'] = {
            'queries': len(results),
            'valid': sum(result['postgres_oracle']['valid'] for result in results),
            'mode_bag_equal': {mode: sum(result['postgres_oracle']['mode_bag_equal'][mode]
                                         for result in results)
                               for mode in ('native', 'replacement')},
        }
    if args.comparison_policies:
        comparisons = [c for r in results for s in r['policy_comparison']['scenarios']
                       for c in s['comparisons'].values()]
        summary['policy_comparison'] = {
            'scope': 'complete_policy_not_individual_rule_effect',
            'arms': list(args.comparison_policies), 'baseline': next(iter(args.comparison_policies)),
            'comparisons': len(comparisons),
            'result_comparable': sum(c['result_comparable'] for c in comparisons)}
    if args.timing_repeats:
        samples = [sample for result in results
                   for sample in policy_experiment(result).get("timing", {}).get("samples", [])]
        summary["profile_timing"] = {}
        for phase in ("warmup", "measurement"):
            subset = [s for s in samples if s["phase"] == phase]
            summary["profile_timing"][phase] = {
                "attempts": len(subset),
                "statuses": {status: sum(s["status"] == status for s in subset)
                             for status in sorted({s["status"] for s in subset})},
                "diagnostic_comparable": sum(not s["comparison_exclusions"] for s in subset),
            }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "experiment_observations.jsonl").write_text(
        "".join(json.dumps(observation, sort_keys=True) + "\n" for observation in observations),
        encoding="utf-8",
    )
    failed = any(
        not result["artifact_provenance"]["endpoints_equal"]
        or result["coverage_status"] == "replacement_failed"
        or (args.postgres_oracle and not result['postgres_oracle']['valid'])
        or not result["outcome_equal"]
        or (
            result["coverage_status"] == "comparable"
            and not result["rows_equal"]
        )
        or not result["join_enumeration_replaced"]
        or result["forbidden_native_origins"]
        or any(
            experiment_run_status(mode, True) != "ok"
            for experiment in result["stats_experiments"]
            for mode in experiment["modes"].values()
        )
        or any(
            not comparison["outcome_equal"] or not comparison["rows_equal"]
            for experiment in result["stats_experiments"]
            for comparison in experiment["comparisons"].values()
        )
        or any(
            not comparison["result_comparable"]
            for scenario in (
                policy_experiment(result).get('scenarios', [])
            )
            for comparison in scenario["comparisons"].values()
        )
        or (args.strict and (not result["trigger_set_equal"] or not result["trigger_order_equal"] or result["plan_comparison"] != "identical"))
        or any(sample["comparison_exclusions"]
               for sample in policy_experiment(result).get("timing", {}).get("samples", []))
        for result in results
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
