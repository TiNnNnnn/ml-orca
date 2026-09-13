#!/usr/bin/env python3
"""Freeze discovery estimates into bounded, reproducible absolute-card experiments."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import random
import re
import zlib

from ml_orca.collect.run_workload_comparison import trace_records


def geometric_factors(alpha: float, delta: float) -> list[float]:
    """Lero Eq. (6): per-coordinate coverage, conditional on the declared q-error bound."""
    if not math.isfinite(alpha) or not math.isfinite(delta) or alpha <= 1 or delta < 1:
        raise ValueError("Lero grid requires finite alpha > 1 and delta >= 1")
    radius = math.ceil(math.log(delta) / math.log(alpha))
    if radius > 10000:
        raise ValueError("grid exceeds 20001 factors; increase alpha or reduce delta")
    return [alpha ** t for t in range(-radius, radius + 1)]


def discovery_targets(records: list[dict], fingerprints: list[str]) -> list[dict]:
    outcomes = [r for r in records if r.get("kind") == "experiment_outcome"]
    if (len(outcomes) != 1 or outcomes[0].get("stats_targets") != []
            or any(r.get("kind") == "stats_injection" for r in records)):
        raise ValueError("require one completed, observation-only discovery trace")
    if not fingerprints or len(set(fingerprints)) != len(fingerprints):
        raise ValueError("select at least one distinct fingerprint")
    targets = []
    for fingerprint in sorted(fingerprints):
        observed = [r for r in records if r.get("kind") == "stats_observation"
                    and r.get("fingerprint") == fingerprint]
        if not observed:
            raise ValueError(f"target not observed: {fingerprint}")
        signatures = {(r["operator"], r["native_rows"]) for r in observed}
        if len(signatures) != 1 or any(
            r.get("experiment") != outcomes[0].get("experiment") for r in observed
        ):
            raise ValueError(f"ambiguous baseline for {fingerprint}; do not average contexts")
        operator, rows = signatures.pop()
        if (not re.fullmatch(r"[0-9a-f]{16}", fingerprint)
                or not re.fullmatch(r"CLogical[A-Za-z0-9]+", operator)
                or not math.isfinite(rows) or not 1e-250 <= rows <= 1e250):
            raise ValueError(f"invalid target or nonpositive/unrepresentable baseline: {fingerprint}")
        targets.append({"fingerprint": fingerprint, "operator": operator, "baseline_rows": rows})
    return targets


def sweep_points(dimensions: int, factors: list[float], joint_samples: int, seed: int) -> list[dict]:
    if (dimensions < 1 or joint_samples < 0 or not factors
            or any(not math.isfinite(f) or f <= 0 for f in factors)):
        raise ValueError("need positive dimensions/factors and nonnegative sample count")
    factors = sorted(set([1.0, *factors]))
    points = [{"design": "control", "factors": [1.0] * dimensions}]
    for dimension in range(dimensions):
        for factor in factors:
            if factor == 1:
                continue
            point = [1.0] * dimensions
            point[dimension] = factor
            points.append({"design": "one_at_a_time", "factors": point})
    if dimensions > 1:
        points.extend({"design": "common_scale", "factors": [f] * dimensions}
                      for f in factors if f != 1)
    if joint_samples and factors[0] != factors[-1]:
        # Plain Latin hypercube: one sample per marginal log-scale stratum.
        # This covers marginals, not every interaction or feasible data distribution.
        rng = random.Random(seed)
        low, high = math.log(factors[0]), math.log(factors[-1])
        columns = []
        for _ in range(dimensions):
            column = [math.exp(low + (high - low) * (i + rng.random()) / joint_samples)
                      for i in range(joint_samples)]
            rng.shuffle(column)
            columns.append(column)
        points.extend({"design": "log_latin_hypercube", "factors": list(point)}
                      for point in zip(*columns))
    return points


def sampled_input_target(records: list[dict], seed: str) -> tuple[dict, int]:
    """Uniform observed leaf input, without looking at rule outcomes or costs."""
    fingerprints = sorted({r["fingerprint"] for r in records if r.get("kind") == "stats_observation"
                           and r.get("operator") in ("CLogicalGet", "CLogicalCTEConsumer")})
    targets = discovery_targets(records, fingerprints)
    return random.Random(seed).choice(targets), len(targets)


def write_cohort_sweeps(cohort_path: Path, results: Path, output: Path,
                       factors: list[float], seed: int, lero_grid: list[float] | None) -> dict:
    cohort = json.loads(cohort_path.read_text())
    selected = [q for q in cohort["queries"] if q["selected"]]
    if not selected or any(q.get("split") != "discovery" for q in selected):
        raise ValueError("require a nonempty discovery-only sample; do not consume anchors or holdout")
    if len({q["family"] for q in selected}) != len(selected):
        raise ValueError("require one member per family")
    suite = {"schema_version": 1, "cohort": str(cohort_path.resolve()),
             "cohort_crc32": f"{zlib.crc32(cohort_path.read_bytes()):08x}", "seed": seed,
             "target_design": "uniform_observed_get_or_cte_consumer_per_selected_family",
             "scope": "one_coordinate_sensitivity_not_joint_statistics_distribution", "queries": []}
    for query in selected:
        base = results / query["query"]
        comparison = json.loads((base / "comparison.json").read_text())
        if comparison.get("query_crc32") != query["query_crc32"]:
            raise ValueError("discovery SQL identity mismatch")
        trace = base / "stats-1.replacement.trace"
        target, size = sampled_input_target(trace_records(trace.read_text()), f"{seed}:{query['query_crc32']}:input-v1")
        suite["queries"].append({"query": query["query"], "query_crc32": query["query_crc32"],
                                 "family": query["family"], "trace": str(trace.resolve()),
                                 "target": target, "eligible_inputs": size, "input_probability": 1 / size})
    output.mkdir(parents=True, exist_ok=False)
    for query in suite["queries"]:
        destination = output / query["query"]
        write_sweep(Path(query["trace"]), destination, [query["target"]["fingerprint"]],
                    factors, 0, seed, lero_grid=lero_grid)
        query["manifest"] = str((destination / "manifest.json").resolve())
    (output / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    return suite


def write_sweep(source: Path, output: Path, fingerprints: list[str], factors: list[float],
                joint_samples: int, seed: int, actual_cardinalities: Path | None = None,
                lero_grid: list[float] | None = None,
                coordinate_groups: list[list[str]] | None = None) -> dict:
    source_bytes = source.read_bytes()
    trace_bytes = gzip.decompress(source_bytes) if source.suffix == '.gz' else source_bytes
    targets = discovery_targets(trace_records(trace_bytes.decode("utf-8")), fingerprints)
    actual_source = None
    if actual_cardinalities is not None:
        measured = json.loads(actual_cardinalities.read_text(encoding="utf-8"))
        if measured.get("baseline_kind") != "measured_actual_rows":
            raise ValueError("actual cardinalities must come from counted SQL probes")
        entries = measured["targets"]
        indexed = {(t["fingerprint"], t["operator"]): t for t in entries}
        if len(indexed) != len(entries):
            raise ValueError("duplicate actual-cardinality targets")
        for target in targets:
            entry = indexed.get((target["fingerprint"], target["operator"]), {})
            rows = entry.get("actual_rows")
            if entry.get("status") != "ok" or type(rows) is not int or not 0 < rows <= 1e250:
                raise ValueError("each selected target needs a successful positive actual count")
            target["native_rows"] = target["baseline_rows"]
            target["baseline_rows"] = rows
            target["count_sql"] = entry["sql"]
        actual_source = {"path": str(actual_cardinalities.resolve()),
                         "crc32": f"{zlib.crc32(actual_cardinalities.read_bytes()):08x}",
                         "fixture": measured.get("fixture")}
    groups = coordinate_groups if coordinate_groups is not None else [[t['fingerprint']] for t in targets]
    members = [fingerprint for group in groups for fingerprint in group]
    if (not groups or any(not group for group in groups) or len(members) != len(set(members))
            or set(members) != {t['fingerprint'] for t in targets}):
        raise ValueError('coordinate groups must partition all selected fingerprints')
    coordinates = {fingerprint: i for i, group in enumerate(groups) for fingerprint in group}
    points = sweep_points(len(groups), factors, joint_samples, seed)
    for point in points:
        independent = point['factors']
        point['factors'] = [independent[coordinates[t['fingerprint']]] for t in targets]
        if coordinate_groups is not None:
            point['coordinate_factors'] = independent
    configs = []
    for index, point in enumerate(points):
        name = f"card-{index:03d}-{point['design']}"
        lines = [f"experiment: {name}", "discover: false", "cardinalities:"]
        cards = []
        for target, factor in zip(targets, point["factors"]):
            rows = target["baseline_rows"] * factor
            if not math.isfinite(rows) or not 1 <= rows <= 1e250:
                raise ValueError("sampled rows outside ORCA's range; narrow factors instead of clamping")
            cards.append(rows)
            lines.extend([f"  - expression: {target['fingerprint']}",
                          f"    operator: {target['operator']}", f"    rows: {rows:.17g}"])
        point.update({"experiment": name, "file": name + ".yaml", "requested_rows": cards})
        configs.append("\n".join(lines) + "\n")
    manifest = {
        "schema_version": 1,
        "sampling_measure": {
            "family": "independent_log_uniform",
            "factor_bounds": [min(1.0, *factors), max(1.0, *factors)],
            "bounds_basis": "user_declared_stress_range_not_empirically_calibrated",
            "estimation_subset": "log_latin_hypercube",
            "joint_samples": joint_samples,
            "diagnostic_points_are_not_probability_samples": True,
        },
        "baseline_kind": "measured_actual_rows" if actual_source else "frozen_native_estimate",
        "actual_cardinalities": actual_source,
        "factor_design": ({"method": "lero_equation_6", "alpha": lero_grid[0],
                           "declared_q_error_bound": lero_grid[1],
                           "full_cartesian_coverage": False} if lero_grid else {"method": "explicit_factors"}),
        "source_trace": str(source.resolve()),
        "source_crc32": f"{zlib.crc32(source_bytes):08x}",
        "seed": seed,
        "targets": targets,
        "points": points,
    }
    if coordinate_groups is not None:
        manifest['coordinate_groups'] = groups
        manifest['coupling'] = 'same_multiplier_preserves_baseline_ratios_within_each_declared_group'
        manifest['sampling_measure']['space'] = 'declared_coordinate_groups_not_individual_targets'
    # Refuse to overwrite an earlier experiment, and validate all cards before writing.
    output.mkdir(parents=True, exist_ok=False)
    for point, config in zip(points, configs):
        (output / point["file"]).write_text(config, encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path)
    source.add_argument("--cohort", type=Path, help="frozen discovery-only sample; draw one observed input per family")
    parser.add_argument("--results", type=Path, help="observation batch for --cohort")
    parser.add_argument("--fingerprint", action="append",
                        help="explicit discovery target; repeat to sample joint interventions")
    factor_options = parser.add_mutually_exclusive_group(required=True)
    factor_options.add_argument("--factors", type=float, nargs="+",
                        help="declared stress factors; bounds are not inferred from workload errors")
    factor_options.add_argument("--lero-grid", type=float, nargs=2, metavar=("ALPHA", "DELTA"),
                                help="Lero Eq. (6) geometric factors; does not enumerate the full Cartesian product")
    parser.add_argument("--joint-samples", type=int, required=True,
                        help="explicit LHS budget, not a statistical sufficiency guarantee")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actual-cardinalities", type=Path,
                        help="runner's counted SQL probes; use actual rather than native rows as baseline")
    parser.add_argument('--coordinate-group', action='append',
                        help='comma-separated fingerprints sharing one multiplier; groups must partition targets')
    parser.add_argument("--output", type=Path, required=True, help="new directory, normally under output/")
    args = parser.parse_args()
    if args.cohort:
        if not args.results or args.fingerprint or args.actual_cardinalities or args.joint_samples or args.coordinate_group:
            parser.error("--cohort requires --results and --joint-samples 0; no explicit targets or actual-cardinalities")
    elif not args.fingerprint or args.results:
        parser.error("--trace requires --fingerprint and cannot use --results")
    try:
        factors = geometric_factors(*args.lero_grid) if args.lero_grid else args.factors
        if args.cohort:
            suite = write_cohort_sweeps(args.cohort, args.results, args.output, factors, args.seed, args.lero_grid)
            print(f"Generated {len(suite['queries'])} fixed-query sweeps: {args.output / 'suite.json'}")
            return
        manifest = write_sweep(args.trace, args.output, args.fingerprint,
                               factors, args.joint_samples, args.seed, args.actual_cardinalities, args.lero_grid,
                               [g.split(',') for g in args.coordinate_group] if args.coordinate_group else None)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        parser.error(str(error))
    print(f"Generated {len(manifest['points'])} experiments: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
