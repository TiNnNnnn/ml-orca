#!/usr/bin/env python3
"""Import fixed optimizer workloads from their upstream repositories."""

from __future__ import annotations

import argparse
import csv
import json
from ml_orca.common.paths import TEST_ASSETS
from pathlib import Path
import subprocess

import sqlglot


SQLSTORM_URL = "https://github.com/SQL-Storm/SQLStorm"
JOB_URL = "https://github.com/gregrahn/join-order-benchmark"
TPCDS_URL = "https://github.com/avamingli/pg_tpcds"
TPCH_URL = "https://github.com/tvondra/pg_tpch"


def git_sha(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def postgres_sql(source: str) -> str:
    statements = sqlglot.parse(source, read="postgres")
    if not statements:
        raise ValueError("SQL source contains no statements")
    return "\n\n".join(
        statement.sql(dialect="postgres", pretty=True) + ";"
        for statement in statements
    ) + "\n"


def write_workload(
    output: Path,
    name: str,
    schema: str,
    queries: list[tuple[str, str]],
    source: dict[str, object],
) -> None:
    root = output / name
    query_dir = root / "sql"
    query_dir.mkdir(parents=True, exist_ok=True)
    for path in query_dir.glob("*.sql"):
        path.unlink()
    (root / "schema.sql").write_text(postgres_sql(schema), encoding="utf-8")
    for query_id, query in queries:
        (query_dir / f"{query_id}.sql").write_text(
            postgres_sql(query), encoding="utf-8"
        )
    manifest = {"name": name, "query_count": len(queries), **source}
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def numbered_queries(directory: Path, count: int, width: int) -> list[tuple[str, str]]:
    return [
        (f"q{number:0{width}d}", (directory / f"{number}.sql").read_text())
        for number in range(1, count + 1)
    ]


def sampled_sqlstorm_queries(
    root: Path, dataset: str, count: int, excluded: set[str] | None = None
) -> list[tuple[str, str]]:
    directory = root / "v1.0" / dataset
    with (directory / "distinct_queries.csv").open(encoding="utf-8", newline="") as stream:
        query_names = [
            row["query"] for row in csv.DictReader(stream)
            if row["query"] not in (excluded or set())
        ][:count]
    if len(query_names) != count:
        raise ValueError(f"SQLStorm {dataset} has only {len(query_names)} selectable queries")
    return [
        (
            f"s{int(Path(name).stem):05d}",
            (directory / "queries" / name).read_text(encoding="utf-8"),
        )
        for name in query_names
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlstorm", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--pg-tpcds", type=Path, required=True)
    parser.add_argument("--pg-tpch", type=Path, required=True)
    parser.add_argument("--sqlstorm-sample-count", type=int, default=250)
    parser.add_argument(
        "--output", type=Path, default=TEST_ASSETS / "workloads"
    )
    args = parser.parse_args()

    sqlstorm_sha = git_sha(args.sqlstorm)
    common = {"source_url": SQLSTORM_URL, "source_commit": sqlstorm_sha}
    sample_count = args.sqlstorm_sample_count
    write_workload(
        args.output,
        "tpch",
        "\n".join(
            (args.pg_tpch / f"dss/{name}").read_text(encoding="utf-8")
            for name in ("tpch-create.sql", "tpch-pkeys.sql", "tpch-alter.sql")
        ),
        numbered_queries(args.sqlstorm / "v0.0/tpch/queries", 22, 2)
        + sampled_sqlstorm_queries(args.sqlstorm, "tpch", sample_count),
        {
            "sources": [
                {**common, "path": "v0.0/tpch/queries/1.sql..22.sql"},
                {
                    "source_url": TPCH_URL,
                    "source_commit": git_sha(args.pg_tpch),
                    "path": "dss/tpch-{create,pkeys,alter}.sql",
                },
            ],
            "sqlstorm_sample": {
                **common,
                "path": "v1.0/tpch/distinct_queries.csv and queries/*.sql",
                "count": sample_count,
            },
        },
    )
    write_workload(
        args.output,
        "tpcds",
        (
            args.pg_tpcds
            / "DSGen-software-code-4.0.0/tools/tpcds.sql"
        ).read_text(encoding="utf-8"),
        numbered_queries(args.sqlstorm / "v0.0/tpcds/queries", 103, 3)
        + sampled_sqlstorm_queries(args.sqlstorm, "tpcds", sample_count),
        {
            "sources": [
                {**common, "path": "v0.0/tpcds/queries/1.sql..103.sql"},
                {
                    "source_url": TPCDS_URL,
                    "source_commit": git_sha(args.pg_tpcds),
                    "path": "DSGen-software-code-4.0.0/tools/tpcds.sql",
                },
            ],
            "selection": "99 templates produce 103 executable statements",
            "sqlstorm_sample": {
                **common,
                "path": "v1.0/tpcds/distinct_queries.csv and queries/*.sql",
                "count": sample_count,
            },
        },
    )

    job_queries = [
        (
            f"q{int(path.stem[:-1]):02d}{path.stem[-1]}",
            path.read_text(encoding="utf-8"),
        )
        for path in sorted(args.job.glob("[0-9]*.sql"))
    ]
    write_workload(
        args.output,
        "job",
        (args.job / "schema.sql").read_text(encoding="utf-8"),
        job_queries + sampled_sqlstorm_queries(args.sqlstorm, "job", sample_count),
        {
            "source_url": JOB_URL,
            "source_commit": git_sha(args.job),
            "source_path": "schema.sql and [0-9]*.sql",
            "sqlstorm_sample": {
                **common,
                "path": "v1.0/job/distinct_queries.csv and queries/*.sql",
                "count": sample_count,
            },
        },
    )

    semantic_csv = args.sqlstorm / "v1.0/stackoverflow/semantic_queries.csv"
    with semantic_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    sqlstorm_queries = [
        (f"q{int(row['query'][:-4]):05d}", row["text"]) for row in rows
    ]
    sqlstorm_queries += sampled_sqlstorm_queries(
        args.sqlstorm,
        "stackoverflow",
        sample_count,
        {row["query"] for row in rows},
    )
    write_workload(
        args.output,
        "sqlstorm",
        (args.sqlstorm / "v1.0/stackoverflow/schema.sql").read_text(
            encoding="utf-8"
        ),
        sqlstorm_queries,
        {
            **common,
            "source_path": "v1.0/stackoverflow/semantic_queries.csv",
            "selection": "23 semantically validated PostgreSQL queries plus a distinct-query sample",
            "sqlstorm_sample": {
                **common,
                "path": "v1.0/stackoverflow/distinct_queries.csv and queries/*.sql",
                "count": sample_count,
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
