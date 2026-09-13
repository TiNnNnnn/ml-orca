#!/usr/bin/env python3
"""Instantiate named SQL value parameters into a separate runner workload."""

import argparse
import json
import math
from ml_orca.common.paths import TEST_ASSETS
from pathlib import Path
import re
import zlib

import sqlglot
from sqlglot import exp


def instantiate(sql: str, parameters: dict) -> str:
    statements = sqlglot.parse(sql, read="postgres")
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("template must contain one query")
    tree = statements[0]
    slots = list(tree.find_all(exp.Placeholder))
    names = {slot.name for slot in slots}
    if (not names or "" in names or any(tree.find_all(exp.Parameter)) or not isinstance(parameters, dict)
            or set(parameters) != names):
        raise ValueError("parameters must exactly match named :value placeholders")
    if any(slot.find_ancestor(exp.Table, exp.Column, exp.Identifier) for slot in slots):
        raise ValueError("parameters bind values, not identifiers")
    for value in parameters.values():
        if type(value) not in (str, bool, int, float, type(None)) or (
                type(value) is float and not math.isfinite(value)):
            raise ValueError("parameter values must be finite JSON scalars")
    rendered = tree.transform(lambda node: exp.convert(parameters[node.name])
                              if isinstance(node, exp.Placeholder) else node)
    return rendered.sql(dialect="postgres", pretty=True) + ";\n"


def instantiate_relations(sql: str, relations: dict[str, str]) -> str:
    """Simultaneous relation substitution, preserving each occurrence's alias.

    The caller must preserve schema and rule integrity constraints and check both
    rule sides. This instantiates Input subtrees; it is not an equivalence rule.
    CTE/name resolution is intentionally outside the generated-example domain.
    """
    def query(text):
        statements = sqlglot.parse(text, read="postgres")
        if (len(statements) != 1 or not isinstance(statements[0], exp.Query)
                or any(statements[0].find_all(exp.CTE, exp.Into))):
            raise ValueError("relation substitution requires one CTE-free query without INTO")
        return statements[0]

    tree = query(sql)
    replacements = {name: query(text) for name, text in relations.items()}
    tables = list(tree.find_all(exp.Table))
    if not tables:
        raise ValueError("require at least one relation occurrence")
    for table in tables:
        if (table.db or table.catalog or not isinstance(table.this, exp.Identifier)
                or table.name not in replacements):
            raise ValueError("every unqualified base relation needs a replacement")
        alias = table.args.get('alias')
        if alias is None:
            alias = exp.TableAlias(this=table.this.copy())
        table.replace(exp.Subquery(this=replacements[table.name].copy(), alias=alias.copy()))
    return tree.sql(dialect="postgres", pretty=True) + ";\n"


def write_workload(specification: Path, source_root: Path, output: Path) -> dict:
    raw_spec = specification.read_bytes()
    spec = json.loads(raw_spec)
    workload = spec.get("workload")
    if workload not in ("tpch", "tpcds", "job", "sqlstorm") or spec.get("schema_version") != 1:
        raise ValueError("require schema_version 1 and a known workload")
    schema = (source_root / workload / "schema.sql").read_bytes()
    entries, rendered, template_ids = [], {}, set()
    for template in spec["templates"]:
        name = template["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in template_ids:
            raise ValueError("template IDs must be unique safe path components")
        template_ids.add(name)
        source = (specification.parent / template["sql_file"]).resolve()
        raw = source.read_bytes()
        if not template["cases"]:
            raise ValueError("each template needs at least one parameter case")
        for case in template["cases"]:
            case_id = case["id"]
            if not re.fullmatch(r"[a-z][a-z0-9_]*", case_id):
                raise ValueError("case IDs must be safe path components")
            query_id = f"{name}__{case_id}"
            if query_id in rendered:
                raise ValueError("duplicate instance output name")
            sql = instantiate(raw.decode("utf-8"), case["parameters"])
            rendered[query_id] = sql
            entries.append({"query": f"{workload}/{query_id}", "template": name,
                            "case": case_id, "parameters": case["parameters"],
                            "template_path": str(source), "template_crc32": f"{zlib.crc32(raw):08x}",
                            "query_crc32": f"{zlib.crc32(sql.encode()):08x}"})
    if not entries:
        raise ValueError("require at least one template")
    manifest = {"schema_version": 1, "sampling_unit": "declared_parameter_instances",
                "parameter_design": spec["parameter_design"], "workload": workload,
                "source_specification": str(specification.resolve()),
                "specification_crc32": f"{zlib.crc32(raw_spec):08x}",
                "schema_crc32": f"{zlib.crc32(schema):08x}", "sqlglot_version": sqlglot.__version__,
                "not_guaranteed": ["search_space_similarity", "population_representativeness", "statistical_independence"],
                "queries": entries}
    # Validate every case before creating anything; never replace an old experiment.
    output.mkdir(parents=True, exist_ok=False)
    destination = output / workload
    (destination / "sql").mkdir(parents=True)
    (destination / "schema.sql").write_bytes(schema)
    for query_id, sql in rendered.items():
        (destination / "sql" / f"{query_id}.sql").write_text(sql, encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--workload-root", type=Path, default=TEST_ASSETS / "workloads")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_workload(args.spec, args.workload_root, args.output)
    print(json.dumps({"instances": len(manifest["queries"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
