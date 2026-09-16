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


def write_relation_workload(specification: Path, output: Path) -> dict:
    """Freeze predeclared Input-subtree substitutions as a runner workload."""
    raw_spec = specification.read_bytes()
    spec = json.loads(raw_spec)
    base = specification.parent
    workload = spec.get("workload")
    template = spec.get("template")
    if (spec.get("schema_version") != 1 or not isinstance(workload, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", workload)
            or not isinstance(template, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", template)):
        raise ValueError("require schema_version 1 and safe workload/template names")

    def read(relative: str) -> tuple[Path, bytes]:
        path = (base / relative).resolve()
        if not path.is_file():
            raise ValueError(f"missing input file: {relative}")
        return path, path.read_bytes()

    source_path, source_raw = read(spec["source_sql"])
    target_path, target_raw = read(spec["target_sql"])
    schema_path, schema = read(spec["schema_sql"])
    setup_parts = [read(path) for path in spec.get("setup_sql", [])]
    if not spec.get("cases"):
        raise ValueError("require predeclared relation cases")

    def relation_names(sql: bytes) -> set[str]:
        statements = sqlglot.parse(sql.decode(), read="postgres")
        if len(statements) != 1 or not isinstance(statements[0], exp.Query):
            raise ValueError("source and target must contain one query")
        return {table.name for table in statements[0].find_all(exp.Table)}

    expected = relation_names(source_raw) | relation_names(target_raw)
    entries, rendered, ids = [], {}, set()
    for case in spec["cases"]:
        case_id = case.get("id")
        if (not isinstance(case_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", case_id)
                or case_id in ids):
            raise ValueError("case IDs must be unique safe path components")
        ids.add(case_id)
        relation_files = case.get("relations")
        if not isinstance(relation_files, dict) or set(relation_files) != expected:
            raise ValueError("each case must replace every relation exactly once")
        relations, inputs = {}, []
        for name, relative in relation_files.items():
            path, raw = read(relative)
            relations[name] = raw.decode()
            inputs.append({"relation": name, "path": str(path),
                           "crc32": f"{zlib.crc32(raw):08x}"})
        source = instantiate_relations(source_raw.decode(), relations)
        target = instantiate_relations(target_raw.decode(), relations)
        rendered[case_id] = (source, target)
        parameters = case.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("each case requires declared parameters")
        entries.append({"query": f"{workload}/{case_id}", "case": case_id,
                        "template": template, "parameters": parameters,
                        "split": case.get("split"), "relations": inputs,
                        "query_crc32": f"{zlib.crc32(source.encode()):08x}",
                        "source_crc32": f"{zlib.crc32(source.encode()):08x}",
                        "target_crc32": f"{zlib.crc32(target.encode()):08x}"})

    if any(entry["split"] != "holdout" for entry in entries):
        raise ValueError("relation workload currently requires an explicit holdout split")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / workload
    (destination / "sql").mkdir(parents=True)
    (output / "checks").mkdir()
    (destination / "schema.sql").write_bytes(schema)
    setup = b"\n".join(raw for _, raw in setup_parts)
    (output / "setup.sql").write_bytes(setup)
    for case_id, (source, target) in rendered.items():
        (destination / "sql" / f"{case_id}.sql").write_text(source)
        (output / "checks" / f"{case_id}_target.sql").write_text(target)
        left, right = source.rstrip(";\n"), target.rstrip(";\n")
        comparison = ("SELECT COUNT(*) AS differences FROM (((" + left
                      + ") EXCEPT ALL (" + right + ")) UNION ALL ((" + right
                      + ") EXCEPT ALL (" + left + "))) AS difference_rows;\n")
        (output / "checks" / f"{case_id}_equivalence.sql").write_text(comparison)
    manifest = {
        "schema_version": 1, "sampling_unit": "predeclared_application_case",
        "split": "holdout", "workload": workload,
        "template": template, "parameter_design": spec.get("parameter_design"),
        "source_specification": str(specification.resolve()),
        "specification_crc32": f"{zlib.crc32(raw_spec):08x}",
        "source_template": {"path": str(source_path), "crc32": f"{zlib.crc32(source_raw):08x}"},
        "target_template": {"path": str(target_path), "crc32": f"{zlib.crc32(target_raw):08x}"},
        "schema": {"path": str(schema_path), "crc32": f"{zlib.crc32(schema):08x}"},
        "setup": [{"path": str(path), "crc32": f"{zlib.crc32(raw):08x}"}
                  for path, raw in setup_parts],
        "cases": entries,
        "not_guaranteed": ["population_representativeness", "independent_rule_equivalence_proof"],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec", type=Path)
    group.add_argument("--relation-spec", type=Path)
    parser.add_argument("--workload-root", type=Path, default=TEST_ASSETS / "workloads")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = (write_relation_workload(args.relation_spec, args.output) if args.relation_spec
                else write_workload(args.spec, args.workload_root, args.output))
    print(json.dumps({"instances": len(manifest.get("queries", manifest.get("cases", []))),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
