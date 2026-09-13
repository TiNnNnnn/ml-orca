#!/usr/bin/env python3
"""Import WeTune's real-world workloads as reproducible pgORCA corpus assets.

The WeTune database stores queries by application and statement id.  This tool
keeps those ids as physical line numbers, translates MySQL queries to
PostgreSQL, and reduces the source schema to metadata relevant to logical
rewrites (columns, nullability, primary/unique keys, and foreign keys).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import sqlglot
    from sqlglot import exp
    from sqlglot.dialects.mysql import MySQL
    from sqlglot.errors import ErrorLevel
except ImportError as error:  # pragma: no cover - developer-only dependency
    raise SystemExit("sqlglot is required to import WeTune workloads") from error


POSTGRES_SCHEMA_MARKERS = ("SET statement_timeout", "SET search_path=public")
MYSQL_VERSION_COMMENT = re.compile(r"/\*!.*?\*/\s*;?", re.DOTALL)
MYSQL_INDEX_METHOD = re.compile(r"\s+USING\s+(?:BTREE|HASH)\b", re.IGNORECASE)
INTEGER_TYPES = {
    "TINYINT": "SMALLINT",
    "SMALLINT": "SMALLINT",
    "MEDIUMINT": "INTEGER",
    "INT": "INTEGER",
    "UINT": "BIGINT",
    "BIGINT": "BIGINT",
    "UBIGINT": "NUMERIC(20)",
}


class WorkloadMySQL(MySQL):
    class Parser(MySQL.Parser):
        # Keep the function instead of lowering it to the invalid PG year 0000.
        FUNCTIONS = {**MySQL.Parser.FUNCTIONS,
                     "TO_DAYS": lambda args: exp.Anonymous(this="TO_DAYS", expressions=args)}


def detect_dialect(schema_sql: str) -> str:
    if any(marker in schema_sql for marker in POSTGRES_SCHEMA_MARKERS):
        return "postgres"
    if "CREATE EXTENSION IF NOT EXISTS plpgsql" in schema_sql:
        return "postgres"
    return "mysql"


def normalized_identifier(identifier: exp.Identifier) -> exp.Identifier:
    return exp.Identifier(this=identifier.name.lower(), quoted=True)


def normalize_mysql_identifiers(expression: exp.Expression) -> exp.Expression:
    def normalize(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Identifier):
            return normalized_identifier(node)
        return node

    return expression.transform(normalize)


def normalize_mysql_query(expression: exp.Expression) -> exp.Expression:
    """Preserve MySQL parameter and boolean semantics in PostgreSQL SQL."""
    parameter = 0

    def normalize(node: exp.Expression) -> exp.Expression:
        nonlocal parameter
        if isinstance(node, exp.Placeholder):
            parameter += 1
            return exp.Parameter(this=exp.Literal.number(parameter))
        if isinstance(node, exp.Boolean):
            return exp.Literal.number(1 if node.this else 0)
        return node

    return expression.transform(normalize)


def normalize_mysql_functions(expression: exp.Expression) -> exp.Expression:
    """Lower functions for the valid Gregorian date/text domain of imported data."""
    # Walk bottom-up so generated replacements retain already-normalized arguments.
    for node in reversed(list(expression.walk())):
        replacement = None
        if isinstance(node, exp.CurrentDate) or (
                isinstance(node, exp.Anonymous) and node.name.upper() == "CURDATE"):
            if (isinstance(node, exp.CurrentDate) and node.args.get('this')) or node.expressions:
                raise ValueError('current date requires no arguments')
            parent = node.parent
            while isinstance(parent, exp.Paren):
                parent = parent.parent
            if isinstance(parent, (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod)) and not (
                    isinstance(parent, (exp.Add, exp.Sub)) and isinstance(parent.expression, exp.Interval)
                    or isinstance(parent, exp.Add) and isinstance(parent.this, exp.Interval)):
                raise ValueError('current date numeric coercion requires explicit MySQL typing')
            # MySQL current-date functions use statement time; PG CURRENT_DATE uses
            # transaction time. Preserve the date domain and session time zone.
            replacement = exp.cast(exp.Anonymous(this='statement_timestamp'), 'DATE')
        elif isinstance(node, exp.DateDiff) and (
                node.args.get("unit") is None or node.args["unit"].name.upper() == "DAY"):
            # MySQL DATEDIFF ignores time of day; AGE returns an interval, not days.
            replacement = exp.Paren(this=exp.Sub(
                this=exp.cast(node.this.copy(), "DATE"),
                expression=exp.cast(node.expression.copy(), "DATE")))
        elif isinstance(node, exp.Anonymous) and node.name.upper() == "TO_DAYS":
            if len(node.expressions) != 1:
                raise ValueError("TO_DAYS requires one argument")
            # MySQL's day number at AD 0001-01-01 is 366. PostgreSQL has no year 0.
            replacement = exp.Paren(this=exp.Add(this=exp.Sub(
                this=exp.cast(node.expressions[0].copy(), "DATE"),
                expression=exp.cast(exp.Literal.string("0001-01-01"), "DATE")),
                expression=exp.Literal.number(366)))
        elif isinstance(node, exp.Concat) and all(
                isinstance(arg, exp.Literal) and arg.is_string for arg in node.expressions):
            # A typed CONCAT result cannot be compared to a timestamp in PG;
            # folding literal-only concatenation preserves contextual literal typing.
            replacement = exp.Literal.string("".join(arg.this for arg in node.expressions))
        elif isinstance(node, exp.GroupConcat):
            value = node.this
            if isinstance(value, exp.Order):
                value = value.this
                if isinstance(value, exp.Distinct):
                    raise ValueError("GROUP_CONCAT DISTINCT with ORDER BY requires typed order normalization")
            if isinstance(value, exp.Distinct):
                value.set("expressions", [exp.cast(arg.copy(), "TEXT") for arg in value.expressions])
            else:
                value.replace(exp.cast(value.copy(), "TEXT"))
        if replacement is not None:
            if node is expression:
                expression = replacement
            else:
                node.replace(replacement)
    return expression


def expand_having_aliases(expression: exp.Expression) -> exp.Expression:
    """Expand SELECT aliases referenced by MySQL HAVING clauses."""
    for select in expression.find_all(exp.Select):
        having = select.args.get("having")
        if having is None:
            continue
        aliases = {
            item.alias.lower(): item.this.copy()
            for item in select.expressions
            if isinstance(item, exp.Alias) and item.alias
        }
        if not aliases:
            continue

        def expand(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Column) and not node.table:
                replacement = aliases.get(node.name.lower())
                if replacement is not None:
                    return replacement.copy()
            return node

        having.set("this", having.this.transform(expand))
    return expression


def translate_query(query: str, dialect: str) -> tuple[str, str | None]:
    attempts = [(query, None)]
    if query.rstrip().endswith('"'):
        attempts.append((query.rstrip()[:-1], "removed unmatched trailing double quote"))

    errors: list[str] = []
    for candidate, repair in attempts:
        try:
            expression = sqlglot.parse_one(
                candidate, read=WorkloadMySQL if dialect == "mysql" else dialect,
                error_level=ErrorLevel.RAISE
            )
            if dialect == "mysql":
                expression = normalize_mysql_identifiers(expression)
                expression = normalize_mysql_query(expression)
                expression = expand_having_aliases(expression)
                expression = normalize_mysql_functions(expression)
            rendered = expression.sql(
                dialect="postgres", unsupported_level=ErrorLevel.RAISE
            )
            return " ".join(rendered.splitlines()), repair
        except Exception as error:  # sqlglot exposes several parse/generation errors
            errors.append(str(error).splitlines()[0])

    raise ValueError("; ".join(errors))


def postgres_type(data_type: exp.DataType) -> str:
    type_name = data_type.this.name
    # MySQL BOOL/BOOLEAN is an alias for TINYINT(1).  This function is used
    # only while converting MySQL schemas, so retain its numeric semantics.
    if type_name == "BOOLEAN":
        return "SMALLINT"
    if type_name in INTEGER_TYPES:
        return INTEGER_TYPES[type_name]

    rendered = data_type.sql(dialect="postgres")
    if type_name in {
        "BINARY",
        "VARBINARY",
        "TINYBLOB",
        "BLOB",
        "MEDIUMBLOB",
        "LONGBLOB",
    }:
        return "BYTEA"
    if type_name in {"TINYTEXT", "MEDIUMTEXT", "LONGTEXT", "ENUM", "SET"}:
        return "TEXT"
    if type_name == "BIT":
        return "INTEGER"
    return re.sub(r"\b(?:U?BIGINT|U?INT|SMALLINT)\s*\(\d+\)", "INTEGER", rendered)


def identifier_sql(identifier: exp.Expression) -> str:
    if isinstance(identifier, exp.Column):
        name = identifier.name
    elif isinstance(identifier, exp.Identifier):
        name = identifier.name
    else:
        found = identifier.find(exp.Identifier)
        if found is None:
            raise ValueError(f"cannot extract identifier from {identifier}")
        name = found.name
    return normalized_identifier(exp.Identifier(this=name)).sql(dialect="postgres")


def column_names(expressions: Iterable[exp.Expression]) -> list[str]:
    return [identifier_sql(expression) for expression in expressions]


def mysql_schema(schema_sql: str) -> str:
    source = MYSQL_VERSION_COMMENT.sub("", schema_sql)
    # The access method changes physical indexing only and sqlglot versions
    # disagree on where MySQL permits this clause inside a table definition.
    source = MYSQL_INDEX_METHOD.sub("", source)
    parsed = sqlglot.parse(source, read="mysql", error_level=ErrorLevel.IGNORE)
    tables: list[str] = []
    foreign_keys: list[str] = []
    table_names: set[str] = set()

    creates = [
        statement
        for statement in parsed
        if isinstance(statement, exp.Create)
        and statement.args.get("kind") == "TABLE"
        and isinstance(statement.this, exp.Schema)
    ]
    for create in creates:
        table_names.add(create.this.this.name.lower())

    for create in creates:
        table_name = create.this.this.name.lower()
        table_sql = normalized_identifier(exp.Identifier(this=table_name)).sql(
            dialect="postgres"
        )
        definitions: list[str] = []

        for definition in create.this.expressions:
            if isinstance(definition, exp.ColumnDef):
                constraints = definition.args.get("constraints") or []
                not_null = any(
                    isinstance(constraint.kind, exp.NotNullColumnConstraint)
                    for constraint in constraints
                )
                rendered = (
                    f"{identifier_sql(definition.this)} {postgres_type(definition.kind)}"
                )
                if not_null:
                    rendered += " NOT NULL"
                definitions.append(rendered)
            elif isinstance(definition, exp.PrimaryKey):
                columns = column_names(definition.expressions)
                definitions.append(f"PRIMARY KEY ({', '.join(columns)})")
            elif isinstance(definition, exp.UniqueColumnConstraint):
                target = definition.this
                expressions = target.expressions if isinstance(target, exp.Schema) else []
                if expressions:
                    columns = column_names(expressions)
                    definitions.append(f"UNIQUE ({', '.join(columns)})")
            elif isinstance(definition, exp.Constraint):
                foreign = next(
                    (
                        item
                        for item in definition.expressions
                        if isinstance(item, exp.ForeignKey)
                    ),
                    None,
                )
                if foreign is None or foreign.args.get("reference") is None:
                    continue
                reference = foreign.args["reference"].this
                if not isinstance(reference, exp.Schema):
                    continue
                referenced_table = reference.this.name.lower()
                if referenced_table not in table_names:
                    continue
                local_columns = column_names(foreign.expressions)
                referenced_columns = column_names(reference.expressions)
                referenced_table_sql = normalized_identifier(
                    exp.Identifier(this=referenced_table)
                ).sql(dialect="postgres")
                foreign_keys.append(
                    f"ALTER TABLE {table_sql} ADD FOREIGN KEY "
                    f"({', '.join(local_columns)}) REFERENCES {referenced_table_sql} "
                    f"({', '.join(referenced_columns)});"
                )

        if definitions:
            tables.append(f"CREATE TABLE {table_sql} (\n  " + ",\n  ".join(definitions) + "\n);")

    if not tables:
        raise ValueError("schema contains no MySQL CREATE TABLE statements")
    return "\n\n".join(tables + foreign_keys) + "\n"


def postgres_schema(schema_sql: str) -> str:
    logging.getLogger("sqlglot").setLevel(logging.ERROR)
    parsed = sqlglot.parse(schema_sql, read="postgres", error_level=ErrorLevel.IGNORE)
    tables: list[str] = []
    constraints: list[str] = []
    unique_constraints: list[str] = []
    for statement in parsed:
        if statement is None:
            continue
        rendered = statement.sql(dialect="postgres")
        normalized = rendered.lstrip().upper()
        if normalized.startswith("CREATE TABLE"):
            tables.append(rendered.rstrip(";") + ";")
        elif (
            isinstance(statement, exp.Create)
            and statement.args.get("kind") == "INDEX"
            and bool(statement.args.get("unique"))
            and isinstance(statement.this, exp.Index)
        ):
            # ORCA exposes UNIQUE constraints as relational keys, but not
            # standalone PostgreSQL indexes. An unconditional unique index over
            # plain columns proves exactly the same key property, so represent
            # it as a constraint in the reduced corpus schema. Partial and
            # expression indexes do not prove global Unique(attrs) and are
            # deliberately omitted.
            index = statement.this
            table = index.args.get("table")
            parameters = index.args.get("params")
            columns: list[exp.Column] = []
            if isinstance(table, exp.Table) and isinstance(
                parameters, exp.IndexParameters
            ) and parameters.args.get("where") is None:
                for ordered in parameters.args.get("columns") or []:
                    column = (
                        ordered.this if isinstance(ordered, exp.Ordered) else ordered
                    )
                    if not isinstance(column, exp.Column):
                        columns = []
                        break
                    columns.append(column)
            if columns:
                name = index.this.sql(dialect="postgres")
                table_sql = table.sql(dialect="postgres")
                columns_sql = ", ".join(
                    column.sql(dialect="postgres") for column in columns
                )
                unique_constraints.append(
                    f"ALTER TABLE ONLY {table_sql} ADD CONSTRAINT {name} "
                    f"UNIQUE ({columns_sql});"
                )
        elif normalized.startswith("ALTER TABLE") and "ADD CONSTRAINT" in normalized:
            if any(kind in normalized for kind in ("PRIMARY KEY", "UNIQUE", "FOREIGN KEY")):
                constraints.append(rendered.rstrip(";") + ";")

    if not tables:
        raise ValueError("schema contains no PostgreSQL CREATE TABLE statements")
    return "\n\n".join(tables + constraints + unique_constraints) + "\n"


def render_lines(statements: dict[int, str]) -> str:
    if not statements:
        return ""
    lines = [""] * max(statements)
    for statement_id, statement in statements.items():
        lines[statement_id - 1] = statement.rstrip().rstrip(";")
    return "\n".join(lines) + "\n"


def quoted_name(name: str) -> str:
    return '"' + name.lower().replace('"', '""') + '"'


ALTER_UNIQUE = re.compile(
    r"ALTER\s+TABLE(?:\s+ONLY)?\s+(?:public\.)?(?:\"([^\"]+)\"|([A-Za-z_][\w$]*))"
    r"\s+ADD\s+CONSTRAINT\s+\S+\s+(?:PRIMARY\s+KEY|UNIQUE)\s*\(([^)]*)\)",
    re.IGNORECASE,
)


def schema_catalog(
    schema_sql: str,
) -> tuple[dict[str, set[str]], dict[str, set[tuple[str, ...]]]]:
    logging.getLogger("sqlglot").setLevel(logging.ERROR)
    tables: dict[str, set[str]] = {}
    unique_keys: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for statement in sqlglot.parse(
        schema_sql, read="postgres", error_level=ErrorLevel.IGNORE
    ):
        if (
            isinstance(statement, exp.Create)
            and statement.args.get("kind") == "INDEX"
            and bool(statement.args.get("unique"))
            and isinstance(statement.this, exp.Index)
        ):
            table_expr = statement.this.args.get("table")
            parameters = statement.this.args.get("params")
            index_columns: list[str] = []
            if isinstance(table_expr, exp.Table) and isinstance(
                parameters, exp.IndexParameters
            ) and parameters.args.get("where") is None:
                for ordered in parameters.args.get("columns") or []:
                    column_expr = (
                        ordered.this if isinstance(ordered, exp.Ordered) else ordered
                    )
                    if not isinstance(column_expr, exp.Column):
                        index_columns = []
                        break
                    index_columns.append(column_expr.name.lower())
            if index_columns:
                unique_keys[table_expr.name.lower()].add(tuple(index_columns))
            continue
        if not (
            isinstance(statement, exp.Create)
            and statement.args.get("kind") == "TABLE"
            and isinstance(statement.this, exp.Schema)
        ):
            continue
        table = statement.this.this.name.lower()
        columns: set[str] = set()
        for definition in statement.this.expressions:
            if isinstance(definition, exp.ColumnDef):
                column = definition.this.name.lower()
                columns.add(column)
                for constraint in definition.args.get("constraints") or []:
                    if isinstance(
                        constraint.kind,
                        (exp.PrimaryKeyColumnConstraint, exp.UniqueColumnConstraint),
                    ):
                        unique_keys[table].add((column,))
            elif isinstance(definition, exp.PrimaryKey):
                unique_keys[table].add(
                    tuple(item.name.lower() for item in definition.expressions)
                )
            elif isinstance(definition, exp.UniqueColumnConstraint):
                target = definition.this
                expressions = (
                    target.expressions if isinstance(target, exp.Schema) else []
                )
                unique_keys[table].add(tuple(item.name.lower() for item in expressions))
        tables[table] = columns

    for match in ALTER_UNIQUE.finditer(schema_sql):
        table = (match.group(1) or match.group(2)).lower()
        columns = tuple(
            item.strip().strip('"').split()[0].lower()
            for item in match.group(3).split(",")
        )
        unique_keys[table].add(columns)
    return tables, unique_keys


def resolve_catalog_name(name: str, candidates: Iterable[str]) -> str | None:
    normalized = name.lower()
    available = set(candidates)
    if normalized in available:
        return normalized
    prefix_matches = [candidate for candidate in available if candidate.startswith(normalized)]
    return prefix_matches[0] if len(prefix_matches) == 1 else None


def apply_schema_patches(
    connection: sqlite3.Connection, app: str, schema_sql: str
) -> tuple[str, dict[str, int], list[dict[str, Any]]]:
    rows = connection.execute(
        "SELECT patch_type, patch_table_name, patch_columns_name, patch_reference "
        "FROM wtune_schema_patches WHERE patch_app = ? ORDER BY rowid",
        (app,),
    ).fetchall()
    tables, unique_keys = schema_catalog(schema_sql)
    statements: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    skipped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    priority = {"NOT_NULL": 0, "UNIQUE": 1, "FOREIGN_KEY": 2}
    for patch_type, table, columns_text, reference in sorted(
        rows, key=lambda row: priority.get(row[0], 3)
    ):
        if patch_type not in priority:
            continue
        identity = (patch_type, table, columns_text, reference or "")
        if identity in seen:
            continue
        seen.add(identity)

        resolved_table = resolve_catalog_name(table, tables)
        if resolved_table is None:
            skipped.append(
                {"type": patch_type, "table": table, "columns": columns_text, "reason": "table_not_found"}
            )
            continue
        columns = [
            resolve_catalog_name(column.strip(), tables[resolved_table])
            for column in columns_text.split(",")
            if column.strip()
        ]
        if not columns or any(column is None for column in columns):
            skipped.append(
                {"type": patch_type, "table": table, "columns": columns_text, "reason": "column_not_found"}
            )
            continue

        table_sql = quoted_name(resolved_table)
        columns_sql = ", ".join(quoted_name(column) for column in columns)
        columns_key = tuple(columns)

        if patch_type == "NOT_NULL":
            for column in columns:
                statements.append(
                    f"ALTER TABLE {table_sql} ALTER COLUMN {quoted_name(column)} SET NOT NULL;"
                )
            counts[patch_type] += 1
        elif patch_type == "UNIQUE":
            digest = f"{app}:unique:{resolved_table}:{columns_text}".encode()
            name = "wetune_u_" + hashlib.sha1(digest).hexdigest()[:16]
            statements.append(
                f"ALTER TABLE {table_sql} ADD CONSTRAINT {quoted_name(name)} "
                f"UNIQUE ({columns_sql});"
            )
            unique_keys[resolved_table].add(columns_key)
            counts[patch_type] += 1
        elif patch_type == "FOREIGN_KEY":
            referenced_table, referenced_column = reference.rsplit(".", 1)
            resolved_reference_table = resolve_catalog_name(referenced_table, tables)
            if resolved_reference_table is None:
                skipped.append(
                    {"type": patch_type, "table": table, "columns": columns_text, "reference": reference, "reason": "reference_table_not_found"}
                )
                continue
            resolved_reference_column = resolve_catalog_name(
                referenced_column, tables[resolved_reference_table]
            )
            if resolved_reference_column is None:
                skipped.append(
                    {"type": patch_type, "table": table, "columns": columns_text, "reference": reference, "reason": "reference_column_not_found"}
                )
                continue
            reference_key = (resolved_reference_column,)
            if reference_key not in unique_keys[resolved_reference_table]:
                skipped.append(
                    {"type": patch_type, "table": table, "columns": columns_text, "reference": reference, "reason": "reference_not_unique"}
                )
                continue
            digest = (
                f"{app}:foreign:{resolved_table}:{columns_text}:"
                f"{resolved_reference_table}.{resolved_reference_column}"
            ).encode()
            name = "wetune_fk_" + hashlib.sha1(digest).hexdigest()[:16]
            statements.append(
                f"ALTER TABLE {table_sql} ADD CONSTRAINT {quoted_name(name)} "
                f"FOREIGN KEY ({columns_sql}) REFERENCES {quoted_name(resolved_reference_table)} "
                f"({quoted_name(resolved_reference_column)});"
            )
            counts[patch_type] += 1

    if statements:
        schema_sql = schema_sql.rstrip() + "\n\n-- WeTune schema patches\n" + "\n".join(statements) + "\n"
    return schema_sql, dict(sorted(counts.items())), skipped


def import_app(
    connection: sqlite3.Connection,
    schema_dir: Path,
    output_root: Path,
    app: str,
) -> dict[str, Any]:
    source_schema_path = schema_dir / f"{app}.base.schema.sql"
    source_schema = source_schema_path.read_text(encoding="utf-8")
    dialect = detect_dialect(source_schema)
    rows = connection.execute(
        "SELECT stmt_id, stmt_raw_sql FROM wtune_stmts "
        "WHERE stmt_app_name = ? ORDER BY stmt_id",
        (app,),
    ).fetchall()

    source_queries: dict[int, str] = {}
    translated_queries: dict[int, str] = {}
    repairs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for statement_id, query in rows:
        source_queries[statement_id] = " ".join(query.splitlines())
        try:
            translated, repair = translate_query(query, dialect)
            translated_queries[statement_id] = translated
            if repair:
                repairs.append({"stmt_id": statement_id, "repair": repair})
        except ValueError as error:
            failures.append({"stmt_id": statement_id, "error": str(error)})

    output_dir = output_root / app
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source.sql").write_text(
        render_lines(source_queries), encoding="utf-8"
    )
    (output_dir / "cases.sql").write_text(
        render_lines(translated_queries), encoding="utf-8"
    )
    converted_schema = (
        postgres_schema(source_schema) if dialect == "postgres" else mysql_schema(source_schema)
    )
    converted_schema, patch_counts, skipped_patches = apply_schema_patches(
        connection, app, converted_schema
    )
    (output_dir / "schema.sql").write_text(converted_schema, encoding="utf-8")

    metadata = {
        "kind": "wetune_workload",
        "application": app,
        "source_dialect": dialect,
        "statement_count": len(rows),
        "translated_count": len(translated_queries),
        "maximum_statement_id": max(source_queries, default=0),
        "schema_patches": patch_counts,
        "skipped_schema_patches": skipped_patches,
        "repairs": repairs,
        "translation_failures": failures,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path, help="WeTune wtune.db")
    parser.add_argument("schema_dir", type=Path, help="WeTune schemas directory")
    parser.add_argument("output_root", type=Path, help="pgORCA corpus output directory")
    parser.add_argument(
        "--app", action="append", default=[], help="application to import; repeatable"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
        available = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT stmt_app_name FROM wtune_stmts "
                "WHERE stmt_app_name NOT LIKE '%_tmp' ORDER BY stmt_app_name"
            )
        ]
        applications = args.app or available
        missing = sorted(set(applications) - set(available))
        if missing:
            raise ValueError(f"unknown application(s): {', '.join(missing)}")

        summaries = [
            import_app(connection, args.schema_dir, args.output_root, app)
            for app in applications
        ]
    except (OSError, sqlite3.Error, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    total = sum(summary["statement_count"] for summary in summaries)
    translated = sum(summary["translated_count"] for summary in summaries)
    failures = total - translated
    print(
        f"imported {translated}/{total} statements from {len(summaries)} applications "
        f"({failures} translation failures)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
