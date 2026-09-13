#!/usr/bin/env python3
"""Audit structurally identical ORCA replacement rules.

An identical operator skeleton can be either a real expression/metadata
rewrite or a view-materialization bridge whose hidden lowering is performed
by C++.  Require the rule file to say which one it is so bridge evidence is
never mistaken for an independent native-xform replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from ml_orca.common.paths import TEST_ASSETS
from pathlib import Path
from typing import TypeAlias


Skeleton: TypeAlias = tuple[str, tuple["Skeleton", ...]]
CLASSIFICATION_PREFIXES = ("# BRIDGE(", "# DIRECT(")


@dataclass(frozen=True)
class ClassifiedIdentity:
    line: int
    classification: str
    skeleton: Skeleton


def operator_skeleton(expression: str) -> Skeleton:
    """Return only operator names and child topology from one DSL expression."""

    cursor = 0

    def parse_identifier() -> str:
        nonlocal cursor
        start = cursor
        while cursor < len(expression) and (
            expression[cursor].isalnum() or expression[cursor] in "_*"
        ):
            cursor += 1
        if cursor == start:
            raise ValueError(f"expected operator at column {cursor + 1}")
        return expression[start:cursor]

    def skip_bindings() -> None:
        nonlocal cursor
        if cursor >= len(expression) or expression[cursor] != "<":
            return
        depth = 1
        cursor += 1
        while cursor < len(expression) and depth:
            if expression[cursor] == "<":
                depth += 1
            elif expression[cursor] == ">":
                depth -= 1
            cursor += 1
        if depth:
            raise ValueError("unterminated operator binding")

    def parse_node() -> Skeleton:
        nonlocal cursor
        operator = parse_identifier()
        skip_bindings()
        children: list[Skeleton] = []
        if cursor < len(expression) and expression[cursor] == "(":
            cursor += 1
            while True:
                children.append(parse_node())
                if cursor < len(expression) and expression[cursor] == ",":
                    cursor += 1
                    continue
                if cursor >= len(expression) or expression[cursor] != ")":
                    raise ValueError(f"expected ')' at column {cursor + 1}")
                cursor += 1
                break
        return operator, tuple(children)

    skeleton = parse_node()
    if cursor != len(expression):
        raise ValueError(f"unexpected text at column {cursor + 1}")
    return skeleton


def audit_rule_text(text: str) -> tuple[list[ClassifiedIdentity], list[str]]:
    comments: list[str] = []
    identities: list[ClassifiedIdentity] = []
    errors: list[str] = []

    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            comments = []
            continue
        if line.startswith("#"):
            comments.append(line)
            continue

        fields = line.split("|")
        if len(fields) < 3:
            errors.append(f"line {line_number}: malformed RuleIR")
            comments = []
            continue

        try:
            source = operator_skeleton(fields[0])
            target = operator_skeleton(fields[1])
        except ValueError as error:
            errors.append(f"line {line_number}: {error}")
            comments = []
            continue

        classifications = [
            comment
            for comment in comments
            if comment.startswith(CLASSIFICATION_PREFIXES)
        ]
        if source == target:
            if len(classifications) != 1:
                errors.append(
                    f"line {line_number}: identical operator skeleton requires "
                    "exactly one immediate BRIDGE(...) or DIRECT(...) tag"
                )
            else:
                classification = classifications[0][2:].split(":", 1)[0]
                identities.append(
                    ClassifiedIdentity(line_number, classification, source)
                )
        elif any(tag.startswith("# BRIDGE(") for tag in classifications):
            errors.append(
                f"line {line_number}: BRIDGE tag requires identical operator skeletons"
            )
        comments = []

    return identities, errors


def audit_rule_file(path: Path) -> tuple[list[ClassifiedIdentity], list[str]]:
    return audit_rule_text(path.read_text(encoding="utf-8"))


def main() -> int:
    rule_file = TEST_ASSETS / "rules" / "orca_replacements.rules"
    identities, errors = audit_rule_file(rule_file)
    if errors:
        for error in errors:
            print(error)
        return 1

    bridge_count = sum(
        identity.classification.startswith("BRIDGE(") for identity in identities
    )
    direct_count = len(identities) - bridge_count
    print(
        f"classified {len(identities)} identical-skeleton rules: "
        f"{bridge_count} BRIDGE, {direct_count} DIRECT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
