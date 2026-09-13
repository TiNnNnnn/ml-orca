"""Native-rule replacement evidence contract shared with E2E tests."""
import re

XFORM_RE = re.compile(r"CXform[A-Za-z0-9_]+")
REPLACEMENT_STATES = ("native", "shadow", "negative", "replacement")
PROVENANCE_SOURCES = {"input", "dsl", "dphyper", "dsl+dphyper", "native"}

def validate_replacement_matrix(expectation: dict[str, object]) -> None:
    """Validate the causal four-state contract declared by an E2E case."""
    replacement = expectation.get("replacement")
    if replacement is None:
        return
    if not isinstance(replacement, dict):
        raise ValueError("replacement must be an object")

    xforms = replacement.get("xforms")
    if (
        not isinstance(xforms, list)
        or not xforms
        or any(
            not isinstance(name, str) or not XFORM_RE.fullmatch(name)
            for name in xforms
        )
        or len(set(xforms)) != len(xforms)
    ):
        raise ValueError("replacement.xforms must contain unique native xform names")

    exclusions = replacement.get("excluded_native_domains", [])
    if (
        not isinstance(exclusions, list)
        or any(not isinstance(item, str) or not item.strip() for item in exclusions)
        or len(set(exclusions)) != len(exclusions)
    ):
        raise ValueError(
            "replacement.excluded_native_domains must contain unique descriptions"
        )

    plans = expectation.get("plans")
    if not isinstance(plans, list):
        raise ValueError("replacement matrix requires plans")
    by_name = {
        plan.get("name"): plan
        for plan in plans
        if isinstance(plan, dict) and isinstance(plan.get("name"), str)
    }
    missing = [name for name in REPLACEMENT_STATES if name not in by_name]
    if missing:
        raise ValueError(f"replacement matrix missing states: {', '.join(missing)}")

    target = set(xforms)
    expected_dsl = {
        "native": False,
        "shadow": True,
        "negative": False,
        "replacement": True,
    }
    for state in REPLACEMENT_STATES:
        plan = by_name[state]
        if bool(plan.get("dsl", True)) != expected_dsl[state]:
            raise ValueError(f"replacement state {state} has the wrong DSL setting")
        disabled = set(plan.get("disable_xforms", []))
        should_disable = state in {"negative", "replacement"}
        if target.issubset(disabled) != should_disable:
            action = "disable" if should_disable else "enable"
            raise ValueError(
                f"replacement state {state} must {action} every target xform"
            )
        provenance = plan.get("provenance")
        if state == "replacement" and provenance is None:
            raise ValueError("replacement state must require memo provenance")
        if provenance is not None:
            if not isinstance(provenance, dict):
                raise ValueError("plan provenance must be an object")
            required_sources = provenance.get("required_sources", [])
            required_origins = provenance.get("required_origins", [])
            forbidden_origins = provenance.get("forbidden_origins", [])
            if (
                not isinstance(required_sources, list)
                or any(
                    not isinstance(source, str)
                    or source not in PROVENANCE_SOURCES
                    for source in required_sources
                )
                or not isinstance(required_origins, list)
                or not isinstance(forbidden_origins, list)
                or any(
                    not isinstance(name, str) or not XFORM_RE.fullmatch(name)
                    for name in [*required_origins, *forbidden_origins]
                )
                or not (required_sources or required_origins)
            ):
                raise ValueError("invalid or empty plan provenance contract")
            if state == "replacement" and (
                "dsl" not in required_sources
                or not target.issubset(set(forbidden_origins))
            ):
                raise ValueError(
                    "replacement provenance must require DSL and forbid every target xform"
                )

    rows = expectation.get("rows")
    if not isinstance(rows, dict) or not target.issubset(
        set(rows.get("disable_xforms", []))
    ):
        raise ValueError("replacement rows must run with every target xform disabled")
