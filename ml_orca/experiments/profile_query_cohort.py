#!/usr/bin/env python3
"""Sample literal-normalized query families, then inspect their rule placement traces."""

from __future__ import annotations

from ml_orca.common.paths import PGORCA_ROOT

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import zlib

import sqlglot
from sqlglot import exp

from ml_orca.experiments.plot_stats_sweep import chinese_plotting
from ml_orca.collect.run_workload_comparison import distribution, experiment_run_status, profile_comparability


ROOT = PGORCA_ROOT


def query_features(sql: str) -> tuple[str, dict]:
    statements = sqlglot.parse(sql, read="postgres")
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("one query statement is required")
    tree = statements[0]
    counts = Counter(node.key for node in tree.walk())
    normalized = tree.transform(lambda node:
        exp.Literal.string("__literal__") if isinstance(node, exp.Literal) and node.is_string
        else exp.Literal.number("0") if isinstance(node, exp.Literal)
        else exp.Boolean(this=False) if isinstance(node, exp.Boolean) else node)
    # ponytail: masks literals only; source-template provenance is needed to group alias/structural variants.
    return normalized.sql(dialect="postgres", comments=False), dict(counts)


def build_cohort(root: Path, workloads: list[str], count: int, seed: int, anchors: list[str]) -> dict:
    if count < 0:
        raise ValueError("count must be nonnegative")
    entries, sources, identities = [], {}, {}
    for workload in workloads:
        base = root / workload
        sources[workload] = json.loads((base / "manifest.json").read_text())
        groups = defaultdict(list)
        for path in sorted((base / "sql").glob("*.sql")):
            raw = path.read_bytes()
            entry = {"query": f"{workload}/{path.stem}", "workload": workload,
                     "query_crc32": f"{zlib.crc32(raw):08x}", "selected": False}
            try:
                canonical, features = query_features(raw.decode("utf-8"))
            except (ValueError, sqlglot.errors.ParseError) as error:
                entries.append({**entry, "status": "parse_error", "error": str(error)})
                continue
            identity = workload + "\n" + canonical
            family = f"{zlib.crc32(identity.encode()):08x}"
            if family in identities and identities[family] != identity:
                raise ValueError("family CRC32 collision; cannot safely partition this corpus")
            identities[family] = identity
            entry.update(status="ok", family=family, features=features)
            groups[family].append(entry)
            entries.append(entry)
        rng = random.Random(f"{seed}:{workload}")
        families = sorted(groups)
        anchor_families = {f for f, rows in groups.items() if any(r["query"] in anchors for r in rows)}
        families = [f for f in families if f not in anchor_families]
        rng.shuffle(families)
        # A declared validation budget, not a theorem about generalization accuracy.
        holdout = set(families[:max(1, len(families) // 5)]) if len(families) > 1 else set()
        discovery = [f for f in families if f not in holdout]
        for family, rows in groups.items():
            split = "anchor" if family in anchor_families else "holdout" if family in holdout else "discovery"
            for row in rows:
                row["split"] = split
                row["selected"] = row["query"] in anchors
        for family in discovery[:count]:
            rng.choice(groups[family])["selected"] = True
    if set(anchors) - {r["query"] for r in entries if r["status"] == "ok"}:
        raise ValueError("anchor missing or not parseable")
    return {"schema_version": 1, "seed": seed, "sqlglot_version": sqlglot.__version__,
            "workload_root": str(root.resolve()), "count_per_workload": count,
            "sampling_unit": "uniform_discovery_family_then_uniform_member",
            "split_scope": "literal_normalized_family_within_workload",
            "not_guaranteed": ["generator_template_independence", "schema_generalization",
                               "production_query_distribution"],
            "sources": sources, "queries": entries}


def cohort_results(cohort: dict, results: Path) -> list[dict]:
    rows = []
    for query in cohort["queries"]:
        if not query["selected"]:
            continue
        path = results / query["query"] / "comparison.json"
        comparison = json.loads(path.read_text()) if path.is_file() else {}
        profile = comparison.get("rule_profile") or {}
        baseline = next((s for s in profile.get("scenarios", []) if s["stats_experiment"] is None), {})
        for arm in profile.get("arms", ("off", "rbo", "cbo")):
            run = baseline.get("arms", {}).get(arm, {})
            off = baseline.get("arms", {}).get("off", {})
            status = experiment_run_status(run, False) if run else "missing"
            if comparison and comparison.get("query_crc32") != query["query_crc32"]:
                status = "query_identity_mismatch"
            memo = run.get("dsl_observability", {}).get("memo") or {}
            off_memo = off.get("dsl_observability", {}).get("memo") or {}
            if status == "ok" and ("group_expressions" not in memo or "group_expressions" not in off_memo):
                status = "incomplete_trace"
            valid = status == "ok" and profile_comparability(off, run, False)["result_comparable"]
            if status == "ok" and not valid:
                status = "off_not_comparable"
            statuses = Counter(run.get("profile_rule_statuses", []))
            applied = sum(statuses.get(s, 0) for s in ("applied", "applied_rbo"))
            rows.append({"query": query["query"], "family": query["family"], "split": query["split"],
                         "context": {key: query.get("features", {}).get(key, 0)
                                     for key in ("table", "join", "select", "group", "window", "union", "exists")}
                         if "features" in query else None,
                         "arm": arm, "rule_hash": profile.get("rule_hash"), "status": status,
                         "result_empty": run.get("rows_hash") == hashlib.sha256(b"").hexdigest() if valid else None,
                         "target_statuses": dict(statuses), "target_applied": applied if valid else None,
                         "memo_expressions": memo.get("group_expressions") if valid else None,
                         "delta_memo_expressions": memo["group_expressions"] - off_memo["group_expressions"]
                         if valid else None})
    if len({r["rule_hash"] for r in rows if r["rule_hash"] is not None}) > 1:
        raise ValueError("cannot pool different target rules in one response report")
    return rows


def incremental_cohort(cohort: dict, previous: dict) -> dict:
    """Extend a frozen sample without rerunning it or changing its holdout."""
    if previous.get("selection") is not None:
        raise ValueError("previous cohort must be cumulative, not an incremental batch")
    old = {q["query"]: q for q in previous["queries"]}
    if (cohort["seed"] != previous["seed"] or len(old) != len(cohort["queries"])
            or any(q["query"] not in old or any(q.get(k) != old[q["query"]].get(k)
                for k in ("query_crc32", "family", "split", "status")) for q in cohort["queries"])
            or any(old[q["query"]]["selected"] and not q["selected"] for q in cohort["queries"])):
        raise ValueError("expansion must preserve the corpus, split and previous selections")
    return {**cohort, "selection": "incremental_discovery",
            "previous_count_per_workload": previous["count_per_workload"],
            "queries": [{**q, "selected": q["selected"] and not old[q["query"]]["selected"]}
                        for q in cohort["queries"]]}


def stratified_cohort(cohort: dict, per_stratum: int) -> dict:
    """SRS without replacement inside pre-outcome AST strata of unseen families."""
    if type(per_stratum) is not int or per_stratum < 1 or cohort.get("selection") is not None:
        raise ValueError("positive per-stratum count and a cumulative cohort are required")
    queries = [{**q, "selected": False} for q in cohort["queries"]]
    excluded = {q.get("family") for q in cohort["queries"] if q["selected"]}
    strata = defaultdict(lambda: defaultdict(list))
    for query in queries:
        if query.get("split") != "discovery" or query["family"] in excluded:
            continue
        f = query["features"]
        # These are SQL syntax strata, not assertions about post-preprocess
        # scalar subqueries or rule applicability. CTE blocks also count.
        key = f"{query['workload']}:blocks{int(f.get('select', 0) > 1)}:group{int(f.get('group', 0) > 0)}:window{int(f.get('window', 0) > 0)}"
        query["stratum"] = key
        strata[key][query["family"]].append(query)
    family_strata = defaultdict(set)
    for key, families in strata.items():
        for family in families:
            family_strata[family].add(key)
    if any(len(keys) != 1 for keys in family_strata.values()):
        raise ValueError("a literal-normalized family crosses syntax strata")
    design = []
    for key, families in sorted(strata.items()):
        size, take = len(families), min(per_stratum, len(families))
        rng = random.Random(f"strata-v1:{cohort['seed']}:{key}")
        for family in rng.sample(sorted(families), take):
            members = sorted(families[family], key=lambda q: q["query"])
            query = rng.choice(members)
            query.update(selected=True, family_inclusion_probability=take / size,
                         member_probability=1 / len(members), sql_inclusion_probability=take / size / len(members))
        design.append({"stratum": key, "families": size, "sampled_families": take})
    return {**cohort, "queries": queries, "selection": "stratified_remaining_discovery",
            "sampling_unit": "stratified_discovery_family_then_uniform_member", "strata": design,
            "eligible_families": sum(s["families"] for s in design), "per_stratum": per_stratum,
            "estimand": "equal_family_mean_over_remaining_discovery_member_means",
            "stratification": ["multiple_select_blocks", "group_by_present", "window_present"],
            "excluded_prior_families": sorted(f for f in excluded if f is not None)}


def rule_distribution(rows: list[dict]) -> dict:
    """Aggregate rule evidence, without using query identities as factors or pooling anchors."""
    groups = []
    for split in sorted({r["split"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows}):
            subset = [r for r in rows if r["split"] == split and r["arm"] == arm]
            valid = [r for r in subset if r["status"] == "ok"]
            groups.append({"split": split, "arm": arm, "attempts": len(subset),
                           "statuses": dict(sorted(Counter(r["status"] for r in subset).items())),
                           "comparable": len(valid),
                           "applied_samples": sum(r["target_applied"] > 0 for r in valid),
                           "empty_result_samples": sum(r["result_empty"] is True for r in valid),
                           "memo_delta": distribution([r["delta_memo_expressions"] for r in valid])})
    return {"rule_hashes": sorted({r["rule_hash"] for r in rows if r["rule_hash"] is not None}),
            "context_scope": "pre_optimization_sql_ast_counts_not_binding_local_statistics",
            "sample_measure": "declared_discovery_family_sample_anchors_separate",
            "generalization_validated": False, "groups": groups}


def local_input_records(comparison: dict) -> list[dict]:
    profile = comparison.get("rule_profile")
    if not profile:
        raise ValueError("local input report requires a target rule profile")
    records = []
    for scenario in profile["scenarios"]:
        for arm, run in scenario["arms"].items():
            for observation in run.get("rule_input_observations", []):
                if observation["rule_hash"] != profile["rule_hash"]:
                    continue
                context = observation["input_context"]
                if context.get("capture") != "before_evaluation":
                    raise ValueError("input features must be captured before evaluation")
                records.append({**observation, "arm": arm,
                                "stats_experiment": scenario["stats_experiment"],
                                "run_status": experiment_run_status(run, scenario["stats_experiment"] is not None)})
    return records


def placement_evidence(comparison: dict) -> dict:
    """Paired RBO-minus-CBO effects within each randomized measurement block."""
    profile = comparison.get("rule_profile") or {}
    timing = profile.get("timing") or {}
    repeats = timing.get("repeats")
    if (timing.get("source") != "untraced_explain_analyze"
            or timing.get("design") != "randomized_complete_blocks"
            or type(repeats) is not int or repeats < 1):
        raise ValueError("placement evidence requires randomized repeated timing blocks")
    scenarios = profile["scenarios"]
    if any("rbo" not in scenario["arms"] for scenario in scenarios):
        raise ValueError("placement comparison requires an RBO arm; OFF/CBO measures contribution, not placement")
    samples = {}
    for sample in timing["samples"]:
        if sample["phase"] != "measurement" or sample["arm"] not in ("rbo", "cbo"):
            continue
        key = sample["scenario"], sample["block"], sample["arm"]
        if (type(key[0]) is not int or not 0 <= key[0] < len(scenarios)
                or type(key[1]) is not int or not 0 <= key[1] < repeats or key in samples):
            raise ValueError("duplicate or invalid measurement block")
        samples[key] = sample
    rows = []
    for index, scenario in enumerate(scenarios):
        diagnostics = scenario["arms"]
        check = profile_comparability(diagnostics["cbo"], diagnostics["rbo"],
                                      scenario["stats_experiment"] is not None, "cbo")
        pairs = []
        for block in range(repeats):
            reasons = list(check["exclusions"])
            values = {}
            for arm in ("rbo", "cbo"):
                sample = samples.get((index, block, arm))
                if sample is None:
                    reasons.append(f"{arm}:missing_sample")
                    continue
                if sample.get("status") != "ok":
                    reasons.append(f"{arm}:{sample.get('status', 'unknown')}")
                if sample.get("diagnostic_plan_matches") is not True:
                    reasons.append(f"{arm}:diagnostic_plan_mismatch")
                if (sample.get("stats_experiment") != scenario["stats_experiment"]
                        or sample.get("diagnostic_rows_hash") != diagnostics[arm].get("rows_hash")):
                    reasons.append(f"{arm}:diagnostic_identity_mismatch")
                values[arm] = [sample.get(field) for field in ("planning_ms", "execution_ms")]
                if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values[arm]):
                    reasons.append(f"{arm}:invalid_timing")
            delta = None
            if not reasons:
                planning, execution = [r - c for r, c in zip(values["rbo"], values["cbo"])]
                delta = {"planning_ms": planning, "execution_ms": execution,
                         "total_ms": planning + execution}
            pairs.append({"block": block, "exclusions": reasons, "delta": delta})
        memos = [(diagnostics[a].get("dsl_observability", {}).get("memo") or {}).get("group_expressions")
                 for a in ("rbo", "cbo")]
        rows.append({"scenario": index, "stats_experiment": scenario["stats_experiment"],
                     "stats_targets": diagnostics["cbo"].get("experiment_outcomes", [{}])[0].get("stats_targets", [])
                     if diagnostics["cbo"].get("experiment_outcomes") else [],
                     "attempted_pairs": repeats, "valid_pairs": sum(p["delta"] is not None for p in pairs),
                     "pairs": pairs,
                     "delta_memo_expressions": memos[0] - memos[1]
                     if check["result_comparable"] and all(type(m) is int and m >= 0 for m in memos) else None,
                     "deltas": {field: distribution([p["delta"][field] for p in pairs if p["delta"] is not None])
                                for field in ("planning_ms", "execution_ms", "total_ms")}})
    return {"rule_hash": profile["rule_hash"], "baseline": "cbo", "contrast": "rbo_minus_cbo",
            "negative_means": "rbo_lower_cost", "generalization_validated": False,
            "automatic_migration": False, "timing_validation": "separate_diagnostic_run",
            "scenarios": rows}


def render_placement(evidence: dict, output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    fig, panels = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    rows = evidence["scenarios"]
    labels = ["未注入" if r["stats_experiment"] is None else
              f"注入 {r['stats_targets'][0]['requested_rows']:g} 行" if len(r["stats_targets"]) == 1
              and type(r["stats_targets"][0].get("requested_rows")) in (int, float)
              else f"统计配置 {r['scenario']}" for r in rows]
    for panel, field, title in zip(panels, ("planning_ms", "execution_ms", "total_ms"),
                                   ("规划时间差", "执行时间差", "规划＋执行时间差")):
        for index, row in enumerate(rows):
            values = [p["delta"][field] for p in row["pairs"] if p["delta"] is not None]
            panel.scatter([index] * len(values), values, color="#4477aa", alpha=0.6)
            if values:
                panel.scatter(index, statistics.median(values), color="#ee7733", marker="_", s=180)
        panel.set_xticks(range(len(rows)), labels, rotation=30)
        panel.axhline(0, color="gray", linewidth=0.8)
        panel.set_title(title)
        panel.set_ylabel("预改写 − 代价搜索（毫秒）；负值更快")
        panel.grid(alpha=0.2)
    valid = sum(r["valid_pairs"] for r in rows)
    attempted = sum(r["attempted_pairs"] for r in rows)
    fig.suptitle(f"规则 {evidence['rule_hash']}｜默认代价搜索，评估提前改写的收益\n"
                 f"同统计配置、同随机区组配对：{valid}/{attempted} 对有效；蓝点为实测差值，橙线为中位数\n"
                 "单个锚点的局部证据；不是总体规则画像，不自动迁移规则")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)


def render_local_inputs(records: list[dict], output: Path, font: Path) -> None:
    if not records:
        raise ValueError("no local input records; enable trace and an observation stats configuration")
    plt = chinese_plotting(font)
    from matplotlib.ticker import MaxNLocator
    external = any("external_reference" in r["input_context"]["root"] for r in records)
    fig, panels = plt.subplots(2 if external else 1, 2, figsize=(11, 8 if external else 5), layout="constrained")
    arms = [a for a in ("rbo", "cbo") if any(r["arm"] == a for r in records)]
    labels = ["预改写阶段" if a == "rbo" else "代价搜索阶段" for a in arms]
    for index, panel in enumerate(panels.flat):
        role = "children" if index % 2 else "root"
        categories = (("available", "冻结参考可关联（不是当前缓存）", "#4477aa"),
                      ("unmatched", "参考未匹配或缺失", "#cc6677"),
                      ("ambiguous", "参考有歧义", "#ee7733"),
                      ("result_not_verified", "结果未核对通过", "#999999")) if index >= 2 else (
                      ("available", "已有缓存基数", "#228833"), ("missing", "基数缓存缺失（不是零行）", "#cc6677"))
        bottom = [0] * len(arms)
        for status, label, color in categories:
            values = []
            for arm in arms:
                contexts = [r["input_context"] for r in records if r["arm"] == arm]
                nodes = [c["root"] for c in contexts] if role == "root" else [
                    child["node"] for c in contexts for child in c["children"]]
                statuses = [n.get("external_reference", {}).get("status", "unmatched") if index >= 2
                            else "available" if n["rows"] is not None else "missing" for n in nodes]
                values.append(statuses.count(status))
            panel.bar(labels, values, bottom=bottom, label=label, color=color)
            bottom = [b + v for b, v in zip(bottom, values)]
        panel.set_title(("来源根节点" if role == "root" else "直接关系输入") +
                        ("的冻结参考统计" if index >= 2 else "的基数缓存"))
        panel.set_ylabel("已保留输入记录数（不是独立随机样本数）")
        panel.yaxis.set_major_locator(MaxNLocator(integer=True))
        panel.legend(fontsize=9)
    omitted = sum(r["input_context"]["omitted_children"] for r in records)
    incomplete = sum(r["run_status"] != "ok" for r in records)
    fig.suptitle(f"规则 {records[0]['rule_hash']}｜求值前局部输入的可观测性\n"
                 f"只读已有统计，不主动推导；省略子输入 {omitted} 个；来自不可比运行的记录 {incomplete} 条\n"
                 "代价搜索只保留规则／状态首次事件，预改写保留已有事件；不是触发概率分布" +
                 ("\n冻结参考在查询前加载，仅供外部分析，优化器本次未使用；缺失缓存仍保留" if external else ""))
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)


def render_rule_distribution(rows: list[dict], output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    from matplotlib.ticker import MaxNLocator
    summary = rule_distribution(rows)
    fig, panels = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    arm_labels = {"off": "禁用目标规则", "rbo": "预改写阶段", "cbo": "代价搜索阶段"}
    arms = [a for a in arm_labels if any(r["arm"] == a for r in rows)]
    labels = [arm_labels[a] for a in arms]
    discovery = [g for g in summary["groups"] if g["split"] == "discovery"]
    status_labels = {"ok": "可比运行", "plan_timeout": "计划／分析执行超时", "rows_timeout": "结果阶段超时",
                     "fallback": "回退原生规划器", "missing": "结果缺失"}
    status_colors = {"ok": "#228833", "plan_timeout": "#cc6677", "rows_timeout": "#ee7733",
                     "fallback": "#888888", "missing": "#cccccc"}
    bottom = [0] * len(arms)
    statuses = sorted({s for g in discovery for s in g["statuses"]})
    for status in statuses:
        values = [sum(g["statuses"].get(status, 0) for g in discovery if g["arm"] == a) for a in arms]
        panels[0].bar(labels, values, bottom=bottom, color=status_colors.get(status, "#aa4499"),
                      label=status_labels.get(status, "其他不可比状态"))
        bottom = [b + v for b, v in zip(bottom, values)]
    panels[0].set_title("规则实验覆盖（仅随机探索样本）")
    panels[0].set_ylabel("诊断运行数；不是目标规则导致的失败数")
    panels[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    if statuses:
        panels[0].legend(fontsize=8)
    for panel, field, title, ylabel in (
        (panels[1], "target_applied", "应用响应的观测点", "目标规则应用次数"),
        (panels[2], "delta_memo_expressions", "搜索响应的观测点", "相对禁用规则的表达式增量"),
    ):
        for arm, label, color in (("rbo", arm_labels["rbo"], "#ee7733"), ("cbo", arm_labels["cbo"], "#228833")):
            if arm not in arms:
                continue
            for split, name, marker in (("discovery", "探索", "o"), ("anchor", "锚点", "*")):
                subset = [r for r in rows if r["arm"] == arm and r["split"] == split
                          and r["status"] == "ok" and r["context"] is not None]
                if arm == "cbo":
                    marker = "x" if split == "discovery" else "+"
                panel.scatter([r["context"]["table"] for r in subset], [r[field] for r in subset],
                              label=label + "／" + name, marker=marker, color=color,
                              s=110 if split == "anchor" else 35, alpha=0.7)
        panel.set_title(title)
        panel.set_xlabel("输入表引用次数（未去重；不是查询编号）")
        panel.set_ylabel(ylabel)
        panel.xaxis.set_major_locator(MaxNLocator(integer=True))
        panel.yaxis.set_major_locator(MaxNLocator(integer=True))
        panel.grid(alpha=0.2)
        panel.legend(fontsize=8)
    identity = ", ".join(summary["rule_hashes"]) or "未知"
    fig.suptitle(f"规则 {identity}｜规则画像的初步采样证据\n"
                 "固定小数据，带跟踪诊断；失败不填零，锚点不混入探索分布\n"
                 "仅展示观测，不拟合曲线；其他输入因素未控制，尚未验证泛化")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)


def render_cohort(rows: list[dict], output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    queries = list(dict.fromkeys(r["query"] for r in rows))
    if not queries:
        raise ValueError("no selected query results")
    lookup = {(r["query"], r["arm"]): r for r in rows}
    labels = [q + ("（锚点）" if lookup[q, "off"]["split"] == "anchor" else "（探索）") for q in queries]
    arms = [a for a in ("off", "rbo", "cbo") if any(r["arm"] == a for r in rows)]
    matrix = [[-1 if lookup.get((q, a), {}).get("target_applied") is None else int(lookup[q, a]["target_applied"] > 0)
               for a in arms] for q in queries]
    fig, panels = plt.subplots(1, 2, figsize=(12, max(4, len(queries) * 0.48)), layout="constrained")
    colors = ListedColormap(["#aaaaaa", "#ddddff", "#228833"])
    shown = panels[0].imshow(matrix, cmap=colors, norm=BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3), aspect="auto")
    panels[0].set_xticks(range(len(arms)), [{"off": "禁用目标规则", "rbo": "预改写阶段", "cbo": "代价搜索阶段"}[a] for a in arms])
    panels[0].set_yticks(range(len(queries)), labels)
    panels[0].set_title("目标规则是否应用（未应用不等于不可匹配）")
    bar = fig.colorbar(shown, ax=panels[0], ticks=[-1, 0, 1], shrink=0.7)
    bar.ax.set_yticklabels(["失败或不可比", "未观察到应用", "观察到应用"])
    for arm, label, marker in (("rbo", "预改写阶段", "o"), ("cbo", "代价搜索阶段", "x")):
        if arm not in arms:
            continue
        subset = [(i, lookup.get((q, arm), {}).get("delta_memo_expressions")) for i, q in enumerate(queries)]
        panels[1].scatter([v for _, v in subset if v is not None],
                          [i for i, v in subset if v is not None], label=label, marker=marker)
    panels[1].set_yticks(range(len(queries)), labels)
    panels[1].invert_yaxis()
    panels[1].axvline(0, color="gray", linewidth=0.8)
    panels[1].set_xlabel("相对禁用目标规则的搜索空间表达式增量")
    panels[1].set_title("只比较同一查询；失败不填零")
    panels[1].legend()
    fig.suptitle("跨查询探索批次｜固定数据、无基数注入、带跟踪诊断\n"
                 "锚点与探索样本分开保存；未使用留出组；不代表生产查询分布")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-root", type=Path, default=ROOT / "test/dsl/workloads")
    parser.add_argument("--workload", action="append", choices=("tpch", "tpcds", "job", "sqlstorm"))
    parser.add_argument("--count", type=int, default=4, help="discovery families per workload, excluding anchors")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--anchor", action="append", default=[])
    parser.add_argument("--cohort", type=Path, help="read a prepared cohort for reporting")
    parser.add_argument("--previous-cohort", type=Path, help="select only new families while validating the frozen split")
    parser.add_argument("--stratified-per-stratum", type=int,
                        help="draw this many remaining families per syntax stratum; requires the cumulative previous cohort")
    parser.add_argument("--local-inputs", type=Path, help="comparison.json containing target-rule local input observations")
    parser.add_argument("--placement", type=Path, help="comparison.json with repeated timing; report RBO relative to CBO")
    parser.add_argument("--results", type=Path, help="runner output directory for reporting")
    parser.add_argument("--rule-view", action="store_true", help="show rule distributions and generic factors, not query IDs")
    parser.add_argument("--font", type=Path, default=ROOT / "output/fonts/NotoSansCJKsc-Regular.otf")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file, or plot stem when reporting")
    args = parser.parse_args()
    if args.previous_cohort and (args.cohort or args.local_inputs or args.placement or args.results):
        parser.error("--previous-cohort is only for creating an incremental sample")
    if args.stratified_per_stratum is not None and not args.previous_cohort:
        parser.error("stratification requires --previous-cohort to freeze exclusions and holdout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.placement:
        if args.local_inputs or args.cohort or args.results or args.rule_view:
            parser.error("--placement is a separate report mode")
        evidence = placement_evidence(json.loads(args.placement.read_text()))
        evidence["source"] = str(args.placement.resolve())
        args.output.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n")
        render_placement(evidence, args.output, args.font)
    elif args.local_inputs:
        if args.cohort or args.results or args.rule_view:
            parser.error("--local-inputs is a separate report mode")
        comparison = json.loads(args.local_inputs.read_text())
        records = local_input_records(comparison)
        args.output.with_suffix(".json").write_text(json.dumps({
            "source": str(args.local_inputs.resolve()), "records": records,
            "input_stats_reference": comparison.get("input_stats_reference"),
            "sampling_is_uniform": False}, indent=2) + "\n")
        render_local_inputs(records, args.output, args.font)
    elif args.cohort:
        if args.results is None:
            parser.error("--cohort requires --results")
        cohort = json.loads(args.cohort.read_text())
        rows = cohort_results(cohort, args.results)
        summary_path = args.results / "summary.json"
        fixture = json.loads(summary_path.read_text()).get("fixture") if summary_path.is_file() else None
        args.output.with_suffix(".json").write_text(json.dumps({"cohort": str(args.cohort.resolve()),
            "results": str(args.results.resolve()), "fixture": fixture,
            "rule_distribution": rule_distribution(rows), "rows": rows}, indent=2) + "\n")
        renderer = render_rule_distribution if args.rule_view else render_cohort
        renderer(rows, args.output, args.font)
    else:
        cohort = build_cohort(args.workload_root, args.workload or ["tpch", "tpcds", "job", "sqlstorm"],
                              args.count, args.seed, args.anchor)
        if args.previous_cohort:
            increment = incremental_cohort(cohort, json.loads(args.previous_cohort.read_text()))
            if args.stratified_per_stratum is not None:
                if any(q["selected"] for q in increment["queries"]):
                    parser.error("use the previous cumulative --count before stratifying the remaining frame")
                cohort = stratified_cohort(cohort, args.stratified_per_stratum)
            else:
                cohort = increment
            cohort["previous_cohort"] = str(args.previous_cohort.resolve())
        with args.output.open("x") as stream:
            stream.write(json.dumps(cohort, indent=2) + "\n")
        print(json.dumps({"queries": len(cohort["queries"]),
                          "selected": [q["query"] for q in cohort["queries"] if q["selected"]]}))


if __name__ == "__main__":
    main()
