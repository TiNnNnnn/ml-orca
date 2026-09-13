#!/usr/bin/env python3
"""Plot measured cardinality-sweep points, without interpolating unobserved responses."""

from __future__ import annotations

from ml_orca.common.paths import PGORCA_ROOT

import argparse
import json
import math
from pathlib import Path
from statistics import median

from ml_orca.collect.run_workload_comparison import experiment_run_status, results_equal, timing_schedule


PLAN_LABELS = {
    "identical": "计划相同", "join_order": "连接顺序变化",
    "expression_or_property": "表达式或属性变化", "physical_shape": "物理结构变化",
    "plan_error": "计划错误", "plan_timeout": "计划阶段超时",
    "rows_timeout": "结果阶段超时", "rows_error": "结果阶段错误",
    "server_failure": "数据库服务故障", "fallback": "回退到原生规划器",
    "unknown_optimizer": "优化器来源不明", "incomplete_trace": "跟踪记录不完整",
    "injection_unconsumed": "基数注入未消费", "unknown": "状态不明",
    "missing": "缺少实验结果", "manifest_target_mismatch": "配置与消费目标不一致",
    "baseline_not_comparable": "与基线不可比较",
}


def measured_points(manifest: dict, comparison: dict, manifest_dir: Path, mode: str) -> list[dict]:
    experiments = {}
    for experiment in comparison["stats_experiments"]:
        key = str(Path(experiment["path"]).resolve())
        if key in experiments:
            raise ValueError("duplicate experiment path; plot repeats separately")
        experiments[key] = experiment
    baseline = comparison["modes"][mode]
    points = []
    for index, point in enumerate(manifest["points"]):
        if any(len(point[field]) != len(manifest["targets"]) for field in ("factors", "requested_rows")):
            raise ValueError("each point must specify every target")
        experiment = experiments.get(str((manifest_dir / point["file"]).resolve()))
        run = experiment["modes"].get(mode, {}) if experiment else {}
        status = experiment_run_status(run, True) if experiment else "missing"
        outcome = run.get("experiment_outcomes", [{}])[0] if status == "ok" else {}
        if status == "ok":
            expected = {(t["fingerprint"], t["operator"]): rows
                        for t, rows in zip(manifest["targets"], point["requested_rows"])}
            actual = outcome["stats_targets"]
            if (len(actual) != len(expected)
                or {(t["fingerprint"], t["operator"]) for t in actual} != set(expected) or any(
                (t["fingerprint"], t["operator"]) not in expected
                or not math.isclose(t["requested_rows"], expected[t["fingerprint"], t["operator"]],
                                    rel_tol=1e-9, abs_tol=1e-6)
                for t in actual
            )):
                status = "manifest_target_mismatch"
            elif (experiment_run_status(baseline, False) != "ok"
                  or not results_equal(baseline, run)):
                status = "baseline_not_comparable"
        valid = status == "ok"
        points.append({
            **point, "index": index, "status": status,
            "plan_change": experiment["comparisons"][mode]["plan_comparison"] if valid else status,
            "optimizer_cost": outcome.get("optimizer_cost") if valid else None,
            "memo_expressions": outcome.get("memo_group_expressions") if valid else None,
            "binding_attempts": sum(r.get("binding_attempts", 0)
                                    for r in run["dsl_observability"]["rules"].values()) if valid else None,
            "rule_applications": outcome.get("rule_applications") if valid else None,
        })
    return points


def chinese_plotting(font: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if not font.is_file():
        raise ValueError("找不到中文字体，请通过 --font 指定中文字体文件")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"  # Keep Chinese readable without client fonts.
    return plt


def profile_points(manifest: dict, comparison: dict, manifest_dir: Path) -> dict:
    scenarios = comparison["rule_profile"]["scenarios"]
    baseline = next(s for s in scenarios if s["stats_experiment"] is None)
    if any(set(s["arms"]) != set(baseline["arms"]) for s in scenarios):
        raise ValueError("profile scenarios have inconsistent arms")
    adapted = {"modes": baseline["arms"], "stats_experiments": [
        {"path": s["stats_experiment"], "modes": s["arms"], "comparisons": {
            arm: {"plan_comparison": "not_evaluated"} for arm in s["arms"]}}
        for s in scenarios if s["stats_experiment"] is not None]}
    return {arm: measured_points(manifest, adapted, manifest_dir, arm) for arm in baseline["arms"]}


def render_profile(manifest: dict, comparison: dict, manifest_dir: Path, output: Path, font: Path) -> None:
    plt = chinese_plotting(font)
    collected = profile_points(manifest, comparison, manifest_dir)
    # This view plots numerical responses, not plan-change classes.
    fig, panels = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    metrics = (("optimizer_cost", "估算计划成本（不是执行耗时）"),
               ("memo_expressions", "搜索空间表达式数量"),
               ("binding_attempts", "规则绑定尝试次数"), ("rule_applications", "规则应用总次数"))
    invalid = 0
    for arm, label in (("off", "禁用目标规则"), ("rbo", "预改写阶段"), ("cbo", "代价搜索阶段")):
        if arm not in collected:
            continue
        points = collected[arm]
        for p in points:
            p.pop("plan_change")  # No fabricated identity classifications in exported data.
        invalid += sum(p["status"] != "ok" for p in points)
        for panel, (field, ylabel) in zip(panels.flat, metrics):
            ordered = sorted(points, key=lambda p: p["factors"][0])
            panel.plot([p["factors"][0] for p in ordered],
                       [p[field] if p[field] is not None else math.nan for p in ordered],
                       "o--", label=label)
            panel.set_xlabel("相对基准基数的倍率")
            panel.set_ylabel(ylabel)
            if field == "optimizer_cost":
                panel.set_yscale("log")
                panel.set_ylabel("估算计划成本（对数坐标；不是执行耗时）")
            panel.grid(alpha=0.2)
            panel.legend(fontsize=9)
    kind = "实测真实基数" if manifest["baseline_kind"] == "measured_actual_rows" else "原生估计基数"
    fig.suptitle(f"{comparison['workload']}/{comparison['query']}｜{kind}扰动与规则放置\n"
                 f"虚线仅连接实测点；无效运行 {invalid} 次；没有实际执行加速结论")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(collected, indent=2) + "\n")


def timing_points(manifest: dict, comparison: dict, manifest_dir: Path) -> list[dict]:
    profile = comparison["rule_profile"]
    diagnostics = profile_points(manifest, comparison, manifest_dir)
    timing = profile["timing"]
    if set(timing.get("arms", ("off", "rbo", "cbo"))) != set(diagnostics):
        raise ValueError("timing and diagnostic arms differ")
    expected = list(timing_schedule(len(profile["scenarios"]), timing["repeats"],
                                    timing["warmups"], timing["seed"], timing.get("arms", ("off", "rbo", "cbo"))))
    samples = timing["samples"]
    if [(s["phase"], s["block"], s["scenario"], s["arm"]) for s in samples] != expected:
        raise ValueError("timing samples do not match the complete randomized schedule")
    lookup = {(s["phase"], s["block"], s["scenario"], s["arm"]): s for s in samples}
    scenarios = {s["stats_experiment"]: i for i, s in enumerate(profile["scenarios"])}
    points = []
    for point_index, point in enumerate(manifest["points"]):
        index = scenarios[str((manifest_dir / point["file"]).resolve())]
        for arm in timing.get("arms", ("off", "rbo", "cbo")):
            warmups_ok = all(not s["comparison_exclusions"] for s in samples
                             if s["phase"] == "warmup" and s["scenario"] == index
                             and s["arm"] in ("off", arm))
            for block in range(timing["repeats"]):
                sample = lookup["measurement", block, index, arm]
                off = lookup["measurement", block, index, "off"]
                reasons = list(sample["comparison_exclusions"])
                for reference in ("off", arm):
                    if diagnostics[reference][point_index]["status"] != "ok":
                        reasons.append(reference + ":" + diagnostics[reference][point_index]["status"])
                if off["comparison_exclusions"]:
                    reasons.append("off_not_comparable")
                if not warmups_ok:
                    reasons.append("warmup_failed")
                points.append({"factors": point["factors"], "file": point["file"], "arm": arm,
                               "block": block, "sequence": sample["sequence"], "exclusions": reasons,
                               **{field: sample[field] if not reasons else None
                                  for field in ("planning_ms", "execution_ms")},
                               **{f"delta_{field}": sample[field] - off[field] if not reasons else None
                                  for field in ("planning_ms", "execution_ms")}})
    return points


def render_timing(manifest: dict, comparison: dict, manifest_dir: Path, output: Path, font: Path) -> None:
    points = timing_points(manifest, comparison, manifest_dir)
    plt = chinese_plotting(font)
    fig, panels = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    metrics = (("planning_ms", "规划耗时（毫秒；不是纯优化耗时）"),
               ("execution_ms", "执行耗时（毫秒；分析执行模式）"),
               ("delta_planning_ms", "同轮规划耗时差（毫秒；相对禁用规则）"),
               ("delta_execution_ms", "同轮执行耗时差（毫秒；相对禁用规则）"))
    for panel, (field, label) in zip(panels.flat, metrics):
        for arm, name, color in (("off", "禁用目标规则", "#4477aa"),
                                 ("rbo", "预改写阶段", "#ee7733"),
                                 ("cbo", "代价搜索阶段", "#228833")):
            if not any(p["arm"] == arm for p in points):
                continue
            if field.startswith("delta_") and arm == "off":
                continue
            valid = [p for p in points if p["arm"] == arm and p[field] is not None]
            panel.scatter([p["factors"][0] for p in valid], [p[field] for p in valid],
                          s=18, alpha=0.4, color=color)
            factors = sorted({p["factors"][0] for p in points if p["arm"] == arm})
            values = [[p[field] for p in valid if p["factors"][0] == f] for f in factors]
            panel.plot(factors, [median(v) if v else math.nan for v in values], "o--", color=color, label=name)
        if field.startswith("delta_"):
            panel.axhline(0, color="gray", linewidth=0.8)
        panel.set_xlabel("相对实测基准基数的倍率" if manifest["baseline_kind"] == "measured_actual_rows"
                         else "相对原生估计基数的倍率")
        panel.set_ylabel(label)
        if field == "execution_ms" and all(p[field] > 0 for p in points if p[field] is not None):
            panel.set_yscale("log")
            panel.set_ylabel(label + "；对数坐标")
        panel.grid(alpha=0.2)
        panel.legend(fontsize=9)
    timing = comparison["rule_profile"]["timing"]
    fig.suptitle(f"{comparison['workload']}/{comparison['query']}｜无跟踪的随机交错计时\n"
                 f"预热 {timing['warmups']} 轮，实测 {timing['repeats']} 轮；散点为样本，虚线连接中位数\n"
                 f"图内排除 {sum(bool(p['exclusions']) for p in points)} 个样本；差值不是相邻执行配对，也不是置信区间")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(points, indent=2) + "\n")


def render(points: list[dict], axes: tuple[int, int], labels: list[str], title: str,
           output: Path, font: Path) -> None:
    plt = chinese_plotting(font)

    fig, panels = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    locations, plans, costs, search = panels.flat
    designs = {"control": ("基准控制点", "*"), "one_at_a_time": ("单目标扰动", "o"),
               "common_scale": ("同倍率扰动", "s"), "log_latin_hypercube": ("随机拉丁超立方采样", "^")}
    colors = {"identical": "#238b45", "join_order": "#e8851a", "expression_or_property": "#377eb8"}
    for kind, (label, marker) in designs.items():
        subset = [p for p in points if p["design"] == kind]
        locations.scatter([p["factors"][axes[0]] for p in subset],
                          [p["factors"][axes[1]] for p in subset], marker=marker, s=75, label=label)
    for change in sorted({p["plan_change"] for p in points}):
        subset = [p for p in points if p["plan_change"] == change]
        plans.scatter([p["factors"][axes[0]] for p in subset],
                      [p["factors"][axes[1]] for p in subset], s=85,
                      color=colors.get(change, "#777777"), label=PLAN_LABELS.get(change, "未识别状态"))
    for panel in (locations, plans):
        panel.set_xscale("log", base=2)
        panel.set_yscale("log", base=2)
        panel.set_xlabel(labels[0] + "基数倍率")
        panel.set_ylabel(labels[1] + "基数倍率")
        for point in points:
            panel.annotate(str(point["index"]), [point["factors"][a] for a in axes],
                           xytext=(5, 5), textcoords="offset points", fontsize=8)
        panel.legend(fontsize=8, loc="upper left")
    locations.set_title("一、采样位置（数字为实验编号）")
    plans.set_title("二、相对无注入基线的实测计划变化")
    for axis, label in zip(axes, labels):
        subset = sorted((p for p in points if p["design"] in ("control", "one_at_a_time")
                         and all(f == 1 for j, f in enumerate(p["factors"]) if j != axis)
                         and p["optimizer_cost"] is not None), key=lambda p: p["factors"][axis])
        costs.plot([p["factors"][axis] for p in subset], [p["optimizer_cost"] for p in subset],
                   "o--", label="仅扰动" + label)
    costs.set_xscale("log", base=2)
    costs.set_xlabel("基数倍率（其他选中目标固定为一倍）")
    costs.set_ylabel("优化器估算成本，不是执行耗时")
    costs.set_title("三、条件响应切面（虚线仅连接实测点，不是拟合）")
    costs.legend(fontsize=8)
    for field, label in (("memo_expressions", "搜索空间表达式数量"), ("binding_attempts", "规则绑定尝试次数"),
                         ("rule_applications", "规则应用次数")):
        subset = [p for p in points if p[field] is not None]
        search.scatter([p["index"] for p in subset], [p[field] for p in subset], label=label, s=30)
    search.set_xlabel("实验编号（分类顺序，不是连续变量）")
    search.set_ylabel("数量／次数")
    search.set_title("四、实测搜索工作量（无效运行不填零）")
    search.legend(fontsize=8)
    for panel in panels.flat:
        panel.grid(alpha=0.2)
    fig.suptitle(title + "\n仅展示实测点；优化器成本不代表实际执行耗时", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "replacement"), default="replacement")
    parser.add_argument("--profile", action="store_true", help="绘制单目标基数扰动的已配置规则实验臂")
    parser.add_argument("--timing", action="store_true", help="绘制单目标已配置实验臂的无跟踪计时分布")
    parser.add_argument("--axes", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--labels", nargs=2, help="display labels for selected targets")
    parser.add_argument("--output", type=Path, required=True, help="output file stem, normally under output/")
    parser.add_argument("--font", type=Path,
                        default=PGORCA_ROOT / "output/fonts/NotoSansCJKsc-Regular.otf",
                        help="中文字体路径；字体仅用于离线绘图")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    comparison = json.loads(args.comparison.read_text())
    if args.profile or args.timing:
        if len(manifest["targets"]) != 1 or not comparison.get("rule_profile"):
            parser.error("规则对比图需要单个基数目标和规则实验结果")
        if args.timing and not comparison["rule_profile"].get("timing"):
            parser.error("没有无跟踪计时样本")
        renderer = render_timing if args.timing else render_profile
        renderer(manifest, comparison, args.manifest.parent, args.output, args.font)
        print(args.output.with_suffix(".png"))
        return
    if len(set(args.axes)) != 2 or any(a < 0 or a >= len(manifest["targets"]) for a in args.axes):
        parser.error("choose two distinct target axes present in the manifest")
    points = measured_points(manifest, comparison, args.manifest.parent, args.mode)
    labels = args.labels or [f"目标{a + 1}（{manifest['targets'][a]['fingerprint'][:8]}）" for a in args.axes]
    mode_label = "原生规则模式" if args.mode == "native" else "规则替代模式"
    render(points, tuple(args.axes), labels,
           f"{comparison['workload']}/{comparison['query']}｜{mode_label}｜随机种子 {manifest['seed']}",
           args.output, args.font)
    args.output.with_suffix(".json").write_text(json.dumps(points, indent=2) + "\n")
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
