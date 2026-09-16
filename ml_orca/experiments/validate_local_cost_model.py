#!/usr/bin/env python3
"""Validate one frozen local cost formula against application observations."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

from ml_orca.experiments.plot_stats_sweep import chinese_plotting


def extract_row(row, input_operator, generated_operator, generated_path, input_coefficient,
                output_coefficient, intercept, clip_max):
    contribution = row.get("contribution") or {}
    generated = contribution.get("target_generated_costs") or {}
    inputs = {x.get("rows") for x in generated.get("source_children", [])
              if x.get("operator") == input_operator and x.get("rows") is not None}
    outputs = {x.get("rows") for x in generated.get("physical_observations", [])
               if x.get("operator") == generated_operator
               and x.get("target_paths") == [generated_path] and x.get("rows") is not None}
    input_rows = next(iter(inputs)) if len(inputs) == 1 else None
    output_rows = next(iter(outputs)) if len(outputs) == 1 else None
    predicted = None if input_rows is None or output_rows is None else min(
        clip_max, intercept + input_coefficient * input_rows + output_coefficient * output_rows)
    observed = (contribution.get("delta") or {}).get("optimizer_cost")
    planning = [x["delta_planning_ms"] for x in contribution.get("pairs", [])
                if not x.get("exclusions") and x.get("delta_planning_ms") is not None]
    return {"case": row["case"], "parameters": row["parameters"], "status": row["status"],
            "postgres_result_equal": row.get("native_rows_equal"), "input_rows": input_rows,
            "generated_rows": output_rows, "predicted_cost_delta": predicted,
            "observed_cost_delta": observed,
            "absolute_error": abs(predicted - observed) if predicted is not None and observed is not None else None,
            "planning_delta_median_ms": median(planning) if planning else None,
            "memo_expressions_delta": (contribution.get("delta") or {}).get("memo_expressions"),
            "prediction_exclusion": None if predicted is not None else "generated_path_not_costed"}


def validate(reports, input_operator, generated_operator, generated_path, input_coefficient,
             output_coefficient, intercept=0.0, clip_max=0.0):
    rows = [extract_row(row, input_operator, generated_operator, generated_path, input_coefficient,
                        output_coefficient, intercept, clip_max)
            for report in reports for row in report["runs"]]
    errors = [row["absolute_error"] for row in rows if row["absolute_error"] is not None]
    return {"formula": {"input_operator": input_operator, "generated_operator": generated_operator,
                        "generated_path": generated_path, "input_coefficient": input_coefficient,
                        "output_coefficient": output_coefficient, "intercept": intercept,
                        "clip_max": clip_max},
            "rows": rows, "exact_predictions": len(errors),
            "mean_absolute_error": sum(errors) / len(errors) if errors else None,
            "all_postgres_results_equal": all(row["postgres_result_equal"] is True for row in rows),
            "scope": "frozen_formula_application_transfer_not_refit_or_population_guarantee"}


def render(report, output, font):
    plt = chinese_plotting(font)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    exact = [row for row in report["rows"] if row["absolute_error"] is not None]
    values = [value for row in exact for value in (row["predicted_cost_delta"], row["observed_cost_delta"])]
    if values:
        low, high = min(values), max(values)
        axes[0].plot([low, high], [low, high], "--", color="gray", label="理想一致线")
    labels = {row["case"]: f"样本{i + 1}" for i, row in enumerate(report["rows"])}
    for i, row in enumerate(exact):
        axes[0].scatter(row["predicted_cost_delta"], row["observed_cost_delta"], s=55)
        axes[0].annotate(labels[row["case"]], (row["predicted_cost_delta"], row["observed_cost_delta"]),
                         xytext=(4, 4 + 10 * (i % 2)), textcoords="offset points", fontsize=8)
    axes[0].set(title=f"冻结公式迁移：{len(exact)} 个可完整计价样本",
                xlabel="公式预测的估算成本差", ylabel="实际观测的估算成本差")
    axes[0].legend()
    positions = range(len(report["rows"]))
    benefit = [-row["observed_cost_delta"] if row["observed_cost_delta"] is not None else math.nan
               for row in report["rows"]]
    planning = [row["planning_delta_median_ms"] if row["planning_delta_median_ms"] is not None else math.nan
                for row in report["rows"]]
    axes[1].bar(positions, benefit, color="#228833", label="估算成本改善")
    timing = axes[1].twinx()
    timing.plot(positions, planning, "o-", color="#cc3311", label="规划耗时净增")
    axes[1].set_xticks(list(positions), [labels[row["case"]] for row in report["rows"]])
    axes[1].set(title="收益与搜索开销是两个独立响应", ylabel="关闭 − 开启的估算成本")
    timing.set_ylabel("开启 − 关闭的规划时间中位数（毫秒）", color="#cc3311")
    axes[1].legend(loc="upper left")
    timing.legend(loc="upper right")
    for axis in axes:
        axis.axhline(0, color="gray", linewidth=.7)
        axis.grid(alpha=.2)
    fig.suptitle("真实应用输入上的局部代价机制复核\n公式冻结后未重拟合；缺少完整物理计价的样本保留在右图，不伪造预测")
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--input-operator", required=True)
    parser.add_argument("--generated-operator", required=True)
    parser.add_argument("--generated-path", required=True)
    parser.add_argument("--input-coefficient", type=float, required=True)
    parser.add_argument("--output-coefficient", type=float, required=True)
    parser.add_argument("--intercept", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("output/fonts/NotoSansCJKsc-Regular.otf"))
    args = parser.parse_args()
    report = validate([json.loads(path.read_text()) for path in args.report], args.input_operator,
                      args.generated_operator, args.generated_path, args.input_coefficient,
                      args.output_coefficient, args.intercept, args.clip_max)
    report["sources"] = [str(path.resolve()) for path in args.report]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    render(report, args.output, args.font)
    print(json.dumps({key: report[key] for key in
                      ("exact_predictions", "mean_absolute_error", "all_postgres_results_equal")}))


if __name__ == "__main__":
    main()
