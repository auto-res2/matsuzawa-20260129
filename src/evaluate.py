import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

PRIMARY_METRIC = "worst_group_unsafe_selection_rate"
METRIC_DIRECTIONS = {
    "worst_group_unsafe_selection_rate": "min",
    "feasible_top1_accuracy": "max",
    "abstention_rate": "min",
    "audit_allocation_disparity": "min",
    "utility_regret_under_gate": "min",
}


def parse_kv_args(argv: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed[key] = value
    return parsed


def save_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_learning_curve(
    run_id: str, df: pd.DataFrame, metric: str, out_dir: str
) -> str:
    if metric not in df.columns or "budget" not in df.columns:
        return ""
    grouped = df.groupby("budget")[metric].mean().reset_index()
    if grouped.empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        grouped["budget"],
        grouped[metric],
        marker="o",
        linewidth=2,
        markersize=6,
        label=metric,
    )

    # Shorten run_id for display
    display_name = (
        run_id.replace("comparative-1-", "comp-")
        .replace("proposed-", "prop-")
        .replace("-realtoxicityprompts", "")
    )
    ax.set_title(
        f"{display_name}\n{metric.replace('_', ' ')}", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Budget", fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.annotate(
        f"{grouped[metric].iloc[-1]:.3f}",
        xy=(grouped["budget"].iloc[-1], grouped[metric].iloc[-1]),
        fontsize=10,
        ha="right",
        va="bottom",
    )
    ax.legend(fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    filename = os.path.join(out_dir, f"{run_id}_{metric}_learning_curve.pdf")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_confusion_matrix(run_id: str, df: pd.DataFrame, out_dir: str) -> str:
    if "selected_strategy" not in df.columns or "true_best_strategy" not in df.columns:
        return ""
    sub = df[["selected_strategy", "true_best_strategy", "abstention_rate"]].dropna()
    if sub.empty:
        return ""
    selected = sub["selected_strategy"].astype(int).astype(str)
    true_best = sub["true_best_strategy"].astype(int).astype(str)
    selected = selected.where(sub["abstention_rate"] < 0.5, other="abstain")
    table = pd.crosstab(true_best, selected)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        table,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Count"},
        annot_kws={"fontsize": 11},
    )

    # Shorten run_id for display
    display_name = (
        run_id.replace("comparative-1-", "comp-")
        .replace("proposed-", "prop-")
        .replace("-realtoxicityprompts", "")
    )
    ax.set_title(
        f"{display_name}\nSelection vs True Best", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Selected Strategy", fontsize=11)
    ax.set_ylabel("True Best Strategy", fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=10)
    fig.tight_layout()
    filename = os.path.join(out_dir, f"{run_id}_confusion_matrix.pdf")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_bar_chart(
    metric: str, values: Dict[str, float], out_dir: str, prefix: str
) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    run_ids = list(values.keys())
    # Shorten run_ids for display
    display_names = [
        r.replace("comparative-1-", "comp-")
        .replace("proposed-", "prop-")
        .replace("-realtoxicityprompts", "")
        for r in run_ids
    ]
    vals = [values[r] for r in run_ids]

    # Use a professional color palette
    colors = sns.color_palette("Set2", n_colors=len(run_ids))
    bars = ax.bar(
        range(len(display_names)), vals, color=colors, edgecolor="black", linewidth=0.5
    )

    ax.set_title(
        metric.replace("_", " ").title(), fontsize=14, fontweight="bold", pad=20
    )
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_xlabel("Run ID", fontsize=12)
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis="y", which="major", labelsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on top of bars
    for idx, (bar, val) in enumerate(zip(bars, vals)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    filename = os.path.join(out_dir, f"{prefix}_{metric}_bar_chart.pdf")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_box_plot(
    metric: str, values: Dict[str, List[float]], out_dir: str, prefix: str
) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    for run_id, vals in values.items():
        # Shorten run_id for display
        display_name = (
            run_id.replace("comparative-1-", "comp-")
            .replace("proposed-", "prop-")
            .replace("-realtoxicityprompts", "")
        )
        for v in vals:
            data.append({"run_id": display_name, metric: v})
    df = pd.DataFrame(data)
    if df.empty:
        return ""

    # Use a professional color palette
    colors = sns.color_palette("Set2", n_colors=len(values))
    sns.boxplot(
        data=df, x="run_id", y=metric, ax=ax, hue="run_id", palette=colors, legend=False
    )

    ax.set_title(
        f"{metric.replace('_', ' ').title()} Distribution",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_xlabel("Run ID", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", which="major", labelsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    filename = os.path.join(out_dir, f"{prefix}_{metric}_box_plot.pdf")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_metric_table(
    metrics: Dict[str, Dict[str, float]], out_dir: str, prefix: str
) -> str:
    # Create DataFrame from metrics dict
    # Structure: metrics = {metric_name: {run_id: value}}
    # DataFrame will have run_ids as rows and metric_names as columns
    df = pd.DataFrame(metrics)

    # Shorten row names (run_ids)
    df.index = [
        idx.replace("comparative-1-", "comp-")
        .replace("proposed-", "prop-")
        .replace("-realtoxicityprompts", "")
        for idx in df.index
    ]

    # Format column names (metrics) for better readability
    df.columns = [col.replace("_", " ").title() for col in df.columns]

    # Calculate appropriate figure size based on content
    num_rows = len(df)
    num_cols = len(df.columns)
    fig_width = max(12, num_cols * 3.5)
    fig_height = max(4, num_rows * 0.9 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=np.round(df.values, 4),
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row (column labels)
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(weight="bold", color="white", fontsize=11)

    # Style row labels
    for i in range(1, len(df) + 1):
        cell = table[(i, -1)]
        cell.set_facecolor("#D9E2F3")
        cell.set_text_props(weight="bold", fontsize=10)

    fig.tight_layout()
    filename = os.path.join(out_dir, f"{prefix}_metrics_table.pdf")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return filename


def extract_summary_metric(summary: Dict[str, Any], metric: str) -> float:
    if metric in summary and isinstance(summary[metric], (int, float)):
        return float(summary[metric])
    if f"best_{metric}" in summary and isinstance(
        summary[f"best_{metric}"], (int, float)
    ):
        return float(summary[f"best_{metric}"])
    if f"final_{metric}" in summary and isinstance(
        summary[f"final_{metric}"], (int, float)
    ):
        return float(summary[f"final_{metric}"])
    return float("nan")


def metric_direction(metric: str) -> str:
    return METRIC_DIRECTIONS.get(metric, "min")


def main() -> None:
    args = parse_kv_args(sys.argv[1:])
    if "results_dir" not in args or "run_ids" not in args:
        raise ValueError(
            "Usage: python -m src.evaluate results_dir=... run_ids='[...]'"
        )
    results_dir = args["results_dir"]
    run_ids = json.loads(args["run_ids"])
    if not isinstance(run_ids, list) or not run_ids:
        raise ValueError("run_ids must be a non-empty JSON list string.")

    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    if cfg.wandb.mode == "disabled":
        raise RuntimeError(
            "WandB must be enabled for evaluation. Set wandb.mode=online in config/config.yaml."
        )

    entity = cfg.wandb.entity
    project = cfg.wandb.project

    api = wandb.Api()
    all_metrics: Dict[str, Dict[str, float]] = {}
    all_distributions: Dict[str, Dict[str, List[float]]] = {}
    generated_files: List[str] = []

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()
        summary = run.summary._json_dict
        config = dict(run.config)

        run_mode = str(config.get("mode", ""))
        wandb_mode = str(config.get("wandb", {}).get("mode", ""))
        if run_mode == "trial" or wandb_mode == "disabled":
            raise RuntimeError(
                f"Run {run_id} was executed in trial mode or with WandB disabled; evaluation requires full runs."
            )

        run_dir = os.path.join(results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        metrics_path = os.path.join(run_dir, "metrics.json")
        save_json(
            metrics_path,
            {
                "history": history.to_dict(orient="list"),
                "summary": summary,
                "config": config,
            },
        )
        generated_files.append(metrics_path)

        for metric in [
            PRIMARY_METRIC,
            "feasible_top1_accuracy",
            "abstention_rate",
            "audit_allocation_disparity",
            "utility_regret_under_gate",
        ]:
            path = plot_learning_curve(run_id, history, metric, run_dir)
            if path:
                generated_files.append(path)

        confusion_path = plot_confusion_matrix(run_id, history, run_dir)
        if confusion_path:
            generated_files.append(confusion_path)

        for metric in [
            PRIMARY_METRIC,
            "feasible_top1_accuracy",
            "abstention_rate",
            "audit_allocation_disparity",
            "utility_regret_under_gate",
        ]:
            value = extract_summary_metric(summary, metric)
            if not np.isnan(value):
                all_metrics.setdefault(metric, {})[run_id] = value

        for metric in [PRIMARY_METRIC, "feasible_top1_accuracy", "abstention_rate"]:
            if metric in history.columns:
                all_distributions.setdefault(metric, {})[run_id] = (
                    history[metric].dropna().tolist()
                )

    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    aggregated = {
        "primary_metric": PRIMARY_METRIC,
        "metrics": all_metrics,
        "best_proposed": {},
        "best_baseline": {},
        "gap": None,
    }

    if PRIMARY_METRIC in all_metrics:
        metric_values = all_metrics[PRIMARY_METRIC]
        proposed = {k: v for k, v in metric_values.items() if "proposed" in k}
        baseline = {
            k: v
            for k, v in metric_values.items()
            if "comparative" in k or "baseline" in k
        }
        if proposed:
            best_prop = (
                min(proposed.items(), key=lambda x: x[1])
                if metric_direction(PRIMARY_METRIC) == "min"
                else max(proposed.items(), key=lambda x: x[1])
            )
            aggregated["best_proposed"] = {
                "run_id": best_prop[0],
                "value": best_prop[1],
            }
        if baseline:
            best_base = (
                min(baseline.items(), key=lambda x: x[1])
                if metric_direction(PRIMARY_METRIC) == "min"
                else max(baseline.items(), key=lambda x: x[1])
            )
            aggregated["best_baseline"] = {
                "run_id": best_base[0],
                "value": best_base[1],
            }
        if aggregated["best_proposed"] and aggregated["best_baseline"]:
            prop_val = aggregated["best_proposed"]["value"]
            base_val = aggregated["best_baseline"]["value"]
            gap = (prop_val - base_val) / max(1e-12, base_val) * 100.0
            if metric_direction(PRIMARY_METRIC) == "min":
                gap = -gap
            aggregated["gap"] = gap

    if PRIMARY_METRIC in all_distributions and len(run_ids) >= 2:
        pairwise = {}
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                r1, r2 = run_ids[i], run_ids[j]
                vals1 = all_distributions.get(PRIMARY_METRIC, {}).get(r1, [])
                vals2 = all_distributions.get(PRIMARY_METRIC, {}).get(r2, [])
                if len(vals1) > 1 and len(vals2) > 1:
                    tstat, pval = stats.ttest_ind(vals1, vals2, equal_var=False)
                    pairwise[f"{r1}_vs_{r2}"] = {
                        "tstat": float(tstat),
                        "pval": float(pval),
                    }
        aggregated["significance_tests"] = pairwise

    aggregated_path = os.path.join(comparison_dir, "aggregated_metrics.json")
    save_json(aggregated_path, aggregated)
    generated_files.append(aggregated_path)

    if PRIMARY_METRIC in all_metrics:
        path = plot_bar_chart(
            PRIMARY_METRIC, all_metrics[PRIMARY_METRIC], comparison_dir, "comparison"
        )
        generated_files.append(path)
    if "feasible_top1_accuracy" in all_metrics:
        path = plot_bar_chart(
            "feasible_top1_accuracy",
            all_metrics["feasible_top1_accuracy"],
            comparison_dir,
            "comparison",
        )
        generated_files.append(path)
    if PRIMARY_METRIC in all_distributions:
        path = plot_box_plot(
            PRIMARY_METRIC,
            all_distributions[PRIMARY_METRIC],
            comparison_dir,
            "comparison",
        )
        if path:
            generated_files.append(path)

    if all_metrics:
        table_path = plot_metric_table(all_metrics, comparison_dir, "comparison")
        generated_files.append(table_path)

    for path in generated_files:
        print(path)


if __name__ == "__main__":
    main()
