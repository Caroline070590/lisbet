#!/usr/bin/env python3
"""Plot training dynamics for the full and leave-one-task-out LISBET encoders.

This release script preserves the calculations and visual settings of the
manuscript plotting script while replacing machine-specific paths with command-
line arguments.
"""

from argparse import ArgumentParser
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASKS = ["cons", "order", "shift", "warp"]

RUNS = {
    "all_tasks": "All tasks",
    "without_cons": "Without cons",
    "without_order": "Without order",
    "without_shift": "Without shift",
    "without_warp": "Without warp",
}

COLORS = {
    "all_tasks": "#222222",
    "without_cons": "#4c78a8",
    "without_order": "#f45b6c",
    "without_shift": "#1b8a3a",
    "without_warp": "#c9b934",
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Directory containing the 75 trained model run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which PDF, PNG and SVG files will be written.",
    )
    return parser.parse_args()


def infer_run_key(run_dir):
    name = run_dir.name
    if name.startswith("all_tasks"):
        return "all_tasks"
    if "triple_order_shift_warp" in name:
        return "without_cons"
    if "triple_cons_shift_warp" in name:
        return "without_order"
    if "triple_cons_order_warp" in name:
        return "without_shift"
    if "triple_cons_order_shift" in name:
        return "without_warp"
    return None


def find_metrics_file(run_dir):
    files = list(run_dir.rglob("metrics.csv"))
    return files[0] if files else None


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="normal",
        ha="left",
        va="top",
    )


def main():
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
    })

    rows = []
    for run_dir in sorted(run_root.glob("*_75ep")):
        run_key = infer_run_key(run_dir)
        if run_key is None:
            continue

        metrics_file = find_metrics_file(run_dir)
        if metrics_file is None:
            continue

        seed_match = re.search(r"_seed(\d+)_", run_dir.name)
        seed = int(seed_match.group(1)) if seed_match else -1

        data = pd.read_csv(metrics_file)
        if "epoch" not in data.columns:
            continue

        score_cols = [
            f"{task}_train_score"
            for task in TASKS
            if f"{task}_train_score" in data.columns
        ]
        loss_cols = [
            f"{task}_train_loss"
            for task in TASKS
            if f"{task}_train_loss" in data.columns
        ]

        if score_cols:
            rows.append(pd.DataFrame({
                "epoch": data["epoch"],
                "value": data[score_cols].mean(axis=1),
                "metric": "score",
                "run_key": run_key,
                "seed": seed,
            }))

        if loss_cols:
            rows.append(pd.DataFrame({
                "epoch": data["epoch"],
                "value": data[loss_cols].mean(axis=1),
                "metric": "loss",
                "run_key": run_key,
                "seed": seed,
            }))

    if not rows:
        raise SystemExit(f"No compatible metrics.csv files found under {run_root}")

    curves = pd.concat(rows, ignore_index=True)
    curves["value"] = pd.to_numeric(curves["value"], errors="coerce")
    curves = curves[np.isfinite(curves["value"])]

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharex=True)
    for axis, metric, ylabel, letter in [
        (axes[0], "score", "Mean training score", "a"),
        (axes[1], "loss", "Mean training loss", "b"),
    ]:
        for run_key, label in RUNS.items():
            subset = curves[
                (curves["metric"] == metric)
                & (curves["run_key"] == run_key)
            ]
            if subset.empty:
                continue

            grouped = (
                subset.groupby("epoch")["value"]
                .agg(["mean", "sem"])
                .reset_index()
                .sort_values("epoch")
            )

            x = grouped["epoch"].to_numpy(float)
            y = grouped["mean"].to_numpy(float)
            sem = (
                grouped["sem"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .to_numpy(float)
            )

            axis.plot(x, y, color=COLORS[run_key], lw=1.7, label=label)
            axis.fill_between(
                x,
                y - sem,
                y + sem,
                color=COLORS[run_key],
                alpha=0.16,
                linewidth=0,
            )

        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        panel_label(axis, letter)
        clean_axis(axis)

    axes[0].set_ylim(0.45, 1.02)
    axes[1].set_ylim(0.0, 0.75)
    axes[0].set_title("Training score")
    axes[1].set_title("Training loss")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    figure.tight_layout(rect=[0, 0.10, 1, 1])

    for extension in ("pdf", "png", "svg"):
        figure.savefig(
            output_dir / f"ab_training_summary_mean_sem.{extension}",
            dpi=600,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
