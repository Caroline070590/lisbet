#!/usr/bin/env python3
"""Reproduce the final six-panel CalMS21 Task 2 and MARS1 annotator/bout figure.

This is a portable extraction of the final executed code cell from
``annotator-bias.ipynb``. Only file-location handling and notebook display
calls were changed; calculations, panel definitions, and plotting settings
were preserved.
"""

# ============================================================
# FIXED combined annotator-bias plot
# CalMS21 Task 2 + MARS1
#
# Panels:
# a: CalMS21 pairwise agreement
# b: MARS1 pairwise agreement, same dotplot style as a
# c: CalMS21 label distribution
# d: MARS1 positive-frame fraction
# e: CalMS21 bout duration
# f: MARS1 bout duration
#
# Fixes:
# - remove global title
# - use one shared centered legend for panels a and b
# - move MARS coverage inset to x ~ 0.26 in main data coordinates
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CALMS_DIR = ROOT / "results/calms21_task2_annotator_bias"
MARS_DIR = ROOT / "results/mars1_annotator_bias"

FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

STYLE = ROOT / "lisbet.mplstyle"

required = [
    CALMS_DIR / "calms21_task2_pairwise_interannotator_summary.csv",
    CALMS_DIR / "calms21_task2_annotator_file_summary.csv",
    CALMS_DIR / "calms21_task2_label_distribution_by_annotator.csv",
    CALMS_DIR / "calms21_task2_bout_stats_by_annotator.csv",
    MARS_DIR / "mars1_pairwise_interannotator_summary.csv",
    MARS_DIR / "mars1_annotator_file_summary.csv",
    MARS_DIR / "mars1_behavior_positive_fraction_by_annotator.csv",
    MARS_DIR / "mars1_bout_stats_by_annotator.csv",
]

missing = [p for p in required if not p.exists()]
if missing:
    print("Missing required files:")
    for p in missing:
        print(" -", p)
    raise FileNotFoundError(
        "Please copy/extract annotator_bias_results_csvs_FIXED_20260701.tar.gz first."
    )

# ============================================================
# Colors
# ============================================================

COL_BLUE = "#68a8e0"
COL_GREEN = "#9ad175"
COL_ORANGE = "#f9a24b"
COL_GRAY = "#8a8a8a"
COL_LINE = "#d6d6d6"
COL_BOX = "#d0d0d0"

label_cmap = LinearSegmentedColormap.from_list(
    "lisbet_label",
    ["#f4f4f4", COL_BLUE, COL_GREEN],
)

bout_cmap = LinearSegmentedColormap.from_list(
    "lisbet_bout",
    ["#f4f4f4", COL_BLUE, COL_ORANGE, COL_GREEN],
)

# ============================================================
# Helper functions
# ============================================================

def short_annotator(x):
    return str(x).replace("annotator", "A")


def add_panel_letter(ax, letter, x=-0.16, y=1.05):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="normal",
        va="top",
        ha="left",
        color="black",
    )


def prepare_calms_pairwise(df):
    df = df.copy()
    df["pair"] = (
        df["annotator_a"].map(short_annotator)
        + "–"
        + df["annotator_b"].map(short_annotator)
    )
    return df.sort_values("macro_f1_mean", ascending=True).reset_index(drop=True)


def prepare_mars_pairwise(df):
    df = df.copy()
    df["pair"] = (
        df["annotator_a"].map(short_annotator)
        + "–"
        + df["annotator_b"].map(short_annotator)
    )
    return df.sort_values("macro_f1_mean", ascending=True).reset_index(drop=True)


def prepare_annot(df):
    df = df.copy()
    df["annotator_short"] = df["annotator"].map(short_annotator)
    return df


def prepare_label_pivot(df):
    piv = df.pivot_table(
        index="behavior",
        columns="annotator",
        values="fraction_within_annotator",
        fill_value=0,
    )
    piv.columns = [short_annotator(c) for c in piv.columns]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    return piv


def prepare_mars_positive_pivot(df):
    piv = df.pivot_table(
        index="behavior",
        columns="annotator",
        values="fraction_within_annotator",
        fill_value=0,
    )
    piv.columns = [short_annotator(c) for c in piv.columns]

    preferred = ["sniff", "mount", "aggressivemount", "attack"]
    order = [b for b in preferred if b in piv.index]
    rest = [b for b in piv.index if b not in order]

    return piv.loc[order + rest]


def prepare_bout_pivot(df, dataset):
    piv = df.pivot_table(
        index="behavior",
        columns="annotator",
        values="mean_bout_duration_s",
        fill_value=np.nan,
    )
    piv.columns = [short_annotator(c) for c in piv.columns]

    if dataset == "calms":
        preferred = ["other", "mount", "attack", "investigation"]
    else:
        preferred = ["sniff", "mount", "aggressivemount", "attack"]

    order = [b for b in preferred if b in piv.index]
    rest = [b for b in piv.index if b not in order]

    if order:
        piv = piv.loc[order + rest]
    else:
        piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

    return piv


# ============================================================
# Plot functions
# ============================================================

def plot_calms_pairwise(ax, df, annot_df, title):
    y = np.arange(len(df))

    for i, row in df.iterrows():
        ax.plot(
            [row["cohen_kappa_mean"], row["macro_f1_mean"]],
            [i, i],
            color=COL_LINE,
            linewidth=0.55,
            alpha=0.95,
            zorder=1,
            solid_capstyle="round",
        )

    ax.scatter(
        df["cohen_kappa_mean"],
        y,
        s=30,
        color=COL_BLUE,
        edgecolor="white",
        linewidth=0.4,
        label="Cohen's κ",
        zorder=3,
    )

    ax.scatter(
        df["macro_f1_mean"],
        y,
        s=30,
        color=COL_GREEN,
        edgecolor="white",
        linewidth=0.4,
        label="Macro-F1",
        zorder=4,
    )

    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["pair"], fontsize=6)
    ax.set_xlabel("Agreement score")
    ax.set_ylabel("Annotator pair")
    ax.set_xlim(-0.12, 1.0)
    ax.set_title(title, loc="left", pad=4)
    ax.grid(axis="x", color="#ededed", linewidth=0.4)
    ax.set_axisbelow(True)

    # No local legend. Shared legend is added later.

    ax_inset = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.56, 0.34, 0.36, 0.24),
        bbox_transform=ax.transAxes,
        loc="lower left",
        borderpad=0,
    )

    ax_inset.bar(
        annot_df["annotator_short"],
        annot_df["total_frames"] / 1000,
        color=COL_GRAY,
        width=0.72,
        zorder=3,
    )

    ax_inset.set_facecolor("white")
    for spine in ax_inset.spines.values():
        spine.set_visible(True)
        spine.set_color(COL_BOX)
        spine.set_linewidth(0.8)

    ax_inset.set_title("coverage", fontsize=6, pad=2)
    ax_inset.set_ylabel("frames\n×10³", fontsize=5)
    ax_inset.tick_params(axis="both", labelsize=5, length=2)
    ax_inset.grid(axis="y", color="#ececec", linewidth=0.35)
    ax_inset.set_axisbelow(True)


def plot_mars_pairwise(ax, df, annot_df, title):
    y = np.arange(len(df))

    for i, row in df.iterrows():
        ax.plot(
            [row["cohen_kappa_mean"], row["macro_f1_mean"]],
            [i, i],
            color=COL_LINE,
            linewidth=0.45,
            alpha=0.95,
            zorder=1,
            solid_capstyle="round",
        )

    ax.scatter(
        df["cohen_kappa_mean"],
        y,
        s=20,
        color=COL_BLUE,
        edgecolor="white",
        linewidth=0.35,
        label="Cohen's κ",
        zorder=3,
    )

    ax.scatter(
        df["macro_f1_mean"],
        y,
        s=20,
        color=COL_GREEN,
        edgecolor="white",
        linewidth=0.35,
        label="Macro-F1",
        zorder=4,
    )

    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["pair"], fontsize=4.3)
    ax.set_xlabel("Agreement score")
    ax.set_ylabel("Annotator pair")

    # Keep the better MARS panel range.
    xlim = (-0.03, 0.92)
    ax.set_xlim(*xlim)

    ax.set_title(title, loc="left", pad=4)
    ax.grid(axis="x", color="#ededed", linewidth=0.4)
    ax.set_axisbelow(True)

    # No local legend. Shared legend is added later.

    # Coverage inset positioned around x ≈ 0.26 in main data coordinates.
    # This avoids overlap with y-axis labels but keeps it in the empty zone.
    inset_w = 0.32
    inset_h = 0.22
    inset_y0 = 0.36

    desired_x_center_data = 0.26
    x_center_frac = (desired_x_center_data - xlim[0]) / (xlim[1] - xlim[0])
    inset_x0 = x_center_frac - inset_w / 2
    inset_x0 = max(0.02, min(1 - inset_w - 0.02, inset_x0))

    ax_inset = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(inset_x0, inset_y0, inset_w, inset_h),
        bbox_transform=ax.transAxes,
        loc="lower left",
        borderpad=0,
    )

    ax_inset.bar(
        annot_df["annotator_short"],
        annot_df["total_frames"] / 1000,
        color=COL_GRAY,
        width=0.72,
        zorder=3,
    )

    ax_inset.set_facecolor("white")
    for spine in ax_inset.spines.values():
        spine.set_visible(True)
        spine.set_color(COL_BOX)
        spine.set_linewidth(0.8)

    ax_inset.set_title("coverage", fontsize=6, pad=2)
    ax_inset.set_ylabel("frames\n×10³", fontsize=5)
    ax_inset.tick_params(axis="both", labelsize=5, length=2)
    ax_inset.grid(axis="y", color="#ececec", linewidth=0.35)
    ax_inset.set_axisbelow(True)


def plot_heatmap(ax, pivot, title, cmap, cbar_label, vmax=None):
    vals = pivot.values.astype(float)

    if vmax is None:
        if np.isfinite(vals).any():
            vmax = np.nanpercentile(vals, 95)
            vmax = max(0.01, vmax)
        else:
            vmax = 1.0

    im = ax.imshow(
        vals,
        aspect="auto",
        cmap=cmap,
        norm=Normalize(vmin=0, vmax=vmax),
    )

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=6)

    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=6)

    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("Annotator")
    ax.set_ylabel("Behavior")

    ax.set_xticks(np.arange(-0.5, pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=5, length=2)


# ============================================================
# Load fixed data
# ============================================================

calms_pair_raw = pd.read_csv(CALMS_DIR / "calms21_task2_pairwise_interannotator_summary.csv")
calms_annot_raw = pd.read_csv(CALMS_DIR / "calms21_task2_annotator_file_summary.csv")
calms_label_raw = pd.read_csv(CALMS_DIR / "calms21_task2_label_distribution_by_annotator.csv")
calms_bout_raw = pd.read_csv(CALMS_DIR / "calms21_task2_bout_stats_by_annotator.csv")

mars_pair_raw = pd.read_csv(MARS_DIR / "mars1_pairwise_interannotator_summary.csv")
mars_annot_raw = pd.read_csv(MARS_DIR / "mars1_annotator_file_summary.csv")
mars_positive_raw = pd.read_csv(MARS_DIR / "mars1_behavior_positive_fraction_by_annotator.csv")
mars_bout_raw = pd.read_csv(MARS_DIR / "mars1_bout_stats_by_annotator.csv")

calms_pair = prepare_calms_pairwise(calms_pair_raw)
calms_annot = prepare_annot(calms_annot_raw)
calms_label = prepare_label_pivot(calms_label_raw)
calms_bout = prepare_bout_pivot(calms_bout_raw, dataset="calms")

mars_pair = prepare_mars_pairwise(mars_pair_raw)
mars_annot = prepare_annot(mars_annot_raw)
mars_positive = prepare_mars_positive_pivot(mars_positive_raw)
mars_bout = prepare_bout_pivot(mars_bout_raw, dataset="mars")

# ============================================================
# Save numerical summary
# ============================================================

summary = pd.DataFrame([
    {
        "dataset": "CalMS21 Task 2",
        "n_annotators": calms_annot["annotator"].nunique(),
        "n_pairwise_comparisons": len(calms_pair_raw),
        "total_frames_annotated": int(calms_annot["total_frames"].sum()),
        "accuracy_mean": calms_pair_raw["accuracy_mean"].mean(),
        "balanced_accuracy_mean": calms_pair_raw["balanced_accuracy_mean"].mean(),
        "macro_f1_mean": calms_pair_raw["macro_f1_mean"].mean(),
        "cohen_kappa_mean": calms_pair_raw["cohen_kappa_mean"].mean(),
        "mcc_mean": calms_pair_raw["mcc_mean"].mean(),
    },
    {
        "dataset": "MARS1",
        "n_annotators": mars_annot["annotator"].nunique(),
        "n_pairwise_comparisons": len(mars_pair_raw),
        "total_frames_annotated": int(mars_annot["total_frames"].sum()),
        "accuracy_mean": mars_pair_raw["accuracy_mean"].mean(),
        "balanced_accuracy_mean": mars_pair_raw["balanced_accuracy_mean"].mean(),
        "macro_f1_mean": mars_pair_raw["macro_f1_mean"].mean(),
        "cohen_kappa_mean": mars_pair_raw["cohen_kappa_mean"].mean(),
        "mcc_mean": mars_pair_raw["mcc_mean"].mean(),
    },
])

summary.to_csv(FIGDIR / "combined_annotator_bias_summary_shared_legend.csv", index=False)

print("Combined numerical summary:")
print(summary.to_string(index=False))

print("MARS positive-frame fraction:")
print(mars_positive.to_string(index=False))

print("MARS bout duration:")
print(mars_bout.to_string(index=False))

# ============================================================
# Plot
# ============================================================

plt.close("all")

context = plt.style.context(STYLE) if STYLE.exists() else plt.rc_context()

with context:
    mpl.rcParams["figure.constrained_layout.use"] = False
    mpl.rcParams["figure.autolayout"] = False
    mpl.rcParams["savefig.bbox"] = "standard"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    fig = plt.figure(figsize=(8.2, 9.0), constrained_layout=False)

    try:
        fig.set_layout_engine(None)
    except Exception:
        pass

    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[2.75, 1.05, 1.05],
        width_ratios=[1.0, 1.0],
        wspace=0.45,
        hspace=0.72,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    plot_calms_pairwise(
        ax_a,
        calms_pair,
        calms_annot,
        "CalMS21 Task 2 agreement",
    )

    plot_mars_pairwise(
        ax_b,
        mars_pair,
        mars_annot,
        "MARS1 agreement",
    )

    # Shared legend for panels a and b
    handles, labels = ax_a.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.965),
        ncol=2,
        frameon=False,
        fontsize=6,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    plot_heatmap(
        ax_c,
        calms_label,
        "CalMS21 label distribution",
        label_cmap,
        "Fraction",
    )

    plot_heatmap(
        ax_d,
        mars_positive,
        "MARS1 positive-frame fraction",
        label_cmap,
        "Positive fraction",
    )

    plot_heatmap(
        ax_e,
        calms_bout,
        "CalMS21 bout duration",
        bout_cmap,
        "Mean duration (s)",
    )

    plot_heatmap(
        ax_f,
        mars_bout,
        "MARS1 bout duration",
        bout_cmap,
        "Mean duration (s)",
    )

    add_panel_letter(ax_a, "a", x=-0.18, y=1.04)
    add_panel_letter(ax_b, "b", x=-0.18, y=1.04)
    add_panel_letter(ax_c, "c", x=-0.18, y=1.10)
    add_panel_letter(ax_d, "d", x=-0.18, y=1.10)
    add_panel_letter(ax_e, "e", x=-0.18, y=1.10)
    add_panel_letter(ax_f, "f", x=-0.18, y=1.10)

    # No global title.

    fig.subplots_adjust(
        left=0.09,
        right=0.965,
        bottom=0.075,
        top=0.925,
        wspace=0.45,
        hspace=0.72,
    )

    out_png = FIGDIR / "combined_calms21_mars1_annotator_bias_shared_legend.png"
    out_pdf = FIGDIR / "combined_calms21_mars1_annotator_bias_shared_legend.pdf"
    out_svg = FIGDIR / "combined_calms21_mars1_annotator_bias_shared_legend.svg"

    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    fig.savefig(out_svg)

    plt.show()

print("Saved:")
print(out_png)
print(out_pdf)
print(out_svg)
print(FIGDIR / "combined_annotator_bias_summary_shared_legend.csv")