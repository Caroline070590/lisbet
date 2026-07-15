from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

parser = argparse.ArgumentParser(
    description="Aggregate five matched CalMS21 decoding runs and reproduce the all-seed figure."
)
parser.add_argument(
    "--root",
    type=Path,
    required=True,
    help="Directory containing reviewer_2_2_calms21_posthoc_seed0 through seed4",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output directory; defaults to <root>/reviewer_2_2_calms21_posthoc_allseeds",
)
args = parser.parse_args()
ROOT = args.root.expanduser().resolve()
OUT = (args.output.expanduser().resolve() if args.output else ROOT / "reviewer_2_2_calms21_posthoc_allseeds")
FIG = OUT / "figures"
RES = OUT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

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

MODEL_ORDER = ["all_tasks", "without_cons", "without_order", "without_shift", "without_warp"]
MODEL_LABELS = ["All tasks", "Without cons", "Without order", "Without shift", "Without warp"]
TASK_ORDER = ["cons", "order", "shift", "warp"]
CLASS_ORDER = ["attack", "investigation", "mount", "other"]

label_map = dict(zip(MODEL_ORDER, MODEL_LABELS))

summary_rows = []
class_rows = []

for seed in range(5):
    base = ROOT / f"reviewer_2_2_calms21_posthoc_seed{seed}" / "results"

    s = pd.read_csv(base / "calms21_model_summary.csv")
    s["seed"] = seed
    summary_rows.append(s)

    c = pd.read_csv(base / "calms21_per_class_metrics.csv")
    c["seed"] = seed
    class_rows.append(c)

summary = pd.concat(summary_rows, ignore_index=True)
perclass = pd.concat(class_rows, ignore_index=True)

summary.to_csv(RES / "allseed_calms21_model_summary_long.csv", index=False)
perclass.to_csv(RES / "allseed_calms21_per_class_metrics_long.csv", index=False)

# Aggregate overall metrics
overall = (
    summary
    .groupby(["model", "label", "removed_task"], as_index=False)
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
    )
)

overall["model"] = pd.Categorical(overall["model"], MODEL_ORDER, ordered=True)
overall = overall.sort_values("model")
overall.to_csv(RES / "allseed_calms21_model_summary_mean_sd.csv", index=False)

# Per-class F1 mean
pc = (
    perclass
    .groupby(["model", "label", "class_name"], as_index=False)
    .agg(f1_mean=("f1", "mean"), f1_sd=("f1", "std"))
)

pc["model"] = pd.Categorical(pc["model"], MODEL_ORDER, ordered=True)
pc["class_name"] = pd.Categorical(pc["class_name"], CLASS_ORDER, ordered=True)
pc = pc.sort_values(["class_name", "model"])
pc.to_csv(RES / "allseed_calms21_per_class_f1_mean_sd.csv", index=False)

# Task-removal effect per seed/class: all_tasks - without_task
effect_rows = []

for seed in range(5):
    sub = perclass[perclass["seed"] == seed].copy()
    all_f1 = sub[sub["model"] == "all_tasks"].set_index("class_name")["f1"]

    for task in TASK_ORDER:
        model = f"without_{task}"
        w = sub[sub["model"] == model].set_index("class_name")["f1"]

        for cls in CLASS_ORDER:
            if cls in all_f1.index and cls in w.index:
                effect_rows.append({
                    "seed": seed,
                    "removed_task": task,
                    "class_name": cls,
                    "delta_f1": all_f1.loc[cls] - w.loc[cls],
                })

effects = pd.DataFrame(effect_rows)
effects.to_csv(RES / "allseed_task_removal_effects_long.csv", index=False)

eff = (
    effects
    .groupby(["class_name", "removed_task"], as_index=False)
    .agg(delta_f1_mean=("delta_f1", "mean"), delta_f1_sd=("delta_f1", "std"))
)

eff["class_name"] = pd.Categorical(eff["class_name"], CLASS_ORDER, ordered=True)
eff["removed_task"] = pd.Categorical(eff["removed_task"], TASK_ORDER, ordered=True)
eff = eff.sort_values(["class_name", "removed_task"])
eff.to_csv(RES / "allseed_task_removal_effects_mean_sd.csv", index=False)

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)

def panel_label(ax, label):
    ax.text(-0.15, 1.08, label, transform=ax.transAxes,
            fontsize=12, fontweight="normal", ha="left", va="top")

fig, axes = plt.subplots(
    1, 3,
    figsize=(11.2, 3.2),
    gridspec_kw={"width_ratios": [1.5, 1.25, 1.1], "wspace": 0.50}
)

ax_c, ax_d, ax_e = axes

# Panel c: per-behavior decoding
mat = (
    pc.pivot(index="class_name", columns="label", values="f1_mean")
    .reindex(index=CLASS_ORDER, columns=MODEL_LABELS)
)

im = ax_c.imshow(mat.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
ax_c.set_title("Per-behavior decoding")
ax_c.set_ylabel("CalMS21 behavior")
ax_c.set_xlabel("Frozen LISBET encoder")
ax_c.set_xticks(np.arange(len(MODEL_LABELS)))
ax_c.set_xticklabels(MODEL_LABELS, rotation=40, ha="right")
ax_c.set_yticks(np.arange(len(CLASS_ORDER)))
ax_c.set_yticklabels(CLASS_ORDER)

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat.values[i, j]
        color = "white" if val < 0.45 else "black"
        ax_c.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.025)
cbar.set_label("F1")
panel_label(ax_c, "c")
clean_axis(ax_c)

# Panel d: task-removal effect
emat = (
    eff.pivot(index="class_name", columns="removed_task", values="delta_f1_mean")
    .reindex(index=CLASS_ORDER, columns=TASK_ORDER)
)

v = np.nanmax(np.abs(emat.values))
v = max(v, 0.01)
im2 = ax_d.imshow(emat.values, aspect="auto", cmap="coolwarm", vmin=-v, vmax=v)
ax_d.set_title("Task-removal effect")
ax_d.set_ylabel("CalMS21 behavior")
ax_d.set_xlabel("Removed task")
ax_d.set_xticks(np.arange(len(TASK_ORDER)))
ax_d.set_xticklabels(TASK_ORDER)
ax_d.set_yticks(np.arange(len(CLASS_ORDER)))
ax_d.set_yticklabels(CLASS_ORDER)

for i in range(emat.shape[0]):
    for j in range(emat.shape[1]):
        val = emat.values[i, j]
        ax_d.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=7, color="black")

cbar2 = fig.colorbar(im2, ax=ax_d, fraction=0.046, pad=0.025)
cbar2.set_label(r"$F1_{all} - F1_{without}$")
panel_label(ax_d, "d")
clean_axis(ax_d)

# Panel e: overall decoding across seeds
x = np.arange(len(overall))
offset = 0.09

ax_e.errorbar(
    x - offset,
    overall["macro_f1_mean"],
    yerr=overall["macro_f1_sd"],
    fmt="o",
    color="#222222",
    ecolor="#222222",
    capsize=3,
    markersize=4,
    label="Macro-F1",
)

ax_e.errorbar(
    x + offset,
    overall["balanced_accuracy_mean"],
    yerr=overall["balanced_accuracy_sd"],
    fmt="o",
    color="#1b8a3a",
    ecolor="#1b8a3a",
    capsize=3,
    markersize=4,
    label="Balanced accuracy",
)

ax_e.set_title("Overall behavioral decoding")
ax_e.set_ylabel("Test performance")
ax_e.set_xlabel("Frozen LISBET encoder")
ax_e.set_xticks(x)
ax_e.set_xticklabels(MODEL_LABELS, rotation=40, ha="right")

vals = np.r_[overall["macro_f1_mean"], overall["balanced_accuracy_mean"]]
errs = np.r_[overall["macro_f1_sd"].fillna(0), overall["balanced_accuracy_sd"].fillna(0)]
ax_e.set_ylim(np.nanmin(vals - errs) - 0.02, np.nanmax(vals + errs) + 0.02)

ax_e.legend(frameon=False, loc="best")
panel_label(ax_e, "e")
clean_axis(ax_e)

fig.savefig(FIG / "Fig_calms21_knn_task_ablation_posthoc_allseeds.png", dpi=600, bbox_inches="tight")
fig.savefig(FIG / "Fig_calms21_knn_task_ablation_posthoc_allseeds.pdf", dpi=600, bbox_inches="tight")
fig.savefig(FIG / "Fig_calms21_knn_task_ablation_posthoc_allseeds.svg", dpi=600, bbox_inches="tight")

print("Saved:")
print(FIG / "Fig_calms21_knn_task_ablation_posthoc_allseeds.png")
print(FIG / "Fig_calms21_knn_task_ablation_posthoc_allseeds.pdf")
print(FIG / "Fig_calms21_knn_task_ablation_posthoc_allseeds.svg")
