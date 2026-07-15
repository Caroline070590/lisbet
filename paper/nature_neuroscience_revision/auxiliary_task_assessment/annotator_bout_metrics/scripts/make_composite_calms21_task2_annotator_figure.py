from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

OUTDIR = Path("results/calms21_task2_annotator_bias")
FIGDIR = Path("figures/calms21_task2_annotator_bias")
FIGDIR.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(OUTDIR / "calms21_task2_pairwise_interannotator_summary.csv")
annot = pd.read_csv(OUTDIR / "calms21_task2_annotator_file_summary.csv")
label = pd.read_csv(OUTDIR / "calms21_task2_label_distribution_by_annotator.csv")
bout = pd.read_csv(OUTDIR / "calms21_task2_bout_stats_by_annotator.csv")

summary["pair"] = (
    summary["annotator_a"].str.replace("annotator", "A", regex=False)
    + "-"
    + summary["annotator_b"].str.replace("annotator", "A", regex=False)
)

# Heatmap tables
label_pivot = label.pivot_table(
    index="behavior",
    columns="annotator",
    values="fraction_within_annotator",
    fill_value=0,
)
label_pivot = label_pivot.loc[label_pivot.mean(axis=1).sort_values(ascending=False).index]

bout_pivot = bout.pivot_table(
    index="behavior",
    columns="annotator",
    values="mean_bout_duration_s",
    fill_value=np.nan,
)
bout_pivot = bout_pivot.loc[bout_pivot.mean(axis=1).sort_values(ascending=False).index]

# ---- Figure ----
fig = plt.figure(figsize=(12, 9))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], width_ratios=[1, 1])

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# ------------------------
# Panel a: annotation coverage
# ------------------------
ax_a.bar(annot["annotator"], annot["total_frames"])
ax_a.set_ylabel("Annotated frames")
ax_a.set_xlabel("Annotator")
ax_a.set_title("Annotation coverage")
ax_a.tick_params(axis="x", rotation=45)

# ------------------------
# Panel b: combined agreement
# ------------------------
x = np.arange(len(summary))
width = 0.35
ax_b.bar(x - width/2, summary["macro_f1_mean"], width, label="Macro-F1")
ax_b.bar(x + width/2, summary["cohen_kappa_mean"], width, label="Cohen's kappa")
ax_b.axhline(0, linewidth=1)
ax_b.set_xticks(x)
ax_b.set_xticklabels(summary["pair"], rotation=45, ha="right")
ax_b.set_ylabel("Agreement score")
ax_b.set_xlabel("Annotator pair")
ax_b.set_title("Pairwise inter-annotator agreement")
ax_b.set_ylim(-0.15, 1.0)
ax_b.legend(frameon=False)

# ------------------------
# Panel c: label distribution heatmap
# ------------------------
im1 = ax_c.imshow(label_pivot.values, aspect="auto")
ax_c.set_xticks(range(len(label_pivot.columns)))
ax_c.set_xticklabels(label_pivot.columns, rotation=45, ha="right")
ax_c.set_yticks(range(len(label_pivot.index)))
ax_c.set_yticklabels(label_pivot.index)
ax_c.set_title("Behavior label distribution")
ax_c.set_xlabel("Annotator")
ax_c.set_ylabel("Behavior")
cbar1 = fig.colorbar(im1, ax=ax_c, fraction=0.046, pad=0.04)
cbar1.set_label("Fraction within annotator")

# ------------------------
# Panel d: bout duration heatmap
# ------------------------
im2 = ax_d.imshow(bout_pivot.values, aspect="auto")
ax_d.set_xticks(range(len(bout_pivot.columns)))
ax_d.set_xticklabels(bout_pivot.columns, rotation=45, ha="right")
ax_d.set_yticks(range(len(bout_pivot.index)))
ax_d.set_yticklabels(bout_pivot.index)
ax_d.set_title("Mean bout duration")
ax_d.set_xlabel("Annotator")
ax_d.set_ylabel("Behavior")
cbar2 = fig.colorbar(im2, ax=ax_d, fraction=0.046, pad=0.04)
cbar2.set_label("Mean bout duration (s)")

# Panel letters
for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], ["a", "b", "c", "d"]):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="left")

# Remove top/right spines for cleaner journal style
for ax in [ax_a, ax_b, ax_c, ax_d]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(FIGDIR / "calms21_task2_annotator_bias_composite.png", dpi=600)
plt.savefig(FIGDIR / "calms21_task2_annotator_bias_composite.pdf")
plt.close()

print("Saved:")
print(FIGDIR / "calms21_task2_annotator_bias_composite.png")
print(FIGDIR / "calms21_task2_annotator_bias_composite.pdf")
