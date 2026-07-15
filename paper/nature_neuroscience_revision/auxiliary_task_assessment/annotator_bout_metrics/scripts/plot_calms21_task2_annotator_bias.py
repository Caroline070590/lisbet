from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# ------------------------------------------------------------
# Plot 1: annotated frames per annotator
# ------------------------------------------------------------
plt.figure(figsize=(5.5, 4))
plt.bar(annot["annotator"], annot["total_frames"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Annotated frames")
plt.xlabel("Annotator")
plt.title("CalMS21 Task 2 annotation coverage")
plt.tight_layout()
plt.savefig(FIGDIR / "annotated_frames_per_annotator.png", dpi=600)
plt.savefig(FIGDIR / "annotated_frames_per_annotator.pdf")
plt.close()

# ------------------------------------------------------------
# Plot 2: pairwise Macro-F1
# ------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.bar(summary["pair"], summary["macro_f1_mean"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Macro-F1")
plt.xlabel("Annotator pair")
plt.title("Pairwise inter-annotator agreement")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(FIGDIR / "pairwise_macro_f1.png", dpi=600)
plt.savefig(FIGDIR / "pairwise_macro_f1.pdf")
plt.close()

# ------------------------------------------------------------
# Plot 3: pairwise Cohen's kappa
# ------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.bar(summary["pair"], summary["cohen_kappa_mean"])
plt.axhline(0, linewidth=1)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Cohen's kappa")
plt.xlabel("Annotator pair")
plt.title("Pairwise inter-annotator reliability")
plt.ylim(-0.15, 0.85)
plt.tight_layout()
plt.savefig(FIGDIR / "pairwise_cohen_kappa.png", dpi=600)
plt.savefig(FIGDIR / "pairwise_cohen_kappa.pdf")
plt.close()

# ------------------------------------------------------------
# Plot 4: behavior label distribution by annotator
# ------------------------------------------------------------
pivot = label.pivot_table(
    index="behavior",
    columns="annotator",
    values="fraction_within_annotator",
    fill_value=0,
)
pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

plt.figure(figsize=(6.5, max(4, 0.35 * len(pivot))))
plt.imshow(pivot.values, aspect="auto")
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
plt.yticks(range(len(pivot.index)), pivot.index)
plt.colorbar(label="Fraction within annotator")
plt.title("Behavior label distribution by annotator")
plt.tight_layout()
plt.savefig(FIGDIR / "label_distribution_by_annotator.png", dpi=600)
plt.savefig(FIGDIR / "label_distribution_by_annotator.pdf")
plt.close()

# ------------------------------------------------------------
# Plot 5: mean bout duration by annotator
# ------------------------------------------------------------
bout_pivot = bout.pivot_table(
    index="behavior",
    columns="annotator",
    values="mean_bout_duration_s",
    fill_value=np.nan,
)
bout_pivot = bout_pivot.loc[bout_pivot.mean(axis=1).sort_values(ascending=False).index]

plt.figure(figsize=(6.5, max(4, 0.35 * len(bout_pivot))))
plt.imshow(bout_pivot.values, aspect="auto")
plt.xticks(range(len(bout_pivot.columns)), bout_pivot.columns, rotation=45, ha="right")
plt.yticks(range(len(bout_pivot.index)), bout_pivot.index)
plt.colorbar(label="Mean bout duration (s)")
plt.title("Mean bout duration by annotator")
plt.tight_layout()
plt.savefig(FIGDIR / "bout_duration_by_annotator.png", dpi=600)
plt.savefig(FIGDIR / "bout_duration_by_annotator.pdf")
plt.close()

# ------------------------------------------------------------
# Plot 6: combined agreement figure, better for reviewer/rebuttal
# ------------------------------------------------------------
x = np.arange(len(summary))
width = 0.35

plt.figure(figsize=(8, 4.5))
plt.bar(x - width / 2, summary["macro_f1_mean"], width, label="Macro-F1")
plt.bar(x + width / 2, summary["cohen_kappa_mean"], width, label="Cohen's kappa")
plt.axhline(0, linewidth=1)
plt.xticks(x, summary["pair"], rotation=45, ha="right")
plt.ylabel("Agreement score")
plt.xlabel("Annotator pair")
plt.title("CalMS21 Task 2 annotator-style variability")
plt.legend(frameon=False)
plt.ylim(-0.15, 1.0)
plt.tight_layout()
plt.savefig(FIGDIR / "combined_pairwise_agreement.png", dpi=600)
plt.savefig(FIGDIR / "combined_pairwise_agreement.pdf")
plt.close()

print("Saved plots:")
for p in sorted(FIGDIR.glob("*")):
    print(p)
