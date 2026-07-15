# Auxiliary-task assessment and annotator/bout analyses

This directory contains code associated with two additions made during the
LISBET manuscript revision:

1. **CalMS21 leave-one-task-out decoding**: five matched encoder seeds for the
   full auxiliary-task model and the four leave-one-task-out variants, followed
   by frozen-embedding k-nearest-neighbor decoding and all-seed aggregation.
2. **CalMS21 Task 2 and MARS1 annotator/bout characterization**: summary-level
   agreement, label-prevalence and bout-duration tables together with the code
   used to reproduce the final six-panel figure.

## CalMS21 leave-one-task-out code

The five `reviewer_2_2_calms21_pipeline_seed*.py` files are exact copies of the
seed-specific analysis scripts recovered from the Dell code archive. Each
script accepts `--lisbet-root`; use that argument rather than relying on its
historical default directory.

Example for seed 0:

```bash
python scripts/reviewer_2_2_calms21_pipeline_seed0.py all \
  --lisbet-root /path/to/Lisbet
```

The all-seed aggregation script was changed only to replace one private,
hard-coded source path with explicit command-line arguments:

```bash
python scripts/plot_calms21_posthoc_allseeds.py \
  --root /path/containing/reviewer_2_2_calms21_posthoc_seed0_to_seed4
```

The seed-level and aggregate CalMS21 result CSVs were not present in the
uploaded source archives, so they are not included here. No result table was
reconstructed from manuscript text.

## Annotator and bout figure

Run from any directory:

```bash
python annotator_bout_metrics/scripts/plot_combined_calms21_mars1_annotator_bias.py
```

The script reads the included summary CSVs and writes the PNG, PDF, SVG and
combined summary CSV to `annotator_bout_metrics/figures/`.

The available source material contained plotting code and derived summary
CSVs, but not the upstream script that originally calculated all agreement and
bout tables from raw annotations. Accordingly, this folder supports exact
figure reproduction from the preserved summary tables but does not claim to
recreate those tables from raw CalMS21 or MARS1 annotations.

Raw annotations, pose data, embeddings, model weights and private machine paths
are intentionally excluded.
