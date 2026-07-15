# Provenance

## Recovered sources

- Seed-specific CalMS21 decoder scripts and the all-seed plotting script were
  recovered from `task_ablation_code_collection_dell.tar.gz`.
- Annotator/bout summary CSVs, final figures and the plotting notebook were
  recovered from `auxiliary_task_final_collection.tar.gz`.

## Transformations

- The five seed-specific CalMS21 scripts are byte-for-byte copies of their
  recovered sources.
- `plot_calms21_posthoc_allseeds.py` preserves the recovered aggregation and
  plotting logic. Its single historical hard-coded absolute root was replaced
  with `--root` and optional `--output` arguments.
- `plot_combined_calms21_mars1_annotator_bias.py` was extracted from the final
  executed cell of `annotator-bias.ipynb`. Absolute paths were replaced with
  package-relative paths, and notebook-only `display()` calls were changed to
  terminal printing. The calculations, panel definitions and plotting
  parameters were otherwise preserved.
- The original notebook is not included because it contains many duplicated
  exploratory cells, embedded outputs and private absolute paths.
- No scientific result was synthesized, recalculated from prose or manually
  entered into a new table.

## Known limitation

The original metric-generation script for the CalMS21/MARS1 annotator and bout
summary CSVs was not found in the supplied code collections. The preserved
summary tables and final plotting implementation are included transparently.
