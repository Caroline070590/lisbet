# Provenance and audit notes

The training material was collected from the Baobab export
`task_ablation_baobab_export_20260715_224015`, generated from the experiment
identified as `task_full_combinations_w200_75ep_train1600`.

## Exact copies

The task matrix, all 75 `model_config.yml` files, Python-version file, and Conda
explicit-package capture are copied without content changes. Their source and
release SHA-256 values are recorded in `metadata/source_checksums.csv`.

## Path-sanitized release copies

The three Slurm launchers preserve the original resource requests, array
ranges, coordinate selection/renaming, task selection, seed handling, epochs,
batch size, learning rate, and weight/history options. Private absolute paths
were replaced by the `DATA`, `OUTBASE`, `BETMAN`, and optional `MATRIX`
environment variables. Slurm log paths were changed to local filenames.

The plotting script is derived from the local executed manuscript script
`plot_ab_training_summary_only.py`. Its hard-coded machine paths were replaced
by `--run-root` and `--output-dir`; the aggregation, mean/SEM calculations,
model mapping, visual settings, and output formats are unchanged.

## Audit results

- Task matrix: 75 rows, 15 unique non-empty task combinations, five seeds each.
- Model configurations: 75 files.
- Matrix/configuration consistency: no mismatches in run label, seed,
  `model_id`, task heads, 200-frame window, or Transformer dimensions.
- Completion evidence: 75 completed log records in the private Baobab export.
- Python captured on Baobab: 3.10.20.

## Excluded candidates

The local `validate_w200_ablation.py` candidate was not included because its
model-ID regular expression requires names ending in `_w200_75ep`, whereas the
75 authoritative exported configurations use names ending in `_75ep`. As
written, that candidate would fail on the authoritative run set.

Exploratory notebooks, notebook checkpoints, 600-epoch experiments, CalMS21
pipelines, panel-e alternatives, and figure-merging scripts were excluded from
this task-ablation-only release. No Bamboo file was included because the
uploaded Bamboo collection was empty and the authoritative experiment records
were found on Baobab.
