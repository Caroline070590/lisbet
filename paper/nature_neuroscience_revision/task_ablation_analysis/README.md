# LISBET task-combination ablation

This directory contains the code and configuration records for the LISBET
self-supervised task-combination ablation used in the Nature Neuroscience
manuscript revision.

## Experiment represented here

The authoritative Baobab export contains 75 runs: all 15 non-empty
combinations of the four LISBET self-supervised objectives (`cons`, `order`,
`shift`, and `warp`) evaluated across five seeds. The training matrix contains
75 data rows plus one header row. All 75 exported `model_config.yml` files were
checked against the matrix and matched the expected task combination and seed.
The exported completion record contains 75 completed run logs.

Training used a 200-frame window, 75 epochs, batch size 32, learning rate
0.0001, and the LISBET `betman train_model` command. The prepared training data,
model weights, training histories, and raw logs are not included.

## Contents

- `jobs/task_full_combo_75ep_matrix.tsv`: exact task/seed matrix from Baobab.
- `jobs/*.sbatch`: public-release copies of the three executed launchers. Only
  machine-specific paths and log destinations were replaced by required
  environment variables; resource requests, array ranges, coordinate mapping,
  task arguments, and training parameters are unchanged.
- `model_configs/`: exact 75 exported model configurations.
- `scripts/plot_ab_training_summary_only.py`: release copy of the manuscript
  training-dynamics plotting script. Only machine-specific input/output paths
  were replaced by command-line arguments; calculations and visual settings
  are unchanged.
- `environment/`: Python version and Conda explicit-package capture exported
  from the Baobab `lisbet_env` environment on 15 July 2026.
- `metadata/`: audit summary and source/release checksums.

## Submit the training array

Set the paths required by the selected launcher:

```bash
export DATA=/path/to/prepared_lisbet_training_data
export OUTBASE=/path/to/task_ablation_results
export BETMAN=/path/to/betman  # optional when betman is on PATH

sbatch jobs/task_full_combo_75ep_array_private_gpu.sbatch
```

The public- and shared-GPU launchers preserve the resource settings and array
ranges used for the corresponding Baobab submissions.

## Plot the manuscript training panels

The plotting script uses the full-task model and the four leave-one-task-out
triple-task variants from the 75-run result directory. It averages the task
heads available in each model and reports mean ± SEM across the five seeds.

```bash
python scripts/plot_ab_training_summary_only.py   --run-root /path/to/task_full_combinations_w200_75ep_train1600   --output-dir /path/to/figures
```

The script writes PDF, PNG, and SVG versions of
`ab_training_summary_mean_sem`.

## Data and results availability

This directory does not contain the prepared behavioral dataset, model
weights, per-epoch training histories, or restricted cluster logs. Those files
must be obtained under the data-access conditions specified by the manuscript
and project owners.

The downstream CalMS21 frozen-embedding/kNN evaluation is not included here;
it belongs to the separate auxiliary-task evaluation release.
