# Provenance and verification record

This record distinguishes original supplied files from reorganized release files.

## Source identity

The released preparation and model-comparison scripts were reorganized from `run_vta_lisbet_posthoc_conditionA_same_as_yours_fixed.py`, which corresponds to the Condition_A analysis summarized in the archived outputs.

## Code mapping

- `01_prepare_vta_event_unit_table.py`: extracted from lines covering event preparation in `run_vta_lisbet_posthoc_conditionA_same_as_yours_fixed.py`. Changes are limited to a shebang/docstring, configurable paths, and additional terminal summaries.
- `02_run_vta_model_comparison.py`: extracted from the model-comparison/residual-analysis section of the same source script and wrapped in `main()` with command-line arguments and input-column validation.
- `03_run_vta_robustness_analyses.py`: exact copy of `run_vta_lisbet_residual_robustness_fast.py`.
- `04_plot_vta_figure.ipynb`: exact final plotting code cell from `vta-plots.ipynb`, with outputs removed.
- `lisbet.mplstyle`: exact copy of the supplied style file.

## Verification performed

- All three Python scripts passed Python syntax compilation.
- The plotting notebook is valid JSON, contains one retained code cell, and contains no saved outputs.
- Running `02_run_vta_model_comparison.py` on the supplied restricted event-unit table reproduced the archived primary model values, including ΔR² = 0.0013602786226875052 and ΔAIC = 29.0221885455976.
- The residual formulas reproduced residual ΔR² = 0.00104127633630835 and ΔAIC = 19.262899785040645.
- `03_run_vta_robustness_analyses.py` completed a smoke test on the supplied event-unit table and reproduced the main nested-model F-test: F = 6.839883, df = (6, 30000), p = 2.877355e-07.
- The public package contains no raw recordings, row-level event tables, original trial identifiers, or private NAS paths.

## Important limits

The full 5,000-bootstrap/5,000-permutation run was not re-executed during this packaging audit because its exact archived outputs and run log were already supplied. The released robustness script is byte-for-byte identical to the archived executed script.

The exact original execution-environment export was not available in the supplied archive and is therefore not included in this release.
