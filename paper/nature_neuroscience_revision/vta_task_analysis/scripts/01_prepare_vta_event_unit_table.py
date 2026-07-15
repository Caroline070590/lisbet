#!/usr/bin/env python3
"""Prepare the event-level and event-unit VTA tables used in the manuscript.

This file is a path-portable split of the preparation sections from the
actual analysis script:
run_vta_lisbet_posthoc_conditionA_same_as_yours_fixed.py

The event construction, duration filtering, z-scoring, and prototype filtering
logic are unchanged. Only input/output paths were made configurable through
environment variables.
"""
from pathlib import Path
import os
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

vta_root_value = os.environ.get("VTA_ROOT")
if not vta_root_value:
    raise SystemExit(
        "VTA_ROOT must be set to the restricted directory containing "
        "the VTA condition folders."
    )

VTA_ROOT = Path(vta_root_value).expanduser().resolve()

# Match Giuseppe's VTA notebook: DATA_FILTER = "Condition_A"
# This keeps the analysis exactly on the Figure 6 subset.
TARGET_CONDITION = os.environ.get("TARGET_CONDITION", "Condition_A")

OUTDIR = Path(os.environ.get(
    "OUTDIR",
    Path.cwd() / "vta_posthoc_results",
)).expanduser().resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

MIN_EVENT_DURATION_S = 0.2
MAX_EVENT_DURATION_S = 2.0

# Minimum valid observations per prototype after converting to long neuron format.
# This is not hand-picking; it avoids unstable model parameters for prototypes with almost no data.
MIN_OBS_PER_PROTOTYPE = 30

print("=" * 80)
print("VTA LISBET analysis")
print("VTA_ROOT:", VTA_ROOT)
print("TARGET_CONDITION:", TARGET_CONDITION)
print("OUTDIR:", OUTDIR)
print("=" * 80)

if not VTA_ROOT.exists():
    raise SystemExit(f"VTA_ROOT not found: {VTA_ROOT}")

# ============================================================
# Helper functions
# ============================================================

def zscore_safe(x):
    x = pd.Series(x)
    if x.notna().sum() < 2:
        return pd.Series(np.nan, index=x.index)
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def infer_condition_from_path(path):
    parts = path.parts
    for p in parts:
        if p.startswith("Condition_"):
            return p
    return "unknown"


def build_event_table_for_trial(trial_dir):
    features_path = trial_dir / "features_metrics.csv"
    states_path = trial_dir / "hmmbet_state_predictions.csv"

    if not features_path.exists() or not states_path.exists():
        return None, "missing_required_files"

    try:
        features = pd.read_csv(features_path)
        states = pd.read_csv(states_path)
    except Exception as e:
        return None, f"read_error: {e}"

    required_feature_cols = ["timing", "velocity", "distance", "angle"]
    missing = [c for c in required_feature_cols if c not in features.columns]
    if missing:
        return None, f"missing_feature_columns: {missing}"

    action_cols = [c for c in states.columns if c.startswith("Action_")]
    if not action_cols:
        return None, "missing_Action_columns"

    # Safety fix: sort Action_ columns numerically.
    # This prevents Action_10 being ordered before Action_2 if column order changes.
    def action_number(col):
        m = re.search(r"(\d+)$", col)
        return int(m.group(1)) if m else 10**9

    action_cols = sorted(action_cols, key=action_number)

    n = min(len(features), len(states))
    features = features.iloc[:n].reset_index(drop=True)
    states = states.iloc[:n].reset_index(drop=True)

    features["frame"] = np.arange(n)

    # Convert one-hot HMM/LISBET state into prototype ID
    features["prototype_id"] = states[action_cols].values.argmax(axis=1)

    # Infer FPS from timing
    dt = features["timing"].diff().median()
    if pd.isna(dt) or dt <= 0:
        fps = 25.0
    else:
        fps = 1.0 / dt

    # Consecutive runs of the same prototype are events
    features["event_id_local"] = (
        features["prototype_id"] != features["prototype_id"].shift()
    ).cumsum()

    neuron_cols = [c for c in features.columns if c.startswith("neuron_")]

    rows = []
    for event_id, g in features.groupby("event_id_local", sort=True):
        start_frame = int(g["frame"].iloc[0])
        end_frame = int(g["frame"].iloc[-1])
        duration_s = float((end_frame - start_frame + 1) / fps)

        row = {
            "trial_id": trial_dir.name,
            "condition": infer_condition_from_path(trial_dir),
            "trial_path": str(trial_dir),
            "event_id_local": int(event_id),
            "prototype_id": int(g["prototype_id"].iloc[0]),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": float(g["timing"].iloc[0]),
            "end_time_s": float(g["timing"].iloc[-1]),
            "duration_s": duration_s,
            "n_frames": int(len(g)),
            "fps_estimated": float(fps),
            "velocity_mean": float(g["velocity"].mean()),
            "distance_mean": float(g["distance"].mean()),
            "angle_mean": float(g["angle"].mean()),
            "acceleration_mean": float(g["acceleration"].mean()) if "acceleration" in g.columns else np.nan,
            "contact_reciprocal_mean": float(g["contact_reciprocal"].mean()) if "contact_reciprocal" in g.columns else np.nan,
            "contact_unilateral_mean": float(g["contact_unilateral"].mean()) if "contact_unilateral" in g.columns else np.nan,
            "contact_passive_mean": float(g["contact_passive"].mean()) if "contact_passive" in g.columns else np.nan,
        }

        for nc in neuron_cols:
            row[f"{nc}_mean"] = float(g[nc].mean()) if g[nc].notna().any() else np.nan

        rows.append(row)

    events = pd.DataFrame(rows)

    events["include_methods_duration_filter"] = events["duration_s"].between(
        MIN_EVENT_DURATION_S,
        MAX_EVENT_DURATION_S,
        inclusive="both"
    )

    events["exclusion_reason"] = ""
    events.loc[events["duration_s"] < MIN_EVENT_DURATION_S, "exclusion_reason"] = "shorter_than_200ms"
    events.loc[events["duration_s"] > MAX_EVENT_DURATION_S, "exclusion_reason"] = "longer_than_2s"

    return events, "ok"


# ============================================================
# 1. Find all VTA trial folders automatically
# ============================================================

trial_dirs = sorted({
    p.parent for p in VTA_ROOT.rglob("features_metrics.csv")
    if (p.parent / "hmmbet_state_predictions.csv").exists()
})

print(f"Found {len(trial_dirs)} trial folders with required files before condition filter.")

# Safety fix: explicitly reproduce Giuseppe's notebook filter.
trial_dirs = [d for d in trial_dirs if TARGET_CONDITION in d.parts]

print(f"Found {len(trial_dirs)} trial folders after {TARGET_CONDITION} filter.")

if len(trial_dirs) == 0:
    raise SystemExit(f"No valid trial folders found for {TARGET_CONDITION}.")

# ============================================================
# 2. Build combined event-level table
# ============================================================

all_events = []
trial_status = []

for i, trial_dir in enumerate(trial_dirs, start=1):
    print(f"[{i}/{len(trial_dirs)}] {trial_dir.name}")
    events, status = build_event_table_for_trial(trial_dir)

    trial_status.append({
        "trial_id": trial_dir.name,
        "trial_path": str(trial_dir),
        "status": status,
        "n_events_total": 0 if events is None else len(events),
        "n_events_included": 0 if events is None else int(events["include_methods_duration_filter"].sum())
    })

    if events is not None and len(events) > 0:
        all_events.append(events)

trial_status_df = pd.DataFrame(trial_status)
trial_status_df.to_csv(OUTDIR / "trial_status_summary.csv", index=False)

if not all_events:
    raise SystemExit("No event tables could be built.")

event_table = pd.concat(all_events, ignore_index=True)
event_table["global_event_id"] = np.arange(len(event_table))

# Final guard: the output must contain only the target condition.
condition_counts = event_table["condition"].value_counts(dropna=False)
print("\nCondition counts in event_table:")
print(condition_counts)

unexpected_conditions = set(event_table["condition"].dropna().unique()) - {TARGET_CONDITION}
if unexpected_conditions:
    raise SystemExit(f"Unexpected conditions in event_table: {unexpected_conditions}")

event_table.to_csv(OUTDIR / "figure6_all_trials_event_table.csv", index=False)

print("\nSaved event table:")
print(OUTDIR / "figure6_all_trials_event_table.csv")
print("Total events:", len(event_table))
print("Included events:", int(event_table["include_methods_duration_filter"].sum()))

# ============================================================
# 3. Convert to long format: one row per event x neuron
# ============================================================

valid_events = event_table[event_table["include_methods_duration_filter"]].copy()

neuron_mean_cols = [c for c in valid_events.columns if c.startswith("neuron_") and c.endswith("_mean")]

long_rows = []

for _, row in valid_events.iterrows():
    for nc in neuron_mean_cols:
        value = row[nc]
        if pd.isna(value):
            continue

        neuron_name = nc.replace("_mean", "")
        unit_id = f"{row['trial_id']}__{neuron_name}"

        long_rows.append({
            "trial_id": row["trial_id"],
            "condition": row["condition"],
            "trial_path": row["trial_path"],
            "global_event_id": row["global_event_id"],
            "event_id_local": row["event_id_local"],
            "prototype_id": int(row["prototype_id"]),
            "unit_id": unit_id,
            "neuron_name": neuron_name,
            "neural_activity": float(value),
            "velocity_mean": row["velocity_mean"],
            "distance_mean": row["distance_mean"],
            "angle_mean": row["angle_mean"],
            "acceleration_mean": row["acceleration_mean"],
            "duration_s": row["duration_s"],
            "n_frames": row["n_frames"],
            "start_time_s": row["start_time_s"],
            "end_time_s": row["end_time_s"],
        })

long_df = pd.DataFrame(long_rows)

if long_df.empty:
    raise SystemExit("No non-missing neuron observations found after event filtering.")

# Z-score within each neuron unit so units are comparable
long_df["neural_z"] = long_df.groupby("unit_id")["neural_activity"].transform(zscore_safe)

# Z-score behavior within trial to reduce trial-level scale differences
long_df["velocity_z"] = long_df.groupby("trial_id")["velocity_mean"].transform(zscore_safe)
long_df["distance_z"] = long_df.groupby("trial_id")["distance_mean"].transform(zscore_safe)
long_df["angle_z"] = long_df.groupby("trial_id")["angle_mean"].transform(zscore_safe)

long_df = long_df.dropna(subset=["neural_z", "velocity_z", "distance_z", "prototype_id", "unit_id"]).copy()

# Remove prototypes with too few observations
proto_counts = long_df["prototype_id"].value_counts()
valid_protos = proto_counts[proto_counts >= MIN_OBS_PER_PROTOTYPE].index
long_df["prototype_id_original"] = long_df["prototype_id"]
long_df = long_df[long_df["prototype_id"].isin(valid_protos)].copy()

long_df["prototype_id"] = long_df["prototype_id"].astype("category")
long_df["unit_id"] = long_df["unit_id"].astype("category")
long_df["trial_id"] = long_df["trial_id"].astype("category")

print("\nCondition counts in long_df:")
print(long_df["condition"].value_counts(dropna=False))

long_df.to_csv(OUTDIR / "figure6_all_trials_event_neuron_long_valid.csv", index=False)

print("\nSaved long table:")
print(OUTDIR / "figure6_all_trials_event_neuron_long_valid.csv")
print("Long observations:", len(long_df))
print("Units:", long_df["unit_id"].nunique())
print("Trials:", long_df["trial_id"].nunique())
print("Prototype counts after filtering:")
print(long_df["prototype_id"].value_counts().sort_index())


print("\nPreparation complete.")
