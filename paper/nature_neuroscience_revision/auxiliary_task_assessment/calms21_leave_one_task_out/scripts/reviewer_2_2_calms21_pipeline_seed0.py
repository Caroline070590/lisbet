#!/usr/bin/env python3
"""Reviewer 2.2: LISBET leave-one-task-out analysis on CalMS21.

Stages
------
validate  Validate the five matched checkpoints and training histories.
prepare   Convert CalMS21 task 1 JSON files to multi-animal DLC CSV files.
embed     Compute frozen embeddings for train/test videos and all five models.
evaluate  Tune a fixed kNN probe on training videos and evaluate official test videos.
plot      Generate training, downstream, and combined figures.
all       Run all stages in sequence.

The downstream comparison uses the official CalMS21 task 1 train/test separation.
The k value is selected using only the all-task model and training videos, then held
fixed for every leave-one-task-out model. Confidence intervals resample test videos,
not frames.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


TASKS = ("cons", "order", "shift", "warp")
RUNS = {
    "all_tasks": {
        "folder": "all_tasks_seed0_75ep",
        "model_id": "all_tasks_seed0_75ep",
        "label": "All tasks",
        "removed": None,
    },
    "without_cons": {
        "folder": "triple_order_shift_warp_seed0_75ep",
        "model_id": "triple_order_shift_warp_seed0_75ep",
        "label": "Without cons",
        "removed": "cons",
    },
    "without_order": {
        "folder": "triple_cons_shift_warp_seed0_75ep",
        "model_id": "triple_cons_shift_warp_seed0_75ep",
        "label": "Without order",
        "removed": "order",
    },
    "without_shift": {
        "folder": "triple_cons_order_warp_seed0_75ep",
        "model_id": "triple_cons_order_warp_seed0_75ep",
        "label": "Without shift",
        "removed": "shift",
    },
    "without_warp": {
        "folder": "triple_cons_order_shift_seed0_75ep",
        "model_id": "triple_cons_order_shift_seed0_75ep",
        "label": "Without warp",
        "removed": "warp",
    },
}

RUN_COLORS = {
    "all_tasks": "#202020",
    "without_cons": "#4477AA",
    "without_order": "#EE6677",
    "without_shift": "#228833",
    "without_warp": "#CCBB44",
}


@dataclass(frozen=True)
class Paths:
    lisbet_root: Path
    ablation_root: Path
    calms_root: Path
    work: Path
    prepared: Path
    embedders: Path
    embeddings: Path
    results: Path
    figures: Path
    logs: Path

    @classmethod
    def from_root(cls, lisbet_root: Path) -> "Paths":
        lisbet_root = lisbet_root.expanduser().resolve()
        ablation_root = lisbet_root / "lisbet_task_ablation_w200_75ep_train1600_FULL_20260625_1624" / "results" / "auxiliary_task_assessment" / "task_full_combinations_w200_75ep_train1600"
        work = lisbet_root / "lisbet_task_ablation_w200_75ep_train1600_FULL_20260625_1624" / "reviewer_2_2_calms21_posthoc_seed0"
        return cls(
            lisbet_root=lisbet_root,
            ablation_root=ablation_root,
            calms_root=lisbet_root / "lisbet_datasets" / "datasets" / "CalMS21",
            work=work,
            prepared=work / "prepared_calms21_task1_dlc",
            embedders=work / "exported_embedders",
            embeddings=work / "embeddings",
            results=work / "results",
            figures=work / "figures",
            logs=work / "logs",
        )

    def create_outputs(self) -> None:
        for path in (
            self.work,
            self.prepared,
            self.embedders,
            self.embeddings,
            self.results,
            self.figures,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)


def model_paths(paths: Paths, run_key: str) -> tuple[Path, Path, Path]:
    run = RUNS[run_key]
    base = paths.ablation_root / run["folder"] / "models" / run["model_id"]
    return (
        base / "model_config.yml",
        base / "weights" / "weights_last.pt",
        base / "training_history" / "version_0" / "metrics.csv",
    )


def validate_runs(paths: Paths) -> pd.DataFrame:
    """Check that encoder settings match and only the expected head is removed."""
    rows = []
    backbone_reference = None
    input_reference = None
    for run_key, run in RUNS.items():
        config_path, weights_path, metrics_path = model_paths(paths, run_key)
        for required in (config_path, weights_path, metrics_path):
            if not required.exists():
                raise FileNotFoundError(f"Missing required file: {required}")

        config = yaml.safe_load(config_path.read_text())
        metrics = pd.read_csv(metrics_path)
        backbone = config.get("backbone")
        input_features = config.get("input_features")
        if backbone_reference is None:
            backbone_reference = backbone
            input_reference = input_features
        if backbone != backbone_reference:
            raise ValueError(f"Backbone mismatch in {run_key}")
        if input_features != input_reference:
            raise ValueError(f"Input-feature mismatch in {run_key}")

        observed_heads = set(config.get("out_heads", {}))
        expected_heads = set(TASKS)
        if run["removed"] is not None:
            expected_heads.remove(run["removed"])
        if observed_heads != expected_heads:
            raise ValueError(
                f"Unexpected heads for {run_key}: observed={sorted(observed_heads)}, "
                f"expected={sorted(expected_heads)}"
            )
        if config.get("window_size") != 200 or config.get("backbone", {}).get("max_length") != 200:
            raise ValueError(f"{run_key} is not a matched window-200 model")
        if "epoch" not in metrics or int(metrics["epoch"].max()) != 599:
            print(f"[warn] {run_key} does not contain all 75 epochs by old validation; continuing for 75ep analysis")

        row = {
            "run": run_key,
            "label": run["label"],
            "removed_task": run["removed"] or "none",
            "epochs": int(metrics["epoch"].max()) + 1,
            "checkpoint_bytes": weights_path.stat().st_size,
            "heads": ",".join(sorted(observed_heads)),
        }
        for task in TASKS:
            for kind in ("score", "loss"):
                col = f"{task}_train_{kind}"
                row[f"final_{task}_{kind}"] = float(metrics[col].dropna().iloc[-1]) if col in metrics else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    paths.create_outputs()
    out.to_csv(paths.results / "validated_run_inventory.csv", index=False)
    print(out.to_string(index=False))
    return out


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value or "record"


def task1_json(paths: Paths, split: str) -> Path:
    expected = (
        paths.calms_root
        / "task1_classic_classification"
        / f"calms21_task1_{split}.json"
    )
    if expected.exists():
        return expected
    cache_root = paths.lisbet_root / "lisbet_datasets" / "datasets" / ".cache" / "lisbet"
    cached = sorted(
        cache_root.glob(
            "*-task1_classic_classification.zip.unzip/"
            f"task1_classic_classification/calms21_task1_{split}.json"
        )
    )
    if len(cached) == 1:
        print(f"Using cached CalMS21 task 1 {split} data: {cached[0]}")
        return cached[0]
    if len(cached) > 1:
        raise RuntimeError(
            f"Found multiple cached CalMS21 task 1 {split} files: {cached}. "
            "Remove stale cache copies or materialize the intended dataset path."
        )
    raise FileNotFoundError(
        f"CalMS21 task 1 {split} JSON was not found at:\n{expected}\n"
        "The CalMS21 directory may be an unmaterialized link. Confirm it with "
        "`find -L lisbet_datasets/datasets/CalMS21 -maxdepth 3 -type f`."
    )


def extract_pose_arrays(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """Normalize CalMS21 task 1 arrays to (frames, individuals, keypoints, axes)."""
    # Task 1 JSON stores keypoints as
    # (frames, individuals, coordinates, keypoints).
    positions = np.asarray(record["keypoints"], dtype=np.float32).transpose((0, 1, 3, 2))
    scores = np.asarray(record["scores"], dtype=np.float32)
    if positions.ndim != 4 or positions.shape[1:] != (2, 7, 2):
        raise ValueError(f"Unexpected CalMS21 position shape after conversion: {positions.shape}")
    # CalMS21 task 1 stores confidence as (frames, individuals, keypoints).
    # Some converted variants use (frames, keypoints, individuals), so accept
    # and normalize either representation for the DLC writer.
    if scores.shape == (positions.shape[0], 7, 2):
        scores = scores.transpose((0, 2, 1))
    if scores.shape != positions.shape[:3]:
        raise ValueError(f"Position/score shape mismatch: {positions.shape}, {scores.shape}")
    return positions, scores


def write_dlc_csv(path: Path, positions: np.ndarray, scores: np.ndarray) -> None:
    individuals = ("resident", "intruder")
    keypoints = ("nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail")
    columns = []
    values = []
    for ind_i, individual in enumerate(individuals):
        for kp_i, keypoint in enumerate(keypoints):
            for coord_i, coord in enumerate(("x", "y")):
                columns.append(("calms21", individual, keypoint, coord))
                values.append(positions[:, ind_i, kp_i, coord_i])
            columns.append(("calms21", individual, keypoint, "likelihood"))
            values.append(scores[:, ind_i, kp_i])
    frame = pd.DataFrame(
        np.column_stack(values),
        columns=pd.MultiIndex.from_tuples(
            columns, names=("scorer", "individuals", "bodyparts", "coords")
        ),
    )
    frame.to_csv(path, index=True)


def prepare_calms21(paths: Paths, force: bool = False) -> pd.DataFrame:
    """Create DLC inputs and exact frame-label tables for official task 1 splits."""
    paths.create_outputs()
    manifest_path = paths.prepared / "record_manifest.csv"
    if manifest_path.exists() and not force:
        manifest = pd.read_csv(manifest_path)
        print(f"Using existing prepared dataset: {manifest_path}")
        return manifest

    rows = []
    global_vocab = None
    seen_ids = set()
    for split in ("train", "test"):
        pose_dir = paths.prepared / split / "poses"
        label_dir = paths.prepared / split / "labels"
        pose_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        source_path = task1_json(paths, split)
        print(f"Loading CalMS21 {split} JSON (this may take several minutes): {source_path}")
        # json.load avoids holding an additional full-size text copy of these
        # large (approximately 0.6-1.2 GB) source files in memory.
        with source_path.open("r", encoding="utf-8") as source:
            raw = json.load(source)

        for condition, condition_records in raw.items():
            for original_id, record in condition_records.items():
                stem = safe_name(f"{condition}__{original_id}")
                print(f"Preparing {split} record: {stem}")
                if stem in seen_ids:
                    raise ValueError(f"Duplicate generated record ID: {stem}")
                seen_ids.add(stem)

                positions, scores = extract_pose_arrays(record)
                annotations = np.asarray(record["annotations"], dtype=int)
                if len(annotations) != len(positions):
                    raise ValueError(f"Annotation/pose length mismatch for {stem}")

                vocab_map = record.get("metadata", {}).get("vocab")
                if not vocab_map:
                    raise ValueError(f"Missing behavior vocabulary for {stem}")
                vocab = [name for name, idx in sorted(vocab_map.items(), key=lambda item: item[1])]
                if global_vocab is None:
                    global_vocab = vocab
                if vocab != global_vocab:
                    raise ValueError(f"Behavior vocabulary differs in {stem}: {vocab} != {global_vocab}")
                if annotations.min() < 0 or annotations.max() >= len(vocab):
                    raise ValueError(f"Annotation IDs outside vocabulary for {stem}")

                # LISBET's DLC loader scans sequence subdirectories and accepts
                # filenames matching `tracking*.csv`.
                record_pose_dir = pose_dir / stem
                record_pose_dir.mkdir(parents=True, exist_ok=True)
                pose_path = record_pose_dir / "tracking.csv"
                labels_path = label_dir / f"{stem}.csv"
                if force or not pose_path.exists():
                    write_dlc_csv(pose_path, positions, scores)
                label_df = pd.DataFrame(
                    {
                        "record_id": stem,
                        "frame_idx": np.arange(len(annotations), dtype=int),
                        "label_id": annotations,
                        "label_name": [vocab[i] for i in annotations],
                    }
                )
                label_df.to_csv(labels_path, index=False)
                rows.append(
                    {
                        "split": split,
                        "condition": condition,
                        "original_id": original_id,
                        "record_id": stem,
                        "n_frames": len(annotations),
                        "pose_csv": str(pose_path),
                        "labels_csv": str(labels_path),
                    }
                )

    manifest = pd.DataFrame(rows).sort_values(["split", "record_id"])
    manifest.to_csv(manifest_path, index=False)
    (paths.prepared / "behavior_vocabulary.json").write_text(json.dumps(global_vocab, indent=2))
    print(f"Prepared {len(manifest)} records in {paths.prepared}")
    print(manifest.groupby("split")["n_frames"].agg(["count", "sum"]))
    return manifest


def run_command(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    print("Running:", " ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True)
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}. See {stderr_path}\n"
            f"Last stderr lines:\n{completed.stderr[-3000:]}"
        )


def find_exported_embedder(output_path: Path) -> tuple[Path, Path] | None:
    configs = sorted(output_path.rglob("model_config.yml"))
    if not configs:
        configs = sorted(output_path.rglob("*.yml")) + sorted(output_path.rglob("*.yaml"))
    weights = sorted(output_path.rglob("*.pt")) + sorted(output_path.rglob("*.pth"))
    if len(configs) == 1 and len(weights) == 1:
        return configs[0], weights[0]
    if len(configs) == 0 and len(weights) == 0:
        return None
    raise ValueError(
        f"Expected one exported config and one weight file under {output_path}; "
        f"found configs={configs}, weights={weights}"
    )


def export_embedder(paths: Paths, run_key: str, force: bool = False) -> tuple[Path, Path]:
    """Export the shared trained backbone with the embedding inference head."""
    output_path = paths.embedders / run_key
    output_path.mkdir(parents=True, exist_ok=True)
    existing = find_exported_embedder(output_path)
    if existing is not None and not force:
        print(f"[skip] {run_key}: using exported embedder {existing[0]}")
        return existing

    source_config, source_weights, _ = model_paths(paths, run_key)
    command = [
        sys.executable,
        "-c",
        "from lisbet.cli import main; main()",
        "export_embedder",
        str(source_config),
        str(source_weights),
        "--output_path",
        str(output_path),
    ]
    run_command(
        command,
        paths.logs / f"export_embedder_{run_key}_stdout.txt",
        paths.logs / f"export_embedder_{run_key}_stderr.txt",
    )
    exported = find_exported_embedder(output_path)
    if exported is None:
        raise FileNotFoundError(f"export_embedder produced no model files in {output_path}")
    print(f"Exported {run_key} embedder: {exported[0]}, {exported[1]}")
    return exported


def compute_embeddings(paths: Paths, force: bool = False) -> None:
    """Run betman for each model and official data split."""
    validate_runs(paths)
    manifest = prepare_calms21(paths, force=False)
    paths.create_outputs()
    for run_key in RUNS:
        config_path, weights_path = export_embedder(paths, run_key, force=False)
        for split in ("train", "test"):
            data_path = paths.prepared / split / "poses"
            output_path = paths.embeddings / run_key / split
            output_path.mkdir(parents=True, exist_ok=True)
            expected = list(output_path.rglob("features_lisbet_embedding.csv"))
            expected_records = int((manifest["split"] == split).sum())
            if len(expected) == expected_records and not force:
                print(f"[skip] {run_key}/{split}: found {len(expected)} embedding files")
                continue
            command = [
                sys.executable,
                "-c",
                "from lisbet.cli import main; main()",
                "compute_embeddings",
                str(data_path),
                str(config_path),
                str(weights_path),
                "--data_format",
                "maDLC",
                "--window_size",
                "200",
                "--output_path",
                str(output_path),
            ]
            run_command(
                command,
                paths.logs / f"embedding_{run_key}_{split}_stdout.txt",
                paths.logs / f"embedding_{run_key}_{split}_stderr.txt",
            )


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if str(c).isdigit()]
    if cols:
        return sorted(cols, key=lambda c: int(str(c)))
    excluded = {"frame_idx", "time", "index"}
    cols = [
        c
        for c in frame.columns
        if c not in excluded
        and not str(c).startswith("Unnamed")
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not cols:
        raise ValueError(f"No embedding dimensions found. Columns: {frame.columns.tolist()}")
    return cols


def index_embedding_files(paths: Paths, run_key: str, split: str, record_ids: list[str]) -> dict[str, Path]:
    files = sorted((paths.embeddings / run_key / split).rglob("features_lisbet_embedding.csv"))
    mapping = {}
    for record_id in record_ids:
        matches = [f for f in files if record_id in f.parts or record_id in str(f)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one embedding file for {run_key}/{split}/{record_id}; "
                f"found {len(matches)}. Available examples: {files[:5]}"
            )
        mapping[record_id] = matches[0]
    return mapping


def load_split_embeddings(
    paths: Paths, run_key: str, split: str, manifest: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    subset = manifest[manifest["split"] == split].sort_values("record_id")
    record_ids = subset["record_id"].tolist()
    file_map = index_embedding_files(paths, run_key, split, record_ids)
    x_parts, y_parts, group_parts = [], [], []
    class_names = json.loads((paths.prepared / "behavior_vocabulary.json").read_text())

    for row in subset.itertuples(index=False):
        labels = pd.read_csv(row.labels_csv)
        emb = pd.read_csv(file_map[row.record_id])
        cols = embedding_columns(emb)
        if "frame_idx" in emb.columns:
            merged = labels.merge(emb[["frame_idx"] + cols], on="frame_idx", how="inner", validate="one_to_one")
            if len(merged) != len(labels):
                raise ValueError(
                    f"Frame-index alignment lost rows for {run_key}/{row.record_id}: "
                    f"labels={len(labels)}, merged={len(merged)}"
                )
            x = merged[cols].to_numpy(dtype=np.float32)
            y = merged["label_id"].to_numpy(dtype=int)
        else:
            if len(emb) != len(labels):
                raise ValueError(
                    f"Embedding/label length mismatch for {run_key}/{row.record_id}: "
                    f"embeddings={len(emb)}, labels={len(labels)}. No silent truncation is performed."
                )
            x = emb[cols].to_numpy(dtype=np.float32)
            y = labels["label_id"].to_numpy(dtype=int)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite embeddings in {run_key}/{row.record_id}")
        x_parts.append(x)
        y_parts.append(y)
        group_parts.append(np.repeat(row.record_id, len(y)))

    return (
        np.concatenate(x_parts),
        np.concatenate(y_parts),
        np.concatenate(group_parts),
        class_names,
    )


def balanced_sample(y: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.append(np.sort(idx))
    return np.sort(np.concatenate(selected))


def metrics_from_cm(cm: np.ndarray) -> dict[str, np.ndarray | float]:
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def predict_in_chunks(clf: KNeighborsClassifier, x: np.ndarray, chunk_size: int) -> np.ndarray:
    predictions = []
    for start in range(0, len(x), chunk_size):
        predictions.append(clf.predict(x[start : start + chunk_size]))
    return np.concatenate(predictions)


def tune_k_on_all_task_training(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    k_grid: list[int],
    max_train_per_class: int,
    max_val_per_class: int,
    seed: int,
) -> tuple[int, pd.DataFrame]:
    n_groups = len(np.unique(groups))
    if n_groups < 3:
        raise ValueError("At least three training videos are required for grouped k selection")
    splitter = GroupKFold(n_splits=min(5, n_groups))
    rows = []
    labels = np.arange(len(np.unique(y)))
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups)):
        train_keep = train_idx[balanced_sample(y[train_idx], max_train_per_class, seed + fold)]
        val_keep = val_idx[balanced_sample(y[val_idx], max_val_per_class, seed + 100 + fold)]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_keep]).astype(np.float32)
        x_val = scaler.transform(x[val_keep]).astype(np.float32)
        for k in k_grid:
            clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean", n_jobs=-1)
            clf.fit(x_train, y[train_keep])
            pred = predict_in_chunks(clf, x_val, chunk_size=5000)
            cm = confusion_matrix(y[val_keep], pred, labels=labels)
            metric = metrics_from_cm(cm)
            rows.append({"fold": fold, "k": k, "macro_f1": metric["macro_f1"]})
    results = pd.DataFrame(rows)
    summary = results.groupby("k", as_index=False)["macro_f1"].agg(["mean", "std"]).reset_index()
    best_k = int(summary.sort_values(["mean", "k"], ascending=[False, True]).iloc[0]["k"])
    print("Grouped training-only k selection:")
    print(summary.to_string(index=False))
    print("Selected k:", best_k)
    return best_k, results


def evaluate(paths: Paths, args: argparse.Namespace) -> None:
    """Evaluate frozen representations on the untouched official task 1 test videos."""
    paths.create_outputs()
    manifest = pd.read_csv(paths.prepared / "record_manifest.csv")
    labels = None

    x_all, y_train, groups_train, class_names = load_split_embeddings(
        paths, "all_tasks", "train", manifest
    )
    labels = np.arange(len(class_names))
    best_k, tuning = tune_k_on_all_task_training(
        x_all,
        y_train,
        groups_train,
        args.k_grid,
        args.max_train_per_class,
        args.max_validation_per_class,
        args.seed,
    )
    tuning.to_csv(paths.results / "knn_k_selection_grouped_training.csv", index=False)
    (paths.results / "selected_knn_k.json").write_text(json.dumps({"k": best_k}, indent=2))
    del x_all, y_train, groups_train

    model_cms: dict[str, dict[str, np.ndarray]] = {}
    summary_rows = []
    per_class_rows = []
    cm_rows = []

    for model_i, run_key in enumerate(RUNS):
        print(f"Evaluating {RUNS[run_key]['label']}...")
        x_train, y_train, _, names_train = load_split_embeddings(paths, run_key, "train", manifest)
        if names_train != class_names:
            raise ValueError("Class-name mismatch across models")
        train_keep = balanced_sample(y_train, args.max_train_per_class, args.seed)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train[train_keep]).astype(np.float32)
        clf = KNeighborsClassifier(
            n_neighbors=best_k, weights="distance", metric="euclidean", n_jobs=-1
        )
        clf.fit(x_train_scaled, y_train[train_keep])
        del x_train, x_train_scaled, y_train

        x_test, y_test, groups_test, names_test = load_split_embeddings(paths, run_key, "test", manifest)
        if names_test != class_names:
            raise ValueError("Class-name mismatch across splits")
        model_cms[run_key] = {}
        for record_id in sorted(np.unique(groups_test)):
            idx = np.flatnonzero(groups_test == record_id)
            x_record = scaler.transform(x_test[idx]).astype(np.float32)
            pred = predict_in_chunks(clf, x_record, args.prediction_chunk_size)
            cm = confusion_matrix(y_test[idx], pred, labels=labels)
            model_cms[run_key][record_id] = cm
            for true_i in labels:
                for pred_i in labels:
                    cm_rows.append(
                        {
                            "model": run_key,
                            "record_id": record_id,
                            "true_class": class_names[true_i],
                            "predicted_class": class_names[pred_i],
                            "count": int(cm[true_i, pred_i]),
                        }
                    )
        total_cm = sum(model_cms[run_key].values())
        metric = metrics_from_cm(total_cm)
        summary_rows.append(
            {
                "model": run_key,
                "label": RUNS[run_key]["label"],
                "removed_task": RUNS[run_key]["removed"] or "none",
                "k": best_k,
                "n_train_probe": len(train_keep),
                "n_test_frames": int(total_cm.sum()),
                "accuracy": metric["accuracy"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "macro_f1": metric["macro_f1"],
            }
        )
        for class_i, class_name in enumerate(class_names):
            per_class_rows.append(
                {
                    "model": run_key,
                    "label": RUNS[run_key]["label"],
                    "class_id": class_i,
                    "class_name": class_name,
                    "precision": metric["precision"][class_i],
                    "recall": metric["recall"][class_i],
                    "f1": metric["f1"][class_i],
                    "support": int(metric["support"][class_i]),
                }
            )
        del x_test, y_test, groups_test

    pd.DataFrame(cm_rows).to_csv(paths.results / "test_video_confusion_matrices.csv", index=False)
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(paths.results / "calms21_per_class_metrics.csv", index=False)

    rng = np.random.default_rng(args.seed)
    test_videos = sorted(next(iter(model_cms.values())).keys())
    bootstrap_rows = []
    delta_rows = []
    for bootstrap_i in range(args.bootstrap_replicates):
        draw = rng.choice(test_videos, size=len(test_videos), replace=True)
        boot_metrics = {}
        for run_key in RUNS:
            cm = sum(model_cms[run_key][video] for video in draw)
            boot_metrics[run_key] = metrics_from_cm(cm)
            bootstrap_rows.append(
                {
                    "bootstrap": bootstrap_i,
                    "model": run_key,
                    "macro_f1": boot_metrics[run_key]["macro_f1"],
                    "balanced_accuracy": boot_metrics[run_key]["balanced_accuracy"],
                }
            )
        for run_key, run in RUNS.items():
            if run["removed"] is None:
                continue
            delta_rows.append(
                {
                    "bootstrap": bootstrap_i,
                    "removed_task": run["removed"],
                    "delta_macro_f1_all_minus_without": (
                        boot_metrics["all_tasks"]["macro_f1"] - boot_metrics[run_key]["macro_f1"]
                    ),
                    "delta_per_class_f1_all_minus_without": (
                        boot_metrics["all_tasks"]["f1"] - boot_metrics[run_key]["f1"]
                    ).tolist(),
                }
            )

    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(paths.results / "test_video_bootstrap_metrics.csv", index=False)
    delta_long = []
    for row in delta_rows:
        for class_i, class_name in enumerate(class_names):
            delta_long.append(
                {
                    "bootstrap": row["bootstrap"],
                    "removed_task": row["removed_task"],
                    "class_name": class_name,
                    "delta_f1_all_minus_without": row["delta_per_class_f1_all_minus_without"][class_i],
                }
            )
    pd.DataFrame(delta_long).to_csv(paths.results / "paired_bootstrap_per_class_task_effects.csv", index=False)
    delta_overall = pd.DataFrame(
        [
            {
                "bootstrap": row["bootstrap"],
                "removed_task": row["removed_task"],
                "delta_macro_f1_all_minus_without": row["delta_macro_f1_all_minus_without"],
            }
            for row in delta_rows
        ]
    )
    delta_overall.to_csv(paths.results / "paired_bootstrap_macro_f1_task_effects.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    ci = (
        bootstrap.groupby("model")["macro_f1"]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "macro_f1_ci_low", 0.975: "macro_f1_ci_high"})
        .reset_index()
    )
    summary = summary.merge(ci, on="model", how="left")
    summary.to_csv(paths.results / "calms21_model_summary.csv", index=False)

    contribution_rows = []
    all_pc = per_class[per_class["model"] == "all_tasks"].set_index("class_name")
    for run_key, run in RUNS.items():
        if run["removed"] is None:
            continue
        ablated_pc = per_class[per_class["model"] == run_key].set_index("class_name")
        for class_name in class_names:
            contribution_rows.append(
                {
                    "removed_task": run["removed"],
                    "class_name": class_name,
                    "delta_f1_all_minus_without": (
                        all_pc.loc[class_name, "f1"] - ablated_pc.loc[class_name, "f1"]
                    ),
                }
            )
    pd.DataFrame(contribution_rows).to_csv(paths.results / "per_class_task_contribution.csv", index=False)
    print(summary.to_string(index=False))


def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="normal")


def save_figure(fig: plt.Figure, paths: Paths, stem: str) -> None:
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(paths.figures / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    print("Saved figure:", paths.figures / stem)


def plot_training(paths: Paths) -> plt.Figure:
    """Plot raw and lightly smoothed trajectories; no false replicate bands."""
    set_plot_style()
    histories = {}
    for run_key in RUNS:
        _, _, metrics_path = model_paths(paths, run_key)
        histories[run_key] = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(4, 2, figsize=(8.2, 8.8), sharex=True)
    for task_i, task in enumerate(TASKS):
        for col_i, kind in enumerate(("score", "loss")):
            ax = axes[task_i, col_i]
            metric_col = f"{task}_train_{kind}"
            for run_key, history in histories.items():
                if metric_col not in history:
                    continue
                x = history["epoch"].to_numpy()
                raw = history[metric_col].to_numpy()
                smooth = pd.Series(raw).rolling(15, center=True, min_periods=1).mean().to_numpy()
                ax.plot(x, raw, color=RUN_COLORS[run_key], alpha=0.16, linewidth=0.45)
                ax.plot(x, smooth, color=RUN_COLORS[run_key], linewidth=1.25, label=RUNS[run_key]["label"])
            ax.set_title(task)
            if col_i == 0:
                ax.set_ylabel("Training score")
                ax.set_ylim(0.4, 1.02)
            else:
                ax.set_ylabel("Training loss")
                ax.set_ylim(bottom=0)
            if task_i == len(TASKS) - 1:
                ax.set_xlabel("Epoch")
    panel_label(axes[0, 0], "a")
    panel_label(axes[0, 1], "b")
    handles = [
        mpl.lines.Line2D([0], [0], color=RUN_COLORS[k], lw=1.5, label=RUNS[k]["label"])
        for k in RUNS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.08, hspace=0.42, wspace=0.28)
    save_figure(fig, paths, "Fig_training_curves_75ep_leave_one_task_out")
    return fig


def annotate_heatmap(ax: plt.Axes, values: np.ndarray, fmt: str, threshold: float | None = None) -> None:
    if threshold is None:
        threshold = float(np.nanmedian(values))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=7, color="white" if value < threshold else "black")


def plot_downstream(paths: Paths) -> plt.Figure:
    set_plot_style()
    summary = pd.read_csv(paths.results / "calms21_model_summary.csv")
    per_class = pd.read_csv(paths.results / "calms21_per_class_metrics.csv")
    contribution = pd.read_csv(paths.results / "per_class_task_contribution.csv")
    model_order = list(RUNS)
    model_labels = [RUNS[k]["label"] for k in model_order]
    class_order = per_class.sort_values("class_id")["class_name"].drop_duplicates().tolist()

    absolute = (
        per_class.pivot(index="class_name", columns="model", values="f1")
        .loc[class_order, model_order]
    )
    effect = (
        contribution.pivot(index="class_name", columns="removed_task", values="delta_f1_all_minus_without")
        .loc[class_order, list(TASKS)]
    )

    fig = plt.figure(figsize=(11.2, 3.65))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.35, 1.15, 1.0), wspace=0.48)
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(absolute.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax0.set_xticks(range(len(model_order)), model_labels, rotation=38, ha="right")
    ax0.set_yticks(range(len(class_order)), class_order)
    ax0.set_xlabel("Frozen LISBET encoder")
    ax0.set_ylabel("CalMS21 behavior")
    ax0.set_title("Per-behavior decoding")
    annotate_heatmap(ax0, absolute.values, ".2f", threshold=0.55)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03, label="Test F1")
    panel_label(ax0, "c")

    ax1 = fig.add_subplot(gs[0, 1])
    vmax = max(0.01, float(np.nanmax(np.abs(effect.values))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im1 = ax1.imshow(effect.values, cmap="coolwarm", norm=norm, aspect="auto")
    ax1.set_xticks(range(len(TASKS)), TASKS)
    ax1.set_yticks(range(len(class_order)), class_order)
    ax1.set_xlabel("Removed task")
    ax1.set_ylabel("CalMS21 behavior")
    ax1.set_title("Task-removal effect")
    for i in range(effect.shape[0]):
        for j in range(effect.shape[1]):
            ax1.text(j, i, f"{effect.values[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03, label=r"$F1_{all}-F1_{without}$")
    panel_label(ax1, "d")

    ax2 = fig.add_subplot(gs[0, 2])
    summary = summary.set_index("model").loc[model_order].reset_index()
    x = np.arange(len(summary))
    y = summary["macro_f1"].to_numpy()
    yerr = np.vstack(
        [
            y - summary["macro_f1_ci_low"].to_numpy(),
            summary["macro_f1_ci_high"].to_numpy() - y,
        ]
    )
    ax2.errorbar(x, y, yerr=yerr, fmt="o", color="#202020", ecolor="#555555", capsize=3, linewidth=1.1)
    ax2.set_xticks(x, model_labels, rotation=38, ha="right")
    ax2.set_ylabel("Test macro-F1")
    ax2.set_xlabel("Frozen LISBET encoder")
    ax2.set_ylim(max(0, float(np.nanmin(yerr[0] * -1 + y)) - 0.05), min(1, float(np.nanmax(yerr[1] + y)) + 0.05))
    ax2.set_title("Overall behavioral decoding")
    panel_label(ax2, "e")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.28)
    save_figure(fig, paths, "Fig_calms21_knn_task_ablation_posthoc")
    return fig


def plot_all(paths: Paths) -> None:
    paths.create_outputs()
    training = plot_training(paths)
    downstream = plot_downstream(paths)
    plt.show()
    plt.close(training)
    plt.close(downstream)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("stage", choices=("validate", "prepare", "embed", "evaluate", "plot", "all"))
    parser.add_argument(
        "--lisbet-root",
        type=Path,
        default=Path.home() / "Dokumente" / "Lisbet",
        help="Directory containing the LISBET repository, datasets, and ablation folder",
    )
    parser.add_argument("--force", action="store_true", help="Recompute prepared data or embeddings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-grid", type=int, nargs="+", default=[1, 3, 5, 11, 21])
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=10000,
        help="Balanced cap per class for the kNN reference set; applied identically to every model",
    )
    parser.add_argument("--max-validation-per-class", type=int, default=5000)
    parser.add_argument("--prediction-chunk-size", type=int, default=5000)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths.from_root(args.lisbet_root)
    paths.create_outputs()
    if args.stage in ("validate", "all"):
        validate_runs(paths)
    if args.stage in ("prepare", "all"):
        prepare_calms21(paths, force=args.force)
    if args.stage in ("embed", "all"):
        compute_embeddings(paths, force=args.force)
    if args.stage in ("evaluate", "all"):
        evaluate(paths, args)
    if args.stage in ("plot", "all"):
        plot_all(paths)


if __name__ == "__main__":
    main()
