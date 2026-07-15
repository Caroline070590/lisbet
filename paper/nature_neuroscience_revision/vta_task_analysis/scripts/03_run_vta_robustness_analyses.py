#!/usr/bin/env python3
"""
Fast residual and robustness analysis for VTA/LISBET Figure 6 post-hoc models.

This version uses within-unit demeaning instead of explicitly fitting C(unit_id)
with thousands of fixed-effect dummy coefficients. It is designed to be much
faster for bootstrap/permutation sensitivity analyses.

It tests:
  Model 1: VTA neural activity ~ velocity + distance + unit fixed effect
  Model 2: VTA neural activity ~ velocity + distance + LISBET prototype + unit fixed effect

Then:
  Residual analysis: residual neural activity after velocity + distance + unit ~ LISBET prototype + unit

Robustness checks:
  - Extended controls: velocity + distance + angle + duration
  - Trial-cluster bootstrap confidence intervals
  - Within-trial prototype-label permutation test
  - Leave-one-trial-out sensitivity
  - LISBET mplstyle plots, if lisbet.mplstyle is available
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")


def zscore_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
    def _z(x: pd.Series) -> pd.Series:
        sd = x.std(ddof=0)
        if x.notna().sum() < 2 or pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=x.index)
        return (x - x.mean()) / sd
    return df.groupby(group_col, observed=True)[value_col].transform(_z)


def demean_by_group_array(values: np.ndarray, groups: pd.Series) -> np.ndarray:
    df = pd.DataFrame(values)
    return (df - df.groupby(groups.to_numpy()).transform("mean")).to_numpy(dtype=float)


def demean_series_by_group(y: pd.Series, groups: pd.Series) -> np.ndarray:
    return (y - y.groupby(groups).transform("mean")).to_numpy(dtype=float)


def ols_metrics(y: np.ndarray, X: np.ndarray) -> dict:
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[good]
    X = X[good]
    n = len(y)
    if n == 0:
        raise ValueError("Empty y")

    # Drop near-zero variance columns after demeaning.
    col_sd = X.std(axis=0)
    X = X[:, col_sd > 1e-12]
    if X.shape[1] == 0:
        resid = y.copy()
        sse = float(np.sum(resid ** 2))
        p = 0
        yhat = np.zeros_like(y)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        resid = y - yhat
        sse = float(np.sum(resid ** 2))
        p = int(np.linalg.matrix_rank(X))

    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sse / sst if sst > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1) if np.isfinite(r2) else np.nan
    sigma2 = max(sse / n, 1e-300)
    aic = n * np.log(sigma2) + 2 * p
    bic = n * np.log(sigma2) + np.log(n) * p
    return {"n": n, "p": p, "sse": sse, "r2": r2, "adj_r2": adj_r2, "aic": aic, "bic": bic, "resid": resid, "yhat": yhat}


def nested_f_test(restricted: dict, full: dict) -> dict:
    df_num = full["p"] - restricted["p"]
    df_den = full["n"] - full["p"]
    if df_num <= 0 or df_den <= 0:
        return {"f_stat": np.nan, "df_num": df_num, "df_den": df_den, "p_value": np.nan}
    num = (restricted["sse"] - full["sse"]) / df_num
    den = full["sse"] / df_den
    f_stat = num / den if den > 0 else np.nan
    p_value = float(stats.f.sf(f_stat, df_num, df_den)) if np.isfinite(f_stat) else np.nan
    return {"f_stat": float(f_stat), "df_num": int(df_num), "df_den": int(df_den), "p_value": p_value}


def make_design(df: pd.DataFrame, controls: list[str], include_prototype: bool, prototype_col: str = "prototype_id") -> tuple[np.ndarray, list[str]]:
    X_parts = []
    names = []

    for c in controls:
        X_parts.append(df[[c]].to_numpy(dtype=float))
        names.append(c)

    if include_prototype:
        proto = pd.get_dummies(df[prototype_col].astype(int), prefix="prototype", drop_first=True, dtype=float)
        if proto.shape[1] > 0:
            X_parts.append(proto.to_numpy(dtype=float))
            names.extend(list(proto.columns))

    X = np.concatenate(X_parts, axis=1) if X_parts else np.empty((len(df), 0))
    X_dm = demean_by_group_array(X, df["unit_id"].astype(str))
    return X_dm, names


def compare_models_fast(df: pd.DataFrame, controls: list[str], prototype_col: str = "prototype_id") -> dict:
    y_dm = demean_series_by_group(df["neural_z"], df["unit_id"].astype(str))
    X1, names1 = make_design(df, controls=controls, include_prototype=False, prototype_col=prototype_col)
    X2, names2 = make_design(df, controls=controls, include_prototype=True, prototype_col=prototype_col)
    m1 = ols_metrics(y_dm, X1)
    m2 = ols_metrics(y_dm, X2)
    ftest = nested_f_test(m1, m2)
    return {
        "controls": "+".join(controls),
        "n": m2["n"],
        "p_m1": m1["p"],
        "p_m2": m2["p"],
        "r2_m1": m1["r2"],
        "r2_m2": m2["r2"],
        "delta_r2": m2["r2"] - m1["r2"],
        "adj_r2_m1": m1["adj_r2"],
        "adj_r2_m2": m2["adj_r2"],
        "delta_adj_r2": m2["adj_r2"] - m1["adj_r2"],
        "aic_m1": m1["aic"],
        "aic_m2": m2["aic"],
        "delta_aic_m1_minus_m2": m1["aic"] - m2["aic"],
        "bic_m1": m1["bic"],
        "bic_m2": m2["bic"],
        "delta_bic_m1_minus_m2": m1["bic"] - m2["bic"],
        "f_stat": ftest["f_stat"],
        "f_df_num": ftest["df_num"],
        "f_df_den": ftest["df_den"],
        "f_p": ftest["p_value"],
        "m1_resid": m1["resid"],
    }


def residual_analysis_fast(df: pd.DataFrame, controls: list[str]) -> tuple[dict, pd.DataFrame]:
    main = compare_models_fast(df, controls=controls)
    tmp = df.copy()

    # The residual corresponds to neural_z after removing controls and unit fixed effects
    # in the within-unit space.
    tmp["neural_residual_after_controls"] = main["m1_resid"]

    # Null residual model: residual around zero, after unit demeaning already performed.
    y = tmp["neural_residual_after_controls"].to_numpy(dtype=float)
    X_null = np.empty((len(tmp), 0))
    null = ols_metrics(y, X_null)

    X_proto, _ = make_design(tmp.rename(columns={"neural_residual_after_controls": "neural_z"}), controls=[], include_prototype=True)
    full = ols_metrics(y, X_proto)
    ftest = nested_f_test(null, full)

    out = {
        "controls_removed": "+".join(controls),
        "n": full["n"],
        "p_null": null["p"],
        "p_model": full["p"],
        "residual_null_r2": null["r2"],
        "residual_model_r2": full["r2"],
        "delta_residual_r2": full["r2"] - null["r2"],
        "residual_null_aic": null["aic"],
        "residual_model_aic": full["aic"],
        "delta_aic_null_minus_model": null["aic"] - full["aic"],
        "residual_null_bic": null["bic"],
        "residual_model_bic": full["bic"],
        "delta_bic_null_minus_model": null["bic"] - full["bic"],
        "f_stat": ftest["f_stat"],
        "f_df_num": ftest["df_num"],
        "f_df_den": ftest["df_den"],
        "f_p": ftest["p_value"],
    }
    return out, tmp


def resample_trials(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    trials = np.array(sorted(df["trial_id"].astype(str).unique()))
    sampled = rng.choice(trials, size=len(trials), replace=True)
    parts = []
    for i, t in enumerate(sampled):
        part = df[df["trial_id"].astype(str) == t].copy()
        part["trial_id"] = part["trial_id"].astype(str) + f"__boot{i}"
        part["unit_id"] = part["unit_id"].astype(str) + f"__boot{i}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def bootstrap_model(df: pd.DataFrame, controls: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        try:
            boot = resample_trials(df, rng)
            res = compare_models_fast(boot, controls=controls)
            rows.append({"bootstrap_id": b, "status": "ok", **{k: v for k, v in res.items() if k != "m1_resid"}})
        except Exception as e:
            rows.append({"bootstrap_id": b, "status": f"error: {e}"})
        if (b + 1) % max(1, n_boot // 10) == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
    return pd.DataFrame(rows)


def permutation_residual(df_resid: pd.DataFrame, observed_delta: float, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = df_resid.copy()
    y = base["neural_residual_after_controls"].to_numpy(dtype=float)

    for i in range(n_perm):
        tmp = base.copy()
        permuted = []
        for _, g in tmp.groupby("trial_id", observed=True, sort=False):
            vals = g["prototype_id"].to_numpy().copy()
            rng.shuffle(vals)
            permuted.append(pd.Series(vals, index=g.index))
        tmp["prototype_perm"] = pd.concat(permuted).sort_index()
        X_proto, _ = make_design(tmp, controls=[], include_prototype=True, prototype_col="prototype_perm")
        null = ols_metrics(y, np.empty((len(tmp), 0)))
        full = ols_metrics(y, X_proto)
        delta = full["r2"] - null["r2"]
        rows.append({"permutation_id": i, "delta_residual_r2_perm": delta})
        if (i + 1) % max(1, n_perm // 10) == 0:
            print(f"  permutation {i + 1}/{n_perm}")

    out = pd.DataFrame(rows)
    pval = (1 + (out["delta_residual_r2_perm"] >= observed_delta).sum()) / (1 + len(out))
    out["observed_delta_residual_r2"] = observed_delta
    out["permutation_p_value"] = pval
    return out


def leave_one_trial_out(df: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    rows = []
    trials = sorted(df["trial_id"].astype(str).unique())
    for idx, t in enumerate(trials, start=1):
        try:
            sub = df[df["trial_id"].astype(str) != t].copy()
            res = compare_models_fast(sub, controls=controls)
            rows.append({"left_out_trial": t, "status": "ok", **{k: v for k, v in res.items() if k != "m1_resid"}})
        except Exception as e:
            rows.append({"left_out_trial": t, "status": f"error: {e}"})
        if idx % 20 == 0:
            print(f"  leave-one-trial-out {idx}/{len(trials)}")
    return pd.DataFrame(rows)


def summarize_distribution(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
    for col in cols:
        x = pd.to_numeric(ok[col], errors="coerce").dropna()
        rows.append({
            "metric": col,
            "n": int(len(x)),
            "mean": x.mean() if len(x) else np.nan,
            "median": x.median() if len(x) else np.nan,
            "ci95_low": np.percentile(x, 2.5) if len(x) else np.nan,
            "ci95_high": np.percentile(x, 97.5) if len(x) else np.nan,
            "fraction_positive": (x > 0).mean() if len(x) else np.nan,
        })
    return pd.DataFrame(rows)


def prototype_residual_ci(df_resid: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observed = (
        df_resid.groupby("prototype_id", observed=True)
        .agg(
            n=("neural_residual_after_controls", "size"),
            mean_residual=("neural_residual_after_controls", "mean"),
            median_residual=("neural_residual_after_controls", "median"),
        )
        .reset_index()
    )
    protos = sorted(df_resid["prototype_id"].astype(int).unique())
    trials = np.array(sorted(df_resid["trial_id"].astype(str).unique()))
    boot_means = {p: [] for p in protos}

    for _ in range(n_boot):
        sampled = rng.choice(trials, size=len(trials), replace=True)
        boot = pd.concat([df_resid[df_resid["trial_id"].astype(str) == t] for t in sampled], ignore_index=True)
        means = boot.groupby("prototype_id", observed=True)["neural_residual_after_controls"].mean()
        for p in protos:
            boot_means[p].append(means.loc[p] if p in means.index else np.nan)

    rows = []
    for p in protos:
        vals = np.array(boot_means[p], dtype=float)
        vals = vals[np.isfinite(vals)]
        obs = observed[observed["prototype_id"].astype(int) == p].iloc[0]
        rows.append({
            "prototype_id": p,
            "n": int(obs["n"]),
            "mean_residual": float(obs["mean_residual"]),
            "median_residual": float(obs["median_residual"]),
            "ci95_low": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
            "ci95_high": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
        })
    return pd.DataFrame(rows)


def find_style(style_arg: str | None, outdir: Path) -> Path | None:
    candidates = []
    if style_arg:
        candidates.append(Path(style_arg).expanduser())
    candidates += [
        outdir / "lisbet.mplstyle",
        Path.cwd() / "lisbet.mplstyle",
        Path.home() / "Dokumente" / "Lisbet" / "lisbet.mplstyle",
        Path.home() / "Dokumente" / "Lisbet" / "vta_posthoc_results" / "lisbet.mplstyle",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def plot_hist(values: pd.Series, observed: float, xlabel: str, title: str, outpath: Path) -> None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    plt.figure(figsize=(3.2, 2.4))
    plt.hist(values, bins=30)
    plt.axvline(observed, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close()


def plot_loo(loo: pd.DataFrame, observed: float, outpath: Path) -> None:
    ok = loo[loo["status"] == "ok"].copy().sort_values("delta_r2")
    plt.figure(figsize=(3.2, 2.4))
    plt.plot(np.arange(len(ok)), ok["delta_r2"], marker="o", linewidth=0.5)
    plt.axhline(observed, linestyle="--", linewidth=1)
    plt.axhline(0, linestyle=":", linewidth=1)
    plt.xlabel("Leave-one-trial-out run")
    plt.ylabel("ΔR²")
    plt.title("Leave-one-trial-out sensitivity")
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close()


def plot_proto_ci(proto_ci: pd.DataFrame, outpath: Path) -> None:
    d = proto_ci.sort_values("prototype_id")
    x = np.arange(len(d))
    y = d["mean_residual"].to_numpy()
    yerr_low = np.maximum(0, y - d["ci95_low"].to_numpy())
    yerr_high = np.maximum(0, d["ci95_high"].to_numpy() - y)
    yerr = np.vstack([yerr_low, yerr_high])
    plt.figure(figsize=(3.2, 2.4))
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=2, linewidth=0.8)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, d["prototype_id"].astype(str))
    plt.xlabel("LISBET/HMM prototype ID")
    plt.ylabel("Residual neural activity")
    plt.title("Residual mean by prototype")
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-csv", default=str(Path.home() / "Dokumente" / "Lisbet" / "vta_posthoc_results" / "figure6_all_trials_event_neuron_long_valid.csv"))
    parser.add_argument("--outdir", default=str(Path.home() / "Dokumente" / "Lisbet" / "vta_posthoc_results" / "robust_residual_analysis_fast"))
    parser.add_argument("--style", default=None)
    parser.add_argument("--n-boot", type=int, default=int(os.environ.get("N_BOOT", 500)))
    parser.add_argument("--n-perm", type=int, default=int(os.environ.get("N_PERM", 1000)))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    long_csv = Path(args.long_csv).expanduser()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    style = find_style(args.style, outdir)
    if style:
        print("Using style:", style)
        plt.style.use(style)
    else:
        print("No lisbet.mplstyle found; using default matplotlib style.")

    print("Loading:", long_csv)
    df = pd.read_csv(long_csv)

    if "duration_z" not in df.columns:
        df["duration_z"] = zscore_by_group(df, "duration_s", "trial_id")
    if "angle_z" not in df.columns:
        df["angle_z"] = zscore_by_group(df, "angle_mean", "trial_id")

    needed = ["neural_z", "velocity_z", "distance_z", "angle_z", "duration_z", "prototype_id", "unit_id", "trial_id"]
    df = df.dropna(subset=needed).copy()
    df["prototype_id"] = df["prototype_id"].astype(int)
    df["unit_id"] = df["unit_id"].astype(str)
    df["trial_id"] = df["trial_id"].astype(str)

    print(f"Rows: {len(df)} | Trials: {df['trial_id'].nunique()} | Units: {df['unit_id'].nunique()} | Prototypes: {sorted(df['prototype_id'].unique())}")

    main_controls = ["velocity_z", "distance_z"]
    extended_controls = ["velocity_z", "distance_z", "angle_z", "duration_z"]

    print("\nMain model comparison...")
    main_res = compare_models_fast(df, main_controls)
    ext_res = compare_models_fast(df, extended_controls)
    main_summary = pd.DataFrame([
        {"analysis": "main_velocity_distance_vs_lisbet", **{k: v for k, v in main_res.items() if k != "m1_resid"}},
        {"analysis": "extended_velocity_distance_angle_duration_vs_lisbet", **{k: v for k, v in ext_res.items() if k != "m1_resid"}},
    ])
    main_summary.to_csv(outdir / "main_and_extended_model_comparison_fast.csv", index=False)
    print(main_summary[["analysis", "delta_r2", "delta_aic_m1_minus_m2", "delta_bic_m1_minus_m2", "f_p"]])

    print("\nResidual analysis...")
    resid_main, df_resid = residual_analysis_fast(df, main_controls)
    resid_ext, df_resid_ext = residual_analysis_fast(df, extended_controls)
    residual_summary = pd.DataFrame([
        {"analysis": "residual_after_velocity_distance", **resid_main},
        {"analysis": "residual_after_velocity_distance_angle_duration", **resid_ext},
    ])
    residual_summary.to_csv(outdir / "residual_model_summary_fast.csv", index=False)
    df_resid.to_csv(outdir / "long_table_with_velocity_distance_residuals.csv", index=False)
    print(residual_summary[["analysis", "delta_residual_r2", "delta_aic_null_minus_model", "delta_bic_null_minus_model", "f_p"]])

    print(f"\nTrial-cluster bootstrap, n={args.n_boot}...")
    boot = bootstrap_model(df, main_controls, args.n_boot, args.seed)
    boot.to_csv(outdir / "trial_cluster_bootstrap_model_comparison_fast.csv", index=False)
    boot_summary = summarize_distribution(boot, ["delta_r2", "delta_adj_r2", "delta_aic_m1_minus_m2", "delta_bic_m1_minus_m2", "f_p"])
    boot_summary.to_csv(outdir / "trial_cluster_bootstrap_summary_fast.csv", index=False)

    print(f"\nWithin-trial residual permutation, n={args.n_perm}...")
    perm = permutation_residual(df_resid, resid_main["delta_residual_r2"], args.n_perm, args.seed + 1)
    perm.to_csv(outdir / "within_trial_permutation_residual_delta_r2_fast.csv", index=False)

    print("\nLeave-one-trial-out sensitivity...")
    loo = leave_one_trial_out(df, main_controls)
    loo.to_csv(outdir / "leave_one_trial_out_model_comparison_fast.csv", index=False)
    loo_summary = summarize_distribution(loo, ["delta_r2", "delta_adj_r2", "delta_aic_m1_minus_m2", "delta_bic_m1_minus_m2", "f_p"])
    loo_summary.to_csv(outdir / "leave_one_trial_out_summary_fast.csv", index=False)

    print("\nPrototype residual CIs...")
    proto_ci = prototype_residual_ci(df_resid, args.n_boot, args.seed + 2)
    proto_ci.to_csv(outdir / "residual_by_prototype_trial_bootstrap_ci_fast.csv", index=False)

    print("\nSaving plots...")
    plot_hist(boot[boot["status"] == "ok"]["delta_r2"], main_res["delta_r2"], "Bootstrap ΔR²", "Trial-bootstrap ΔR²", outdir / "trial_bootstrap_delta_r2_fast.png")
    plot_hist(perm["delta_residual_r2_perm"], resid_main["delta_residual_r2"], "Null Δ residual R²", "Within-trial permutation", outdir / "within_trial_permutation_residual_delta_r2_fast.png")
    plot_loo(loo, main_res["delta_r2"], outdir / "leave_one_trial_out_delta_r2_fast.png")
    plot_proto_ci(proto_ci, outdir / "residual_mean_by_prototype_lisbet_style_fast.png")

    report = outdir / "robust_residual_analysis_summary_fast.txt"
    perm_p = perm["permutation_p_value"].iloc[0]
    with open(report, "w") as f:
        f.write("Fast VTA/LISBET residual and robustness analysis\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Input: {long_csv}\n")
        f.write(f"Rows: {len(df)}\nTrials: {df['trial_id'].nunique()}\nUnits: {df['unit_id'].nunique()}\n")
        f.write(f"Prototypes: {sorted(df['prototype_id'].unique())}\n")
        f.write(f"Bootstrap iterations: {args.n_boot}\nPermutation iterations: {args.n_perm}\n\n")
        f.write("Main and extended model comparison\n")
        f.write("-" * 72 + "\n")
        f.write(main_summary.to_string(index=False))
        f.write("\n\nResidual analyses\n")
        f.write("-" * 72 + "\n")
        f.write(residual_summary.to_string(index=False))
        f.write("\n\nTrial-cluster bootstrap summary\n")
        f.write("-" * 72 + "\n")
        f.write(boot_summary.to_string(index=False))
        f.write("\n\nLeave-one-trial-out summary\n")
        f.write("-" * 72 + "\n")
        f.write(loo_summary.to_string(index=False))
        f.write("\n\nWithin-trial residual permutation\n")
        f.write("-" * 72 + "\n")
        f.write(f"Observed delta residual R2: {resid_main['delta_residual_r2']:.8g}\n")
        f.write(f"Permutation p-value: {perm_p:.8g}\n")
        f.write("\nResidual mean by prototype with trial-bootstrap CIs\n")
        f.write("-" * 72 + "\n")
        f.write(proto_ci.to_string(index=False))
        f.write("\n")

    print("Saved report:", report)
    print("Saved output folder:", outdir)
    print("DONE")


if __name__ == "__main__":
    main()
