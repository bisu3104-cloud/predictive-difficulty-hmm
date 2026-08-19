#!/usr/bin/env python3
"""Reconstruct the dataset-realization robustness tables from public CSV inputs."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

ARCHITECTURES = ("GRU", "Transformer")
OUTCOMES = ("A_width_shape", "mean_excess_ce_H")
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 2026080402


def percentile_ci(values, level=0.95):
    arr = np.asarray(values, dtype=float)
    alpha = 1.0 - level
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def one_way_icc(values):
    P, R = values.shape
    grand = values.mean()
    means = values.mean(axis=1)
    ms_between = R * np.sum((means - grand) ** 2) / (P - 1)
    ms_within = np.sum((values - means[:, None]) ** 2) / (P * (R - 1))
    sigma_process = max((ms_between - ms_within) / R, 0.0)
    sigma_dataset = max(ms_within, 0.0)
    denom = sigma_process + sigma_dataset
    return {
        "ms_between": float(ms_between), "ms_within": float(ms_within),
        "process_variance": float(sigma_process),
        "dataset_realization_variance": float(sigma_dataset),
        "process_icc": float(sigma_process / denom if denom > 0 else np.nan),
    }


def exact_sign_flip(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed = abs(values.mean())
    total = 1 << len(values)
    count = 0
    for mask in range(total):
        signs = np.array([1.0 if (mask >> j) & 1 else -1.0 for j in range(len(values))])
        if abs(np.mean(signs * values)) >= observed - 1e-15:
            count += 1
    return count / total


def build_summaries(run, selected):
    width = run.groupby(["dataset_replicate", "hmm_id", "architecture", "dimension"], as_index=False).agg(
        shape_rmse=("shape_rmse", "mean"), shape_rmse_sd=("shape_rmse", "std"),
        excess_ce_H=("excess_ce_H", "mean"), excess_ce_H_sd=("excess_ce_H", "std"),
        best_val_ce=("best_val_ce", "mean"),
    )
    rows = []
    for (rep, hmm, arch), g in width.groupby(["dataset_replicate", "hmm_id", "architecture"]):
        g = g.sort_values("dimension")
        e = g["shape_rmse"].to_numpy(float)
        d = g["dimension"].to_numpy(int)
        rows.append({"dataset_replicate": int(rep), "hmm_id": hmm, "architecture": arch,
                     "A_width_shape": float(e.mean()), "best_shape_rmse": float(e.min()),
                     "best_dimension": int(d[int(np.argmin(e))]),
                     "mean_excess_ce_H": float(g["excess_ce_H"].mean())})
    hmm = pd.DataFrame(rows).merge(selected[["hmm_id", "K"]], on="hmm_id", how="left")
    return width, hmm


def pairwise_rank(hmm):
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    reps = sorted(hmm.dataset_replicate.unique())
    for arch in ARCHITECTURES:
        for outcome in OUTCOMES:
            pivot = hmm[hmm.architecture == arch].pivot(index="hmm_id", columns="dataset_replicate", values=outcome).dropna()
            for r1, r2 in combinations(reps, 2):
                x, y = pivot[r1].to_numpy(float), pivot[r2].to_numpy(float)
                boot = []
                for _ in range(BOOTSTRAP_REPEATS):
                    idx = rng.integers(0, len(x), size=len(x))
                    val = spearmanr(x[idx], y[idx]).statistic
                    if np.isfinite(val): boot.append(float(val))
                lo, hi = percentile_ci(boot)
                rows.append({"architecture": arch, "outcome": outcome, "replicate_1": int(r1),
                             "replicate_2": int(r2), "n_hmms": len(x),
                             "spearman_rho": float(spearmanr(x, y).statistic),
                             "bootstrap_ci_low": lo, "bootstrap_ci_high": hi})
    return pd.DataFrame(rows)


def icc_table(hmm):
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for arch in ARCHITECTURES:
        for outcome in OUTCOMES:
            pivot = hmm[hmm.architecture == arch].pivot(index="hmm_id", columns="dataset_replicate", values=outcome).dropna()
            arr = pivot.to_numpy(float)
            point = one_way_icc(arr)
            boot = [one_way_icc(arr[rng.integers(0, len(arr), size=len(arr))])["process_icc"] for _ in range(BOOTSTRAP_REPEATS)]
            lo, hi = percentile_ci(boot)
            rows.append({"architecture": arch, "outcome": outcome, "n_hmms": len(arr),
                         "n_dataset_replicates": arr.shape[1], **point, "icc_ci_low": lo, "icc_ci_high": hi})
    return pd.DataFrame(rows)


def pca_tables(width):
    summaries, loadings, scores = [], [], []
    for rep, g in width.groupby("dataset_replicate"):
        g = g.copy(); g["condition"] = g.architecture + "_d" + g.dimension.astype(str)
        mat = g.pivot(index="hmm_id", columns="condition", values="shape_rmse").dropna()
        X = mat.to_numpy(float); sd = X.std(axis=0, ddof=0); sd[sd < 1e-12] = 1.0
        X = (X - X.mean(axis=0)) / sd
        pca = PCA(n_components=2); sc = pca.fit_transform(X); ld = pca.components_.copy()
        if ld[0].mean() < 0: ld[0] *= -1; sc[:, 0] *= -1
        summaries.append({"dataset_replicate": int(rep), "n_hmms": len(mat),
                          "pc1_explained_variance": pca.explained_variance_ratio_[0],
                          "pc1_pc2_explained_variance": pca.explained_variance_ratio_[:2].sum()})
        loadings += [{"dataset_replicate": int(rep), "condition": c, "pc1_loading": float(v)} for c, v in zip(mat.columns, ld[0])]
        scores += [{"hmm_id": h, "dataset_replicate": int(rep), "pc1_score": float(v)} for h, v in zip(mat.index, sc[:, 0])]
    score = pd.DataFrame(scores)
    pivot = score.pivot(index="hmm_id", columns="dataset_replicate", values="pc1_score").dropna()
    pairs = [{"replicate_1": int(a), "replicate_2": int(b), "n_hmms": len(pivot),
              "pc1_score_spearman": float(spearmanr(pivot[a], pivot[b]).statistic)} for a, b in combinations(pivot.columns, 2)]
    return pd.DataFrame(summaries), pd.DataFrame(loadings), pd.DataFrame(pairs)


def frozen_tables(hmm, frozen):
    frozen = frozen[["hmm_id", "architecture", "outcome", "model", "predicted"]].drop_duplicates()
    perf, paired = [], []
    for rep in sorted(hmm.dataset_replicate.unique()):
        obsrep = hmm[hmm.dataset_replicate == rep]
        for arch in ARCHITECTURES:
            for outcome in OUTCOMES:
                obs = obsrep[obsrep.architecture == arch][["hmm_id", outcome]].rename(columns={outcome: "observed"})
                merged = frozen[(frozen.architecture == arch) & (frozen.outcome == outcome)].merge(obs, on="hmm_id")
                for model, g in merged.groupby("model"):
                    perf.append({"dataset_replicate": int(rep), "architecture": arch, "outcome": outcome,
                                 "model": model, "n_hmms": len(g),
                                 "rmse": float(np.sqrt(mean_squared_error(g.observed, g.predicted))),
                                 "r2": float(r2_score(g.observed, g.predicted))})
                k = merged[merged.model == "K_only"].set_index("hmm_id")
                a = merged[merged.model == "augmented_profile"].set_index("hmm_id")
                common = k.index.intersection(a.index); y = k.loc[common, "observed"].to_numpy(float)
                sek = (y - k.loc[common, "predicted"].to_numpy(float)) ** 2
                sea = (y - a.loc[common, "predicted"].to_numpy(float)) ** 2
                rk, ra = np.sqrt(sek.mean()), np.sqrt(sea.mean())
                paired.append({"dataset_replicate": int(rep), "architecture": arch, "outcome": outcome,
                               "n_hmms": len(common), "K_only_rmse": rk, "augmented_rmse": ra,
                               "augmented_rmse_reduction_percent": 100 * (rk - ra) / rk,
                               "mean_squared_error_advantage": float((sek - sea).mean()),
                               "exact_two_sided_sign_flip_p": exact_sign_flip(sek - sea)})
    return pd.DataFrame(perf), pd.DataFrame(paired)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(); root = args.root
    out = args.output or root / "outputs/dataset_realization_reconstruction"
    out.mkdir(parents=True, exist_ok=True)
    run = pd.read_csv(root / "data/run_level/dataset_realization__training_with_metrics.csv")
    selected = pd.read_csv(root / "data/selections/dataset_realization_selected_hmms.csv")
    frozen = pd.read_csv(root / "data/aggregate/confirmatory__confirmatory_holdout_predictions.csv")
    keys = ["dataset_replicate", "hmm_id", "architecture", "dimension", "seed"]
    if len(run) != 1152 or run[keys].duplicated().any():
        raise RuntimeError("Expected 1,152 unique balanced neural runs.")
    width, hmm = build_summaries(run, selected)
    ranks = pairwise_rank(hmm); icc = icc_table(hmm)
    ps, pl, pp = pca_tables(width); fp, fa = frozen_tables(hmm, frozen)
    outputs = {
        "width_level_summary.csv": width, "hmm_dataset_summary.csv": hmm,
        "dataset_pair_rank_stability.csv": ranks, "hmm_level_process_icc.csv": icc,
        "pca_summary_by_dataset.csv": ps, "pca_pc1_loadings_by_dataset.csv": pl,
        "pca_pc1_score_stability.csv": pp, "frozen_profile_performance_by_dataset.csv": fp,
        "frozen_augmented_vs_K_by_dataset.csv": fa,
    }
    for name, df in outputs.items(): df.to_csv(out / name, index=False)
    rr = ranks.groupby(["architecture", "outcome"])["spearman_rho"].agg(["min", "max"]).reset_index()
    t24 = icc.merge(rr, on=["architecture", "outcome"])[["architecture", "outcome", "process_icc", "icc_ci_low", "icc_ci_high", "min", "max"]]
    t24.columns = ["architecture", "outcome", "process_icc", "icc_ci_low", "icc_ci_high", "dataset_pair_spearman_min", "dataset_pair_spearman_max"]
    t24.to_csv(out / "table24_dataset_realization_stability.csv", index=False)
    fa[fa.outcome == "A_width_shape"][["architecture", "dataset_replicate", "K_only_rmse", "augmented_rmse", "augmented_rmse_reduction_percent"]].sort_values(["architecture", "dataset_replicate"]).to_csv(out / "table25_dataset_realization_frozen_prediction.csv", index=False)
    print(json.dumps({"status": "ok", "run_rows": len(run), "selected_hmms": selected.hmm_id.nunique(), "output": str(out)}, indent=2))

if __name__ == "__main__":
    main()
