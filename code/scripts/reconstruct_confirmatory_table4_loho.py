#!/usr/bin/env python3
"""Reconstruct manuscript Table 4 with nested leave-one-HMM-out ridge selection.

The analysis uses the 64-HMM confirmatory cohort and the eight seed-averaged
Shape-RMSE conditions {GRU, Transformer} x {8, 32, 128, 256}.

Models
------
1. Categorical K-only comparator.
2. PLS-1 composite.
3. PLS-2 composite.
4. Full multivariate ridge with an inner LOHO choice of alpha from
   {1, 2, 4, 8, 16, 32} in every outer fold.

All predictor preprocessing, response standardization, model selection, and
model fitting use only the HMMs available in the corresponding training fold.
Ties in the inner ridge selection are resolved in favor of the larger alpha.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = ["GRU", "Transformer"]
WIDTHS = [8, 32, 128, 256]
RIDGE_ALPHAS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
FEATURE_COLUMNS = [
    "log_gain",
    "log_rank",
    "neg_log_sigma",
    "log_tau",
    "bayes_ce_h4",
    "symbol_entropy",
    "log_min_symbol_prob",
    "log_signal",
]
RUN_TABLE_SUFFIX = "data/run_level/confirmatory__confirmatory_training_with_metrics.csv"
FEATURE_TABLE_SUFFIX = "data/hmms/confirmatory/selected_hmm_metrics_full.csv"


def read_csv_by_suffix(artifact_zip: Path, suffix: str) -> pd.DataFrame:
    with zipfile.ZipFile(artifact_zip) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one file ending with {suffix!r}; found {len(matches)}."
            )
        return pd.read_csv(io.BytesIO(archive.read(matches[0])))


def load_inputs(artifact_zip: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if artifact_zip is None:
        return (
            pd.read_csv(ROOT / RUN_TABLE_SUFFIX),
            pd.read_csv(ROOT / FEATURE_TABLE_SUFFIX),
            "extracted artifact",
        )
    if not artifact_zip.exists():
        raise FileNotFoundError(artifact_zip)
    return (
        read_csv_by_suffix(artifact_zip, RUN_TABLE_SUFFIX),
        read_csv_by_suffix(artifact_zip, FEATURE_TABLE_SUFFIX),
        f"artifact ZIP: {artifact_zip.name}",
    )


def build_response_matrix(
    run_table: pd.DataFrame, hmm_ids: list[str]
) -> tuple[np.ndarray, list[str]]:
    required = {"hmm_id", "architecture", "dimension", "seed", "shape_rmse"}
    missing = required.difference(run_table.columns)
    if missing:
        raise ValueError(f"Run table is missing columns: {sorted(missing)}")

    averaged = (
        run_table.groupby(["hmm_id", "architecture", "dimension"], as_index=False)[
            "shape_rmse"
        ].mean()
    )
    pivot = averaged.pivot(
        index="hmm_id", columns=["architecture", "dimension"], values="shape_rmse"
    )
    ordered = [(a, d) for a in ARCHITECTURES for d in WIDTHS]
    missing_conditions = [condition for condition in ordered if condition not in pivot]
    if missing_conditions:
        raise ValueError(f"Missing architecture-width conditions: {missing_conditions}")
    matrix = pivot.loc[hmm_ids, ordered].to_numpy(dtype=float)
    if np.isnan(matrix).any():
        raise ValueError("Response matrix contains missing values")
    return matrix, [f"{a}_d{d}" for a, d in ordered]


def categorical_k_prediction(
    y: np.ndarray, k_values: np.ndarray, train_idx: np.ndarray, test_idx: int
) -> np.ndarray:
    same_k = k_values[train_idx] == k_values[test_idx]
    if not np.any(same_k):
        raise RuntimeError("No training HMM with the held-out value of K")
    return y[train_idx][same_k].mean(axis=0)


def fit_scaled_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_scaler.transform(x_train), y_scaler.transform(y_train))
    pred_scaled = model.predict(x_scaler.transform(x_test))
    pred_raw = y_scaler.inverse_transform(pred_scaled)
    return pred_scaled, pred_raw


def select_ridge_alpha_nested(
    x: np.ndarray,
    y: np.ndarray,
    outer_train_idx: np.ndarray,
    alpha_grid: list[float],
) -> tuple[float, dict[float, float]]:
    """Choose alpha by inner LOHO pooled RMSE in fold-standardized response space."""
    scores: dict[float, float] = {}
    for alpha in alpha_grid:
        squared_errors: list[np.ndarray] = []
        for validation_idx in outer_train_idx:
            inner_train_idx = outer_train_idx[outer_train_idx != validation_idx]
            x_scaler = StandardScaler().fit(x[inner_train_idx])
            y_scaler = StandardScaler().fit(y[inner_train_idx])
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(
                x_scaler.transform(x[inner_train_idx]),
                y_scaler.transform(y[inner_train_idx]),
            )
            pred_scaled = model.predict(x_scaler.transform(x[[validation_idx]]))[0]
            true_scaled = y_scaler.transform(y[[validation_idx]])[0]
            squared_errors.append((pred_scaled - true_scaled) ** 2)
        scores[alpha] = float(np.sqrt(np.mean(np.vstack(squared_errors))))

    minimum = min(scores.values())
    tied = [
        alpha
        for alpha, score in scores.items()
        if np.isclose(score, minimum, rtol=0.0, atol=1e-12)
    ]
    return max(tied), scores


def loho_predictions(
    x: np.ndarray, y: np.ndarray, k_values: np.ndarray
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    n_hmms = y.shape[0]
    predictions = {
        "K-only": np.zeros_like(y),
        "PLS-1 composite": np.zeros_like(y),
        "PLS-2 composite": np.zeros_like(y),
        "Full ridge": np.zeros_like(y),
    }
    selected_rows: list[dict[str, float | int]] = []
    inner_rows: list[dict[str, float | int]] = []

    all_idx = np.arange(n_hmms)
    for test_idx in range(n_hmms):
        train_idx = all_idx[all_idx != test_idx]
        predictions["K-only"][test_idx] = categorical_k_prediction(
            y, k_values, train_idx, test_idx
        )

        x_scaler = StandardScaler().fit(x[train_idx])
        y_scaler = StandardScaler().fit(y[train_idx])
        x_train_scaled = x_scaler.transform(x[train_idx])
        x_test_scaled = x_scaler.transform(x[[test_idx]])
        y_train_scaled = y_scaler.transform(y[train_idx])

        for n_components, name in (
            (1, "PLS-1 composite"),
            (2, "PLS-2 composite"),
        ):
            model = PLSRegression(
                n_components=n_components, scale=False, max_iter=5000, tol=1e-6
            )
            model.fit(x_train_scaled, y_train_scaled)
            predictions[name][test_idx] = y_scaler.inverse_transform(
                model.predict(x_test_scaled)
            )[0]

        selected_alpha, scores = select_ridge_alpha_nested(
            x=x, y=y, outer_train_idx=train_idx, alpha_grid=RIDGE_ALPHAS
        )
        for alpha, score in scores.items():
            inner_rows.append(
                {
                    "outer_test_index": test_idx,
                    "alpha": alpha,
                    "inner_standardized_rmse": score,
                    "selected": alpha == selected_alpha,
                }
            )
        selected_rows.append(
            {"outer_test_index": test_idx, "selected_alpha": selected_alpha}
        )

        ridge = Ridge(alpha=selected_alpha, fit_intercept=True)
        ridge.fit(x_train_scaled, y_train_scaled)
        predictions["Full ridge"][test_idx] = y_scaler.inverse_transform(
            ridge.predict(x_test_scaled)
        )[0]

    return predictions, pd.DataFrame(selected_rows), pd.DataFrame(inner_rows)


def calculate_metrics(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    condition_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_sample_sd = y_true.std(axis=0, ddof=1)
    if np.any(condition_sample_sd <= 0):
        raise ValueError("At least one response condition has zero variance")

    summary_rows: list[dict[str, float | str]] = []
    detail_rows: list[dict[str, float | str]] = []
    for model_name, y_pred in predictions.items():
        residuals = (y_pred - y_true) / condition_sample_sd
        condition_spearman: list[float] = []
        for j, label in enumerate(condition_labels):
            rho = float(spearmanr(y_true[:, j], y_pred[:, j]).statistic)
            condition_spearman.append(rho)
            detail_rows.append(
                {
                    "model": model_name,
                    "condition": label,
                    "sample_sd": condition_sample_sd[j],
                    "condition_standardized_rmse": float(
                        np.sqrt(np.mean(residuals[:, j] ** 2))
                    ),
                    "condition_r2": float(r2_score(y_true[:, j], y_pred[:, j])),
                    "condition_spearman": rho,
                }
            )
        summary_rows.append(
            {
                "model": model_name,
                "standardized_rmse": float(np.sqrt(np.mean(residuals**2))),
                "pooled_r2": float(
                    r2_score(y_true, y_pred, multioutput="variance_weighted")
                ),
                "mean_spearman": float(np.mean(condition_spearman)),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def build_prediction_table(
    hmm_ids: list[str],
    k_values: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    condition_labels: list[str],
    selected_alpha: pd.DataFrame,
) -> pd.DataFrame:
    selected_map = dict(
        zip(selected_alpha["outer_test_index"], selected_alpha["selected_alpha"])
    )
    rows: list[dict[str, object]] = []
    for i, hmm_id in enumerate(hmm_ids):
        for j, condition in enumerate(condition_labels):
            row: dict[str, object] = {
                "hmm_id": hmm_id,
                "K": int(k_values[i]),
                "condition": condition,
                "observed_shape_rmse": y_true[i, j],
                "selected_full_ridge_alpha": selected_map[i],
            }
            for name, values in predictions.items():
                safe = name.lower().replace("-", "_").replace(" ", "_")
                row[f"predicted_{safe}"] = values[i, j]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-zip", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or ROOT / "outputs/reconstructed_table4"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_table, feature_table, input_mode = load_inputs(args.artifact_zip)

    required = {"hmm_id", "K", *FEATURE_COLUMNS}
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")

    hmm_ids = feature_table["hmm_id"].tolist()
    y, condition_labels = build_response_matrix(run_table, hmm_ids)
    indexed = feature_table.set_index("hmm_id")
    x = indexed.loc[hmm_ids, FEATURE_COLUMNS].to_numpy(dtype=float)
    k_values = indexed.loc[hmm_ids, "K"].to_numpy()

    predictions, selected_alpha, inner_scores = loho_predictions(x, y, k_values)
    summary, condition_metrics = calculate_metrics(y, predictions, condition_labels)
    prediction_table = build_prediction_table(
        hmm_ids,
        k_values,
        y,
        predictions,
        condition_labels,
        selected_alpha,
    )
    selected_alpha.insert(0, "hmm_id", hmm_ids)
    selected_counts = (
        selected_alpha.groupby("selected_alpha", as_index=False)
        .size()
        .rename(columns={"size": "outer_fold_count"})
    )

    summary.to_csv(output_dir / "table4_confirmatory_loho_reconstructed.csv", index=False)
    condition_metrics.to_csv(output_dir / "table4_condition_metrics.csv", index=False)
    prediction_table.to_csv(output_dir / "table4_loho_predictions.csv", index=False)
    selected_alpha.to_csv(output_dir / "table4_full_ridge_selected_alphas.csv", index=False)
    selected_counts.to_csv(
        output_dir / "table4_full_ridge_selected_alpha_counts.csv", index=False
    )
    inner_scores.to_csv(output_dir / "table4_full_ridge_inner_scores.csv", index=False)

    metadata = {
        "analysis": "Confirmatory LOHO neural Shape composite (manuscript Table 4)",
        "input_mode": input_mode,
        "n_hmms": len(hmm_ids),
        "conditions": condition_labels,
        "predictor_columns": FEATURE_COLUMNS,
        "ridge_alpha_grid": RIDGE_ALPHAS,
        "ridge_selection": (
            "Nested LOHO within each 63-HMM outer training fold; pooled RMSE in "
            "the response-standardized space of each inner training fold; ties "
            "resolved in favor of the larger alpha."
        ),
        "selected_alpha_counts": {
            str(row.selected_alpha): int(row.outer_fold_count)
            for row in selected_counts.itertuples(index=False)
        },
        "fold_preprocessing": (
            "Predictor standardization, response standardization, ridge penalty "
            "selection, and model fitting use only HMMs available in the corresponding fold."
        ),
        "standardized_rmse": (
            "Pooled RMSE after dividing raw outer-LOHO residuals by the full-confirmatory "
            "condition-wise sample SD (ddof=1)."
        ),
        "pooled_r2": "Variance-weighted multi-output R2 on raw Shape RMSE.",
        "mean_spearman": "Arithmetic mean of eight condition-specific Spearman correlations.",
    }
    (output_dir / "table4_reconstruction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print("\nSelected ridge alpha counts:")
    print(selected_counts.to_string(index=False))


if __name__ == "__main__":
    main()
