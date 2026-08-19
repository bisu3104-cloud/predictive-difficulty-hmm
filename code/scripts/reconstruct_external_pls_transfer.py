#!/usr/bin/env python3
"""
Reconstruct the external PLS composite-transfer result reported in Section 5.6.

Training response:
    eight confirmatory Shape-RMSE conditions
    {GRU, Transformer} x {8, 32, 128, 256}

External evaluation:
    six architecture-width conditions shared by both cohorts
    {GRU, Transformer} x {8, 32, 128}

The script reads only files contained in this public artifact and writes:
    outputs/paper_tables/external_pls_composite_transfer.csv
    outputs/analysis_metadata/external_pls_composite_transfer_metadata.json

Run:
    python code/scripts/reconstruct_external_pls_transfer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

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

ARCHITECTURES = ["GRU", "Transformer"]
CONFIRMATORY_WIDTHS = [8, 32, 128, 256]
EXTERNAL_WIDTHS = [8, 32, 128]


def build_shape_matrix(
    run_table: pd.DataFrame,
    hmm_ids: list[str],
    widths: list[int],
) -> tuple[np.ndarray, list[str]]:
    required = {"hmm_id", "architecture", "dimension", "shape_rmse"}
    missing = required.difference(run_table.columns)
    if missing:
        raise ValueError(f"Run table is missing columns: {sorted(missing)}")

    aggregated = (
        run_table.groupby(
            ["hmm_id", "architecture", "dimension"],
            as_index=False,
        )["shape_rmse"]
        .mean()
    )
    pivot = aggregated.pivot(
        index="hmm_id",
        columns=["architecture", "dimension"],
        values="shape_rmse",
    )

    ordered_columns = [
        (architecture, width)
        for architecture in ARCHITECTURES
        for width in widths
    ]
    missing_conditions = [
        column for column in ordered_columns if column not in pivot.columns
    ]
    if missing_conditions:
        raise ValueError(
            f"Missing architecture-width conditions: {missing_conditions}"
        )

    matrix = pivot.loc[hmm_ids, ordered_columns].to_numpy(dtype=float)
    labels = [
        f"{architecture}_d{width}"
        for architecture, width in ordered_columns
    ]
    return matrix, labels


def main() -> None:
    confirmatory_runs = pd.read_csv(
        ROOT
        / "data/run_level/"
        "confirmatory__confirmatory_training_with_metrics.csv"
    )
    external_runs = pd.read_csv(
        ROOT
        / "data/run_level/"
        "external__external_training_with_metrics.csv"
    )
    confirmatory_features = pd.read_csv(
        ROOT
        / "data/hmms/confirmatory/"
        "selected_hmm_metrics_full.csv"
    )
    external_features = pd.read_csv(
        ROOT
        / "data/hmms/external/"
        "external_selected_hmm_metrics_full.csv"
    )

    confirmatory_ids = confirmatory_features["hmm_id"].tolist()
    external_ids = external_features["hmm_id"].tolist()

    y_confirmatory, confirmatory_labels = build_shape_matrix(
        confirmatory_runs,
        confirmatory_ids,
        CONFIRMATORY_WIDTHS,
    )
    y_external, external_labels = build_shape_matrix(
        external_runs,
        external_ids,
        EXTERNAL_WIDTHS,
    )

    shared_indices = [
        confirmatory_labels.index(label)
        for label in external_labels
    ]

    x_confirmatory = (
        confirmatory_features.set_index("hmm_id")
        .loc[confirmatory_ids, FEATURE_COLUMNS]
        .to_numpy(dtype=float)
    )
    x_external = (
        external_features.set_index("hmm_id")
        .loc[external_ids, FEATURE_COLUMNS]
        .to_numpy(dtype=float)
    )

    x_scaler = StandardScaler().fit(x_confirmatory)
    y_scaler = StandardScaler().fit(y_confirmatory)

    x_confirmatory_std = x_scaler.transform(x_confirmatory)
    x_external_std = x_scaler.transform(x_external)
    y_confirmatory_std = y_scaler.transform(y_confirmatory)

    predictions: dict[str, np.ndarray] = {}

    k_confirmatory = (
        confirmatory_features.set_index("hmm_id")
        .loc[confirmatory_ids, "K"]
        .to_numpy()
    )
    k_external = (
        external_features.set_index("hmm_id")
        .loc[external_ids, "K"]
        .to_numpy()
    )
    k_prediction_8 = np.empty(
        (len(external_ids), len(confirmatory_labels))
    )
    for k_value in sorted(np.unique(k_confirmatory)):
        k_prediction_8[k_external == k_value] = (
            y_confirmatory[k_confirmatory == k_value].mean(axis=0)
        )
    predictions["K-only categorical"] = (
        k_prediction_8[:, shared_indices]
    )

    for n_components in (1, 2):
        model = PLSRegression(
            n_components=n_components,
            scale=False,
            max_iter=5000,
        )
        model.fit(x_confirmatory_std, y_confirmatory_std)
        prediction_std_8 = model.predict(x_external_std)
        prediction_8 = y_scaler.inverse_transform(prediction_std_8)
        predictions[f"PLS-{n_components}"] = (
            prediction_8[:, shared_indices]
        )

    rows = []
    baseline_rmse = None
    for model_name, prediction in predictions.items():
        rmse = float(
            np.sqrt(np.mean((prediction - y_external) ** 2))
        )
        if model_name == "K-only categorical":
            baseline_rmse = rmse
        rows.append(
            {
                "model": model_name,
                "external_rmse": rmse,
            }
        )

    if baseline_rmse is None:
        raise RuntimeError("K-only baseline was not computed.")

    for row in rows:
        row["improvement_vs_K_only_percent"] = (
            100.0
            * (baseline_rmse - row["external_rmse"])
            / baseline_rmse
        )

    performance = pd.DataFrame(rows)

    table_dir = ROOT / "outputs/paper_tables"
    metadata_dir = ROOT / "outputs/analysis_metadata"
    table_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    table_path = (
        table_dir / "external_pls_composite_transfer.csv"
    )
    metadata_path = (
        metadata_dir
        / "external_pls_composite_transfer_metadata.json"
    )

    performance.to_csv(table_path, index=False)

    metadata = {
        "analysis": "external PLS composite transfer",
        "confirmatory_training_conditions": confirmatory_labels,
        "external_evaluation_conditions": external_labels,
        "n_confirmatory_hmms": len(confirmatory_ids),
        "n_external_hmms": len(external_ids),
        "feature_columns": FEATURE_COLUMNS,
        "PLS_training_response": (
            "All eight confirmatory architecture-width Shape-RMSE "
            "conditions."
        ),
        "external_scoring_response": (
            "The six architecture-width Shape-RMSE conditions shared "
            "by the confirmatory and external cohorts."
        ),
        "K_only_comparator": (
            "Condition-wise confirmatory mean within categorical K."
        ),
        "standardization": (
            "Predictors and the eight-condition response matrix were "
            "standardized using the full confirmatory cohort."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(performance.to_string(index=False))
    print(f"\nWrote: {table_path.relative_to(ROOT)}")
    print(f"Wrote: {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
