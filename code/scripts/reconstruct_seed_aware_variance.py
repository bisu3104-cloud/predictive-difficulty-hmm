#!/usr/bin/env python3
"""Reconstruct the confirmatory seed-aware variance-component analysis.

The script reads the 2,560 confirmatory run-level rows, standardizes each
architecture-by-width condition once on the full confirmatory cohort, estimates
balanced ANOVA variance components, and bootstraps HMMs as clusters.

Default extracted-artifact usage:
    python code/scripts/reconstruct_seed_aware_variance.py

ZIP usage:
    python code/scripts/reconstruct_seed_aware_variance.py \
        --artifact-zip predictive_difficulty_public_artifact_v0_4.zip \
        --output-dir seed_aware_variance_outputs
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_LEVEL_SUFFIX = "data/run_level/confirmatory__confirmatory_training_with_metrics.csv"
OUTCOMES = {
    "Shape RMSE": "shape_rmse",
    "Terminal excess CE": "excess_ce_H",
}


@dataclass
class Estimate:
    n_hmms: int
    n_conditions: int
    n_seeds: int
    process_ss: float
    condition_ss: float
    interaction_ss: float
    residual_ss: float
    process_ms: float
    condition_ms: float
    interaction_ms: float
    residual_ms: float
    process_variance_raw: float
    condition_variance_raw: float
    interaction_variance_raw: float
    residual_variance: float
    process_variance: float
    interaction_variance: float
    interaction_share: float


def locate_member(zf: zipfile.ZipFile) -> str:
    matches = [name for name in zf.namelist() if name.endswith(RUN_LEVEL_SUFFIX)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one member ending with {RUN_LEVEL_SUFFIX!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def load_run_level(artifact_zip: Path | None) -> Tuple[pd.DataFrame, str]:
    if artifact_zip is None:
        path = ROOT / RUN_LEVEL_SUFFIX
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        input_reference = RUN_LEVEL_SUFFIX
    else:
        if not artifact_zip.exists():
            raise FileNotFoundError(artifact_zip)
        with zipfile.ZipFile(artifact_zip) as zf:
            member = locate_member(zf)
            df = pd.read_csv(io.BytesIO(zf.read(member)))
        input_reference = f"{artifact_zip.name}:{member}"

    required = {
        "hmm_id",
        "architecture",
        "dimension",
        "seed",
        "shape_rmse",
        "excess_ce_H",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["condition"] = (
        df["architecture"].astype(str) + "_d" + df["dimension"].astype(str)
    )
    return df, input_reference


def ordered_conditions(df: pd.DataFrame) -> List[str]:
    arch_order = {"GRU": 0, "Transformer": 1}
    pairs = (
        df[["architecture", "dimension", "condition"]]
        .drop_duplicates()
        .sort_values(
            ["architecture", "dimension"],
            key=lambda s: s.map(arch_order) if s.name == "architecture" else s,
        )
    )
    return pairs["condition"].tolist()


def to_array(
    df: pd.DataFrame, outcome: str
) -> Tuple[np.ndarray, List[str], List[str], List[int]]:
    hmm_ids = sorted(df["hmm_id"].astype(str).unique())
    conditions = ordered_conditions(df)
    seeds = sorted(int(x) for x in df["seed"].unique())
    expected = len(hmm_ids) * len(conditions) * len(seeds)
    if len(df) != expected:
        raise ValueError(
            f"Unbalanced design: {len(df)} rows, expected {expected} "
            f"({len(hmm_ids)} HMMs x {len(conditions)} conditions x {len(seeds)} seeds)."
        )
    arr = np.empty((len(hmm_ids), len(conditions), len(seeds)), dtype=float)
    for i, hmm_id in enumerate(hmm_ids):
        for c, condition in enumerate(conditions):
            sub = df.loc[
                (df["hmm_id"].astype(str) == hmm_id)
                & (df["condition"] == condition),
                ["seed", outcome],
            ].sort_values("seed")
            if sub["seed"].tolist() != seeds:
                raise ValueError(f"Incomplete seed set for {hmm_id}, {condition}")
            arr[i, c, :] = sub[outcome].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {outcome}")
    return arr, hmm_ids, conditions, seeds


def standardize_conditions(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize each condition over all HMM x seed observations (ddof=0)."""
    means = arr.mean(axis=(0, 2), keepdims=True)
    sds = arr.std(axis=(0, 2), ddof=0, keepdims=True)
    if np.any(sds <= 0):
        raise ValueError("At least one condition has zero variance")
    return (arr - means) / sds, means.ravel(), sds.ravel()


def estimate_components(z: np.ndarray) -> Estimate:
    """Balanced two-way random-effects ANOVA with replicated cells.

    Model: Z_ics = mu + P_i + C_c + (PC)_ic + e_ics.
    Interaction share among latent process-varying terms:
        sigma_PC^2 / (sigma_P^2 + sigma_PC^2).
    Negative method-of-moments estimates are truncated at zero for the share.
    """
    n_hmms, n_conditions, n_seeds = z.shape
    grand = float(z.mean())
    mean_process = z.mean(axis=(1, 2))
    mean_condition = z.mean(axis=(0, 2))
    mean_cell = z.mean(axis=2)

    process_ss = float(
        n_conditions * n_seeds * np.sum((mean_process - grand) ** 2)
    )
    condition_ss = float(
        n_hmms * n_seeds * np.sum((mean_condition - grand) ** 2)
    )
    interaction_residual = (
        mean_cell - mean_process[:, None] - mean_condition[None, :] + grand
    )
    interaction_ss = float(n_seeds * np.sum(interaction_residual**2))
    residual_ss = float(np.sum((z - mean_cell[:, :, None]) ** 2))

    process_ms = process_ss / (n_hmms - 1)
    condition_ms = condition_ss / (n_conditions - 1)
    interaction_ms = interaction_ss / ((n_hmms - 1) * (n_conditions - 1))
    residual_ms = residual_ss / (n_hmms * n_conditions * (n_seeds - 1))

    process_variance_raw = (process_ms - interaction_ms) / (
        n_conditions * n_seeds
    )
    condition_variance_raw = (condition_ms - interaction_ms) / (
        n_hmms * n_seeds
    )
    interaction_variance_raw = (interaction_ms - residual_ms) / n_seeds

    process_variance = max(float(process_variance_raw), 0.0)
    interaction_variance = max(float(interaction_variance_raw), 0.0)
    denominator = process_variance + interaction_variance
    interaction_share = (
        interaction_variance / denominator if denominator > 0 else float("nan")
    )

    return Estimate(
        n_hmms=n_hmms,
        n_conditions=n_conditions,
        n_seeds=n_seeds,
        process_ss=process_ss,
        condition_ss=condition_ss,
        interaction_ss=interaction_ss,
        residual_ss=residual_ss,
        process_ms=float(process_ms),
        condition_ms=float(condition_ms),
        interaction_ms=float(interaction_ms),
        residual_ms=float(residual_ms),
        process_variance_raw=float(process_variance_raw),
        condition_variance_raw=float(condition_variance_raw),
        interaction_variance_raw=float(interaction_variance_raw),
        residual_variance=float(residual_ms),
        process_variance=process_variance,
        interaction_variance=interaction_variance,
        interaction_share=float(interaction_share),
    )


def extract_k(hmm_id: str) -> int:
    match = re.search(r"_K(\d+)_", hmm_id)
    if not match:
        raise ValueError(f"Cannot infer K from HMM id: {hmm_id}")
    return int(match.group(1))


def bootstrap_shares(
    z: np.ndarray,
    hmm_ids: List[str],
    repeats: int,
    seed: int,
    stratified_by_k: bool,
) -> np.ndarray:
    """Cluster bootstrap over HMMs, preserving all conditions and seeds."""
    rng = np.random.default_rng(seed)
    n_hmms = z.shape[0]
    values = np.empty(repeats, dtype=float)
    if stratified_by_k:
        k_values = np.array([extract_k(x) for x in hmm_ids])
        groups = [np.flatnonzero(k_values == k) for k in sorted(set(k_values))]
    else:
        groups = []

    for b in range(repeats):
        if stratified_by_k:
            indices = np.concatenate(
                [rng.choice(group, size=len(group), replace=True) for group in groups]
            )
        else:
            indices = rng.integers(0, n_hmms, size=n_hmms)
        values[b] = estimate_components(z[indices]).interaction_share
    return values


def summarize_bootstrap(values: np.ndarray) -> Dict[str, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "zero_fraction": float(np.mean(values == 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-zip", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or ROOT / "outputs/variance_components"
    output_dir.mkdir(parents=True, exist_ok=True)
    df, input_reference = load_run_level(args.artifact_zip)

    summary_rows: List[Dict[str, object]] = []
    component_rows: List[Dict[str, object]] = []
    scaling_rows: List[Dict[str, object]] = []
    bootstrap_frames: List[pd.DataFrame] = []

    for label, column in OUTCOMES.items():
        arr, hmm_ids, conditions, seeds = to_array(df, column)
        z, means, sds = standardize_conditions(arr)
        point = estimate_components(z)

        ordinary = bootstrap_shares(
            z,
            hmm_ids,
            repeats=args.bootstrap_repeats,
            seed=args.bootstrap_seed,
            stratified_by_k=False,
        )
        stratified = bootstrap_shares(
            z,
            hmm_ids,
            repeats=args.bootstrap_repeats,
            seed=args.bootstrap_seed,
            stratified_by_k=True,
        )
        ordinary_ci = summarize_bootstrap(ordinary)
        stratified_ci = summarize_bootstrap(stratified)

        summary_rows.append(
            {
                "outcome": label,
                "run_level_column": column,
                "interaction_share": point.interaction_share,
                "interaction_share_percent": 100.0 * point.interaction_share,
                "ordinary_cluster_ci_lower_percent": 100.0
                * ordinary_ci["ci_lower"],
                "ordinary_cluster_ci_upper_percent": 100.0
                * ordinary_ci["ci_upper"],
                "ordinary_zero_fraction": ordinary_ci["zero_fraction"],
                "k_stratified_ci_lower_percent": 100.0
                * stratified_ci["ci_lower"],
                "k_stratified_ci_upper_percent": 100.0
                * stratified_ci["ci_upper"],
                "k_stratified_zero_fraction": stratified_ci["zero_fraction"],
            }
        )

        component_row = {"outcome": label, "run_level_column": column}
        component_row.update(asdict(point))
        component_rows.append(component_row)

        for condition, mean, sd in zip(conditions, means, sds):
            scaling_rows.append(
                {
                    "outcome": label,
                    "condition": condition,
                    "full_cohort_mean": float(mean),
                    "full_cohort_population_sd": float(sd),
                }
            )

        bootstrap_frames.append(
            pd.DataFrame(
                {
                    "outcome": label,
                    "replicate": np.arange(1, args.bootstrap_repeats + 1),
                    "ordinary_cluster_share": ordinary,
                    "k_stratified_cluster_share": stratified,
                }
            )
        )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "seed_aware_variance_summary.csv", index=False
    )
    pd.DataFrame(component_rows).to_csv(
        output_dir / "seed_aware_variance_components.csv", index=False
    )
    pd.DataFrame(scaling_rows).to_csv(
        output_dir / "seed_aware_condition_scaling.csv", index=False
    )
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(
        output_dir / "seed_aware_bootstrap_shares.csv", index=False
    )

    metadata = {
        "analysis": "Confirmatory neural seed-aware variance components",
        "input_reference": input_reference,
        "input_rows": len(df),
        "n_hmms": 64,
        "n_conditions": 8,
        "n_seeds": 5,
        "condition_definition": "architecture x representation width",
        "standardization": (
            "Each outcome is standardized separately within each of the eight "
            "conditions over all 64 HMM x 5 seed observations using population "
            "standard deviation (ddof=0). The full-cohort transformation is fixed "
            "during bootstrap resampling."
        ),
        "variance_model": "Z_ics = mu + P_i + C_c + (PC)_ic + e_ics",
        "estimator": "balanced ANOVA method-of-moments",
        "interaction_share_definition": (
            "max(sigma_PC^2,0) / [max(sigma_P^2,0) + max(sigma_PC^2,0)]"
        ),
        "bootstrap": {
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
            "ordinary": (
                "Resample 64 HMMs with replacement; preserve all 8 conditions "
                "and 5 seeds for each draw."
            ),
            "k_stratified_sensitivity": (
                "Resample 16 HMMs with replacement within each K=3,4,5,6 stratum."
            ),
            "interval": "2.5th and 97.5th percentile",
        },
        "historical_note": (
            "The point estimates reproduce the earlier manuscript draft before "
            "rounding. The original historical bootstrap code was not preserved; "
            "v0.4 therefore defines and archives this fully reproducible replacement."
        ),
    }
    (output_dir / "seed_aware_variance_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    readme = """# Seed-aware variance-component outputs

Generated by `code/scripts/reconstruct_seed_aware_variance.py` from the
2,560 confirmatory run-level rows.

The primary ordinary HMM-cluster bootstrap uses 10,000 replicates and random
seed 20260718. The K-stratified results are a sensitivity analysis.

Expected rounded primary results:

| Outcome | Interaction share | 95% interval |
|---|---:|---:|
| Shape RMSE | 0.2% | 0.0%--8.5% |
| Terminal excess CE | 16.5% | 2.7%--28.4% |
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
