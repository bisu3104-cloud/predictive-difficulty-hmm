# -*- coding: utf-8 -*-
"""
Dataset-realization robustness experiment (public v0.5 runner) for the paper
"Predictive Difficulty Beyond Hidden-State Cardinality".

Designed for a single Google Colab code cell. Re-running the same cell:
- mounts Google Drive,
- reuses the locked HMM subset,
- reuses completed reference-replicate runs,
- resumes an interrupted neural condition from its per-epoch checkpoint,
- skips completed conditions,
- runs all analyses once the grid is complete.

Default design:
- 16 confirmatory HMMs selected without learner outcomes (4 per K),
- 3 independent dataset realizations per HMM: existing replicate 0 + 2 new replicates,
- GRU and Transformer,
- widths 8, 32, 128, 256,
- training seeds 1, 2, 3,
- 768 new neural runs; replicate 0 reuses the corresponding existing runs.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError("matplotlib is required") from exc

try:
    from scipy.stats import spearmanr
except Exception as exc:
    raise RuntimeError("scipy is required") from exc

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error, r2_score
except Exception as exc:
    raise RuntimeError("scikit-learn is required") from exc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence
except Exception as exc:
    raise RuntimeError("PyTorch is required") from exc


# =============================================================================
# 1. User-editable configuration
# =============================================================================

@dataclass
class Config:
    # Existing independent 64-HMM confirmatory project. This folder is read only.
    CONFIRMATORY_ROOT: str = os.environ.get(
        "PREDICTIVE_CONFIRMATORY_ROOT",
        str(Path.cwd() / "predictive_profile_confirmatory64_v1"),
    )

    # New output folder. The original confirmatory project is never overwritten.
    PROJECT_ROOT: str = os.environ.get(
        "DATASET_REALIZATION_PROJECT_ROOT",
        str(Path.cwd() / "dataset_realization_robustness_v1"),
    )

    MASTER_SEED: int = 2026080401
    DEVICE: str = "auto"  # auto / cuda / cpu
    DETERMINISTIC_TORCH: bool = True

    # Outcome-blind subset selection from the existing 64 HMMs.
    K_VALUES: Tuple[int, ...] = (3, 4, 5, 6)
    HMMS_PER_K: int = 4

    # Dataset replicate 0 is the existing cached confirmatory dataset.
    # Replicates 1 and 2 are newly generated train/validation/test datasets.
    NEW_DATASET_REPLICATES: Tuple[int, ...] = (1, 2)
    REFERENCE_REPLICATE: int = 0
    REUSE_EXISTING_REFERENCE_RUNS: bool = True
    TRAIN_MISSING_REFERENCE_RUNS: bool = True

    ARCHITECTURES: Tuple[str, ...] = ("GRU", "Transformer")
    DIMENSIONS: Tuple[int, ...] = (8, 32, 128, 256)
    TRAINING_SEEDS: Tuple[int, ...] = (1, 2, 3)

    VOCAB_SIZE: int = 8
    MAX_CONTEXT: int = 4
    N_TRAIN: int = 256
    N_VAL: int = 32
    N_TEST: int = 64
    SEQUENCE_LENGTH: int = 300
    EVAL_WINDOWS_PER_SEQUENCE: int = 8

    BATCH_SIZE: int = 16
    MIN_BATCH_SIZE: int = 2
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 0.0
    MAX_EPOCHS: int = 48
    MIN_EPOCHS: int = 12
    EARLY_STOP_PATIENCE: int = 8
    GRAD_CLIP_NORM: float = 1.0
    DROPOUT: float = 0.0
    TRANSFORMER_HEADS: int = 1
    TRANSFORMER_LAYERS: int = 1
    TRANSFORMER_FF_MULTIPLIER: int = 4

    # Crash recovery / storage controls.
    CONTINUE_AFTER_ERROR: bool = True
    RETRY_FAILED_CONDITIONS: bool = True
    FORCE_RETRAIN: bool = False
    FORCE_REGENERATE_DATASETS: bool = False
    CLEAN_SUCCESSFUL_CHECKPOINTS: bool = True
    # None = run every remaining condition until Colab stops.
    MAX_NEW_CONDITIONS_PER_EXECUTION: Optional[int] = None

    RUN_TRAINING: bool = True
    RUN_ANALYSIS: bool = True
    PREPARE_ONLY: bool = False
    ANALYZE_ONLY: bool = False

    BOOTSTRAP_REPEATS: int = 5000
    BOOTSTRAP_SEED: int = 2026080402

    # Set True only for a tiny end-to-end check. Use a separate output folder.
    QUICK_SMOKE_TEST: bool = False

    def apply_smoke_test(self) -> None:
        if not self.QUICK_SMOKE_TEST:
            return
        self.PROJECT_ROOT = str(Path(self.PROJECT_ROOT).with_name(
            Path(self.PROJECT_ROOT).name + "_smoke"
        ))
        self.K_VALUES = (3, 4)
        self.HMMS_PER_K = 1
        self.NEW_DATASET_REPLICATES = (1,)
        self.ARCHITECTURES = ("GRU",)
        self.DIMENSIONS = (8,)
        self.TRAINING_SEEDS = (1,)
        self.N_TRAIN = 16
        self.N_VAL = 8
        self.N_TEST = 8
        self.SEQUENCE_LENGTH = 40
        self.EVAL_WINDOWS_PER_SEQUENCE = 2
        self.BATCH_SIZE = 8
        self.MAX_EPOCHS = 2
        self.MIN_EPOCHS = 1
        self.EARLY_STOP_PATIENCE = 1
        self.REUSE_EXISTING_REFERENCE_RUNS = False
        self.TRAIN_MISSING_REFERENCE_RUNS = True
        self.BOOTSTRAP_REPEATS = 100


# Ordinary use: edit only this dictionary.
USER_OVERRIDES: Dict[str, Any] = {
    "QUICK_SMOKE_TEST": False,
    # "HMMS_PER_K": 6,  # 24 HMMs total; increases new runs from 768 to 1152.
    # "MAX_NEW_CONDITIONS_PER_EXECUTION": 100,
    # "ANALYZE_ONLY": True,
}

CFG = Config()
for _key, _value in USER_OVERRIDES.items():
    if not hasattr(CFG, _key):
        raise KeyError(f"Unknown configuration key: {_key}")
    setattr(CFG, _key, _value)
if os.environ.get("DATAREP_SMOKE_TEST", "0") == "1":
    CFG.QUICK_SMOKE_TEST = True
CFG.apply_smoke_test()


# =============================================================================
# 2. Paths, logging, atomic writes, reproducibility
# =============================================================================

def in_colab() -> bool:
    return "google.colab" in sys.modules


def mount_drive_if_needed() -> None:
    if not in_colab():
        return
    from google.colab import drive  # type: ignore
    if not Path("/content/drive") / "MyDrive".exists():
        drive.mount("/content/drive")


def paths_for(cfg: Config) -> Dict[str, Path]:
    root = Path(cfg.PROJECT_ROOT)
    paths = {
        "root": root,
        "config": root / "config",
        "selection": root / "selection",
        "datasets": root / "datasets",
        "runs": root / "runs",
        "analysis": root / "analysis",
        "tables": root / "analysis" / "tables",
        "figures": root / "analysis" / "figures",
        "logs": root / "logs",
        "failures": root / "failures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logger(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("dataset_realization_robustness")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def safe_torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def stable_hash(text: str, modulo: int = 2_000_000_000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def resolve_device(cfg: Config) -> torch.device:
    if cfg.DEVICE == "cpu":
        return torch.device("cpu")
    if cfg.DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE='cuda', but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sanitize_identifier(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def record_failure(paths: Dict[str, Path], stage: str, identifier: str, exc: BaseException) -> None:
    atomic_write_json(
        paths["failures"] / f"{stage}__{sanitize_identifier(identifier)}.json",
        {
            "stage": stage,
            "identifier": identifier,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


# =============================================================================
# 3. Existing confirmatory project discovery and locked subset selection
# =============================================================================

RAW_PROFILE_COLUMNS = {
    "gain": "context_gain_area",
    "rank": "pred_h4_m2_r95",
    "sigma": "pred_h4_m2_sigma95",
    "tau": "tau_observed_proxy",
    "bayes": "bayes_ce_h4",
    "entropy": "symbol_entropy",
    "pmin_candidates": ("min_symbol_prob_recomputed", "min_symbol_prob"),
    "signal": "pred_h4_m2_signal",
}
TRANSFORMED_PROFILE_COLUMNS = (
    "log_gain",
    "log_rank",
    "neg_log_sigma",
    "log_tau",
    "bayes_ce_h4",
    "symbol_entropy",
    "log_min_symbol_prob",
    "log_signal",
)


def first_existing(candidates: Sequence[Path], description: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {description}. Checked:\n" + "\n".join(str(p) for p in candidates)
    )


def confirmatory_files(cfg: Config) -> Dict[str, Path]:
    root = Path(cfg.CONFIRMATORY_ROOT)
    selected_metrics = first_existing(
        [
            root / "selected_hmms" / "confirmatory_selected_hmm_metrics_full.csv",
            root / "selected_hmms" / "selected_hmm_metrics_full.csv",
            root / "analysis" / "tables" / "confirmatory_feature_outcome_table.csv",
        ],
        "confirmatory selected-HMM metrics",
    )
    frozen_predictions = first_existing(
        [
            root / "analysis" / "tables" / "confirmatory_holdout_predictions.csv",
            root / "analysis" / "tables" / "holdout_predictions.csv",
        ],
        "frozen development-to-confirmatory predictions",
    )
    frozen_coefficients = first_existing(
        [
            root / "analysis" / "tables" / "frozen_development_model_coefficients.csv",
            root / "analysis" / "tables" / "confirmatory_frozen_development_model_coefficients.csv",
        ],
        "frozen development model coefficients",
    )
    reference_results_candidates = [
        root / "analysis" / "tables" / "confirmatory_training_with_metrics.csv",
        root / "analysis" / "tables" / "confirmatory_training_raw.csv",
    ]
    reference_results_csv = next(
        (p for p in reference_results_candidates if p.exists()),
        None,
    )
    return {
        "root": root,
        "selected_metrics": selected_metrics,
        "selected_hmm_dir": root / "selected_hmms",
        "datasets": root / "datasets",
        "runs": root / "runs",
        "reference_results_csv": reference_results_csv,
        "frozen_predictions": frozen_predictions,
        "frozen_coefficients": frozen_coefficients,
    }


def load_confirmatory_metrics(cfg: Config, files: Dict[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(files["selected_metrics"])
    if "architecture" in df.columns:
        df = df.sort_values(["hmm_id", "architecture"]).drop_duplicates("hmm_id", keep="first")
    else:
        df = df.drop_duplicates("hmm_id", keep="first")
    if "K" not in df.columns:
        raise KeyError("Selected-HMM metrics must contain K")

    pmin_col = next((c for c in RAW_PROFILE_COLUMNS["pmin_candidates"] if c in df.columns), None)
    required = [
        "hmm_id", "K", RAW_PROFILE_COLUMNS["gain"], RAW_PROFILE_COLUMNS["rank"],
        RAW_PROFILE_COLUMNS["sigma"], RAW_PROFILE_COLUMNS["tau"], RAW_PROFILE_COLUMNS["bayes"],
        RAW_PROFILE_COLUMNS["entropy"], RAW_PROFILE_COLUMNS["signal"],
        "bayes_ce_h1", "bayes_ce_h2", "bayes_ce_h3", "bayes_ce_h4",
    ]
    missing = [c for c in required if c not in df.columns]
    if pmin_col is None:
        missing.append("minimum symbol probability")
    if missing:
        raise KeyError(f"Missing columns in selected-HMM metrics: {missing}")

    eps = 1e-15
    out = df.copy()
    out["log_gain"] = np.log(np.clip(out[RAW_PROFILE_COLUMNS["gain"]].astype(float), eps, None))
    out["log_rank"] = np.log(np.clip(out[RAW_PROFILE_COLUMNS["rank"]].astype(float), eps, None))
    out["neg_log_sigma"] = -np.log(np.clip(out[RAW_PROFILE_COLUMNS["sigma"]].astype(float), eps, None))
    out["log_tau"] = np.log(np.clip(out[RAW_PROFILE_COLUMNS["tau"]].astype(float), eps, None))
    out["log_min_symbol_prob"] = np.log(np.clip(out[pmin_col].astype(float), eps, None))
    out["log_signal"] = np.log(np.clip(out[RAW_PROFILE_COLUMNS["signal"]].astype(float), eps, None))

    hmm_files = []
    dataset_files = []
    for hmm_id in out["hmm_id"].astype(str):
        hmm_path = files["selected_hmm_dir"] / f"{sanitize_identifier(hmm_id)}.npz"
        if not hmm_path.exists():
            hmm_path = files["selected_hmm_dir"] / f"{hmm_id}.npz"
        dataset_path = files["datasets"] / f"{sanitize_identifier(hmm_id)}.npz"
        hmm_files.append(str(hmm_path))
        dataset_files.append(str(dataset_path))
    out["hmm_file"] = hmm_files
    out["reference_dataset_file"] = dataset_files
    return out


def selection_config_hash(cfg: Config) -> str:
    payload = {
        "K_VALUES": list(cfg.K_VALUES),
        "HMMS_PER_K": cfg.HMMS_PER_K,
        "MASTER_SEED": cfg.MASTER_SEED,
        "features": list(TRANSFORMED_PROFILE_COLUMNS),
        "method": "within-K kmeans medoids",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def runs_or_checkpoints_exist(paths: Dict[str, Path]) -> bool:
    return any(paths["runs"].rglob("DONE.json")) or any(paths["runs"].rglob("checkpoint.pt"))


def select_and_lock_subset(
    all_metrics: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
    logger: logging.Logger,
) -> pd.DataFrame:
    selected_path = paths["selection"] / "dataset_replication_selected_hmms.csv"
    lock_path = paths["selection"] / "SELECTION_LOCK.json"
    config_hash = selection_config_hash(cfg)

    if selected_path.exists() and lock_path.exists():
        lock = json.load(lock_path.open("r", encoding="utf-8"))
        if lock.get("selection_config_sha256") != config_hash:
            raise RuntimeError(
                "Selection settings changed after the selection was locked. "
                "Restore the original settings or use a new PROJECT_ROOT."
            )
        selected = pd.read_csv(selected_path)
        logger.info("Reusing locked outcome-blind HMM subset: %s", selected_path)
        return selected

    if runs_or_checkpoints_exist(paths):
        raise RuntimeError("Runs exist but no valid selection lock was found; refusing to reselect")

    rows: List[pd.DataFrame] = []
    for K in cfg.K_VALUES:
        group = all_metrics[all_metrics["K"].astype(int) == int(K)].copy().reset_index(drop=True)
        if len(group) < cfg.HMMS_PER_K:
            raise RuntimeError(f"K={K}: only {len(group)} HMMs available")
        X = group[list(TRANSFORMED_PROFILE_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError(f"K={K}: non-finite profile value")
        mean = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        scale[scale < 1e-12] = 1.0
        Z = (X - mean) / scale
        km = KMeans(
            n_clusters=cfg.HMMS_PER_K,
            random_state=cfg.MASTER_SEED + int(K),
            n_init=50,
            max_iter=1000,
        )
        labels = km.fit_predict(Z)
        chosen_idx: List[int] = []
        chosen_meta: Dict[int, Tuple[int, float, int]] = {}
        for cluster in range(cfg.HMMS_PER_K):
            members = np.where(labels == cluster)[0]
            center = km.cluster_centers_[cluster]
            distances = np.sqrt(np.mean((Z[members] - center) ** 2, axis=1))
            local = int(members[int(np.argmin(distances))])
            chosen_idx.append(local)
            chosen_meta[local] = (cluster, float(distances.min()), int(len(members)))
        chosen = group.iloc[chosen_idx].copy()
        chosen["subset_cluster"] = [chosen_meta[i][0] for i in chosen_idx]
        chosen["distance_to_subset_cluster_center"] = [chosen_meta[i][1] for i in chosen_idx]
        chosen["subset_cluster_size"] = [chosen_meta[i][2] for i in chosen_idx]
        rows.append(chosen)

    selected = pd.concat(rows, ignore_index=True).sort_values(["K", "subset_cluster"])
    selected = selected.reset_index(drop=True)
    expected = len(cfg.K_VALUES) * cfg.HMMS_PER_K
    if len(selected) != expected or selected["hmm_id"].nunique() != expected:
        raise AssertionError(f"Expected {expected} unique HMMs, found {len(selected)}")

    missing_hmm_files = []
    missing_reference_datasets = []
    for _, row in selected.iterrows():
        hmm_path = Path(str(row["hmm_file"]))
        if not hmm_path.exists():
            missing_hmm_files.append(str(hmm_path))
        reference_path = Path(str(row["reference_dataset_file"]))
        if not reference_path.exists():
            missing_reference_datasets.append(str(reference_path))
    if missing_hmm_files and not cfg.QUICK_SMOKE_TEST:
        raise FileNotFoundError(
            "Missing selected-HMM parameter files:\n" + "\n".join(missing_hmm_files[:20])
        )
    if missing_reference_datasets:
        logger.warning(
            "%d selected HMMs do not have the original cached dataset. "
            "This is acceptable when the corresponding replicate-0 run summaries are available; "
            "only newly generated dataset replicates will be trained.",
            len(missing_reference_datasets),
        )

    atomic_write_csv(selected_path, selected)
    ids = selected["hmm_id"].astype(str).tolist()
    atomic_write_json(
        lock_path,
        {
            "status": "locked_before_new_training",
            "selection_method": "within-K k-means coverage; nearest observed HMM to each centroid",
            "learner_outcomes_used": False,
            "features": list(TRANSFORMED_PROFILE_COLUMNS),
            "selection_config_sha256": config_hash,
            "hmm_ids_sha256": hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest(),
            "hmm_ids": ids,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    logger.info("Selected and locked %d HMMs without learner outcomes", len(selected))
    return selected


# =============================================================================
# 4. HMM simulation and independent dataset replicates
# =============================================================================

def generate_sequences(
    T: np.ndarray,
    O: np.ndarray,
    pi: np.ndarray,
    n_sequences: int,
    length: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    K, V = O.shape
    sequences = np.empty((n_sequences, length), dtype=np.int16)
    for n in range(n_sequences):
        z = int(rng.choice(K, p=pi))
        for t in range(length):
            sequences[n, t] = int(rng.choice(V, p=O[z]))
            z = int(rng.choice(K, p=T[z]))
    return sequences


def validate_dataset(path: Path, cfg: Config) -> None:
    with np.load(path) as data:
        for key, n in (("train", cfg.N_TRAIN), ("val", cfg.N_VAL), ("test", cfg.N_TEST)):
            if key not in data:
                raise KeyError(f"{path}: missing {key}")
            arr = data[key]
            if arr.shape[0] < n or arr.shape[1] != cfg.SEQUENCE_LENGTH:
                raise ValueError(f"{path}: unexpected {key} shape {arr.shape}")
            if arr.min() < 0 or arr.max() >= cfg.VOCAB_SIZE:
                raise ValueError(f"{path}: invalid symbols in {key}")


def dataset_path_for(paths: Dict[str, Path], replicate: int, hmm_id: str) -> Path:
    return paths["datasets"] / f"replicate_{replicate}" / f"{sanitize_identifier(hmm_id)}.npz"


def prepare_dataset_replicates(
    selected: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
    logger: logging.Logger,
) -> Dict[Tuple[int, str], Path]:
    mapping: Dict[Tuple[int, str], Path] = {}
    metadata_rows: List[Dict[str, Any]] = []

    for _, row in selected.iterrows():
        hmm_id = str(row["hmm_id"])
        reference = Path(str(row["reference_dataset_file"]))
        if reference.exists():
            validate_dataset(reference, cfg)
            mapping[(cfg.REFERENCE_REPLICATE, hmm_id)] = reference
            metadata_rows.append({
                "hmm_id": hmm_id,
                "dataset_replicate": cfg.REFERENCE_REPLICATE,
                "source": "existing_confirmatory_cache",
                "dataset_file": str(reference),
                "sha256": file_sha256(reference),
                "train_seed": np.nan,
                "val_seed": np.nan,
                "test_seed": np.nan,
            })
        else:
            metadata_rows.append({
                "hmm_id": hmm_id,
                "dataset_replicate": cfg.REFERENCE_REPLICATE,
                "source": "existing_run_summaries_only_cache_unavailable",
                "dataset_file": "",
                "sha256": "",
                "train_seed": np.nan,
                "val_seed": np.nan,
                "test_seed": np.nan,
            })

        hmm_file = Path(str(row["hmm_file"]))
        if not hmm_file.exists():
            raise FileNotFoundError(f"HMM file not found: {hmm_file}")
        with np.load(hmm_file) as hmm:
            T = np.asarray(hmm["T"], dtype=np.float64)
            O = np.asarray(hmm["O"], dtype=np.float64)
            pi = np.asarray(hmm["pi"], dtype=np.float64)

        reps_to_generate = list(cfg.NEW_DATASET_REPLICATES)
        if cfg.QUICK_SMOKE_TEST and not reference.exists():
            reps_to_generate = [cfg.REFERENCE_REPLICATE] + reps_to_generate

        for rep in reps_to_generate:
            out = dataset_path_for(paths, rep, hmm_id)
            run_rep_dir = paths["runs"] / f"replicate_{rep}"
            existing_runs = any(run_rep_dir.rglob("DONE.json")) or any(run_rep_dir.rglob("checkpoint.pt"))
            if cfg.FORCE_REGENERATE_DATASETS and existing_runs:
                raise RuntimeError(
                    f"Refusing to regenerate dataset replicate {rep} after training started"
                )
            if out.exists() and not cfg.FORCE_REGENERATE_DATASETS:
                validate_dataset(out, cfg)
            else:
                train_seed = cfg.MASTER_SEED + stable_hash(f"{hmm_id}|dataset={rep}|train")
                val_seed = cfg.MASTER_SEED + stable_hash(f"{hmm_id}|dataset={rep}|val")
                test_seed = cfg.MASTER_SEED + stable_hash(f"{hmm_id}|dataset={rep}|test")
                train = generate_sequences(T, O, pi, cfg.N_TRAIN, cfg.SEQUENCE_LENGTH, train_seed)
                val = generate_sequences(T, O, pi, cfg.N_VAL, cfg.SEQUENCE_LENGTH, val_seed)
                test = generate_sequences(T, O, pi, cfg.N_TEST, cfg.SEQUENCE_LENGTH, test_seed)
                atomic_save_npz(out, train=train, val=val, test=test)
                logger.info("Generated dataset replicate %d for %s", rep, hmm_id)
            validate_dataset(out, cfg)
            mapping[(rep, hmm_id)] = out
            metadata_rows.append({
                "hmm_id": hmm_id,
                "dataset_replicate": rep,
                "source": "new_independent_generation" if rep != 0 else "smoke_generated_reference",
                "dataset_file": str(out),
                "sha256": file_sha256(out),
                "train_seed": cfg.MASTER_SEED + stable_hash(f"{hmm_id}|dataset={rep}|train"),
                "val_seed": cfg.MASTER_SEED + stable_hash(f"{hmm_id}|dataset={rep}|val"),
                "test_seed": cfg.MASTER_SEED + stable_hash(f"{hmm_id}|dataset={rep}|test"),
            })

    metadata = pd.DataFrame(metadata_rows).drop_duplicates(["hmm_id", "dataset_replicate"], keep="last")
    atomic_write_csv(paths["tables"] / "dataset_replicate_manifest.csv", metadata)

    # Assert that each HMM has distinct files/checksums across replicates.
    for hmm_id, group in metadata.groupby("hmm_id"):
        checksums = group.loc[group["sha256"].astype(str).str.len() > 0, "sha256"]
        if checksums.duplicated().any():
            raise AssertionError(f"Duplicate dataset contents detected across replicates for {hmm_id}")
    return mapping


# =============================================================================
# 5. Neural learners and fixed-window evaluation
# =============================================================================

def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].copy() for key in ("train", "val", "test")}


def sample_training_windows(
    sequences: np.ndarray,
    max_context: int,
    vocab_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, length = sequences.shape
    h_values = rng.integers(1, max_context + 1, size=n)
    end_positions = np.array([rng.integers(h, length) for h in h_values], dtype=np.int64)
    x = np.full((n, max_context), vocab_size, dtype=np.int64)
    y = np.empty(n, dtype=np.int64)
    for idx, (h, end) in enumerate(zip(h_values, end_positions)):
        x[idx, :h] = sequences[idx, end - h:end]
        y[idx] = sequences[idx, end]
    return x, h_values.astype(np.int64), y


def fixed_eval_windows(
    sequences: np.ndarray,
    h: int,
    max_context: int,
    vocab_size: int,
    windows_per_sequence: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, length = sequences.shape
    total = n * windows_per_sequence
    x = np.full((total, max_context), vocab_size, dtype=np.int64)
    lengths = np.full(total, h, dtype=np.int64)
    y = np.empty(total, dtype=np.int64)
    row = 0
    for seq_idx in range(n):
        candidates = np.arange(h, length)
        if windows_per_sequence >= len(candidates):
            ends = candidates
        else:
            ends = rng.choice(candidates, size=windows_per_sequence, replace=False)
        for end in ends[:windows_per_sequence]:
            x[row, :h] = sequences[seq_idx, end - h:end]
            y[row] = sequences[seq_idx, end]
            row += 1
    return x[:row], lengths[:row], y[:row]


class GRUPredictor(nn.Module):
    def __init__(self, vocab_size: int, dimension: int, pad_token: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, dimension, padding_idx=pad_token)
        self.gru = nn.GRU(dimension, dimension, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dimension, vocab_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        packed = pack_padded_sequence(
            embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.output(self.dropout(hidden[-1]))


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dimension: int,
        max_context: int,
        pad_token: int,
        nhead: int,
        layers: int,
        ff_multiplier: int,
        dropout: float,
    ):
        super().__init__()
        if dimension % nhead != 0:
            raise ValueError("dimension must be divisible by nhead")
        self.pad_token = pad_token
        self.embedding = nn.Embedding(vocab_size + 1, dimension, padding_idx=pad_token)
        self.position = nn.Embedding(max_context, dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=nhead,
            dim_feedforward=max(dimension, ff_multiplier * dimension),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, vocab_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        hidden = self.embedding(x) + self.position(pos)
        key_padding_mask = x.eq(self.pad_token)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        last = (lengths - 1).clamp_min(0)
        representation = encoded[torch.arange(batch, device=x.device), last]
        return self.output(self.norm(representation))


def build_model(architecture: str, dimension: int, cfg: Config) -> nn.Module:
    if architecture.lower() == "gru":
        return GRUPredictor(cfg.VOCAB_SIZE, dimension, cfg.VOCAB_SIZE, cfg.DROPOUT)
    if architecture.lower() == "transformer":
        return TransformerPredictor(
            cfg.VOCAB_SIZE,
            dimension,
            cfg.MAX_CONTEXT,
            cfg.VOCAB_SIZE,
            cfg.TRANSFORMER_HEADS,
            cfg.TRANSFORMER_LAYERS,
            cfg.TRANSFORMER_FF_MULTIPLIER,
            cfg.DROPOUT,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def iter_minibatches(
    x: np.ndarray,
    lengths: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        yield x[idx], lengths[idx], y[idx]


def evaluate_model_ce(
    model: nn.Module,
    sequences: np.ndarray,
    cfg: Config,
    device: torch.device,
    batch_size: int,
    seed_base: int,
) -> Dict[int, float]:
    model.eval()
    curve: Dict[int, float] = {}
    with torch.no_grad():
        for h in range(1, cfg.MAX_CONTEXT + 1):
            x, lengths, y = fixed_eval_windows(
                sequences,
                h,
                cfg.MAX_CONTEXT,
                cfg.VOCAB_SIZE,
                cfg.EVAL_WINDOWS_PER_SEQUENCE,
                seed_base + h,
            )
            total_loss = 0.0
            total_n = 0
            for start in range(0, len(y), batch_size):
                xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
                lb = torch.as_tensor(lengths[start:start + batch_size], dtype=torch.long, device=device)
                yb = torch.as_tensor(y[start:start + batch_size], dtype=torch.long, device=device)
                loss = F.cross_entropy(model(xb, lb), yb, reduction="sum")
                total_loss += float(loss.item())
                total_n += int(yb.numel())
            curve[h] = total_loss / max(total_n, 1)
    return curve


def validation_score(
    model: nn.Module,
    sequences: np.ndarray,
    cfg: Config,
    device: torch.device,
    batch_size: int,
    seed_base: int,
) -> float:
    return float(np.mean(list(evaluate_model_ce(
        model, sequences, cfg, device, batch_size, seed_base
    ).values())))


# =============================================================================
# 6. Per-condition crash-safe training
# =============================================================================

def run_directory(
    paths: Dict[str, Path],
    replicate: int,
    hmm_id: str,
    architecture: str,
    dimension: int,
    seed: int,
    n_train: int,
) -> Path:
    return (
        paths["runs"] / f"replicate_{replicate}" / sanitize_identifier(hmm_id)
        / architecture.lower() / f"d{dimension}" / f"n{n_train}" / f"seed{seed}"
    )


def train_one_condition(
    dataset_replicate: int,
    hmm_id: str,
    architecture: str,
    dimension: int,
    training_seed: int,
    dataset_path: Path,
    cfg: Config,
    paths: Dict[str, Path],
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, Any]:
    run_dir = run_directory(
        paths, dataset_replicate, hmm_id, architecture, dimension, training_seed, cfg.N_TRAIN
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "DONE.json"
    failed_path = run_dir / "FAILED.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    best_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.csv"

    if done_path.exists() and not cfg.FORCE_RETRAIN:
        if cfg.CLEAN_SUCCESSFUL_CHECKPOINTS:
            for p in (checkpoint_path, best_path):
                if p.exists():
                    p.unlink()
        return json.load(done_path.open("r", encoding="utf-8"))
    if failed_path.exists() and not cfg.RETRY_FAILED_CONDITIONS and not cfg.FORCE_RETRAIN:
        return json.load(failed_path.open("r", encoding="utf-8"))
    if cfg.FORCE_RETRAIN:
        for p in (done_path, failed_path, checkpoint_path, best_path, history_path):
            if p.exists():
                p.unlink()

    data = load_dataset(dataset_path)
    train_sequences = data["train"][:cfg.N_TRAIN]
    val_sequences = data["val"][:cfg.N_VAL]
    test_sequences = data["test"][:cfg.N_TEST]

    condition_seed = cfg.MASTER_SEED + stable_hash(
        f"{hmm_id}|dataset_rep={dataset_replicate}|{architecture}|d={dimension}|seed={training_seed}"
    )
    set_global_seed(condition_seed, cfg.DETERMINISTIC_TORCH)
    model = build_model(architecture, dimension, cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    batch_size = cfg.BATCH_SIZE
    history: List[Dict[str, Any]] = []

    if checkpoint_path.exists() and not cfg.FORCE_RETRAIN:
        checkpoint = safe_torch_load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val"])
        best_epoch = int(checkpoint["best_epoch"])
        no_improve = int(checkpoint.get("no_improve", 0))
        batch_size = int(checkpoint.get("batch_size", batch_size))
        history = list(checkpoint.get("history", []))
        logger.info(
            "Resume rep=%d %s %s d=%d seed=%d from epoch %d",
            dataset_replicate, hmm_id, architecture, dimension, training_seed, start_epoch,
        )

    try:
        for epoch in range(start_epoch, cfg.MAX_EPOCHS + 1):
            while True:
                try:
                    model.train()
                    x, lengths, y = sample_training_windows(
                        train_sequences,
                        cfg.MAX_CONTEXT,
                        cfg.VOCAB_SIZE,
                        condition_seed + 1000 * epoch,
                    )
                    total_loss = 0.0
                    total_n = 0
                    for xb_np, lb_np, yb_np in iter_minibatches(
                        x, lengths, y, batch_size, condition_seed + 2000 * epoch
                    ):
                        xb = torch.as_tensor(xb_np, dtype=torch.long, device=device)
                        lb = torch.as_tensor(lb_np, dtype=torch.long, device=device)
                        yb = torch.as_tensor(yb_np, dtype=torch.long, device=device)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(xb, lb)
                        loss = F.cross_entropy(logits, yb)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Non-finite loss: {loss.item()}")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                        optimizer.step()
                        total_loss += float(loss.item()) * len(yb_np)
                        total_n += len(yb_np)
                    train_ce = total_loss / max(total_n, 1)
                    val_ce = validation_score(
                        model,
                        val_sequences,
                        cfg,
                        device,
                        max(batch_size, 32),
                        condition_seed + 300000,
                    )
                    break
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and batch_size > cfg.MIN_BATCH_SIZE:
                        old = batch_size
                        batch_size = max(cfg.MIN_BATCH_SIZE, batch_size // 2)
                        logger.warning("CUDA OOM: batch %d -> %d", old, batch_size)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise

            if val_ce < best_val - 1e-8:
                best_val = val_ce
                best_epoch = epoch
                no_improve = 0
                atomic_torch_save(
                    best_path,
                    {k: v.detach().cpu() for k, v in model.state_dict().items()},
                )
            else:
                no_improve += 1

            history.append({
                "epoch": epoch,
                "train_ce": train_ce,
                "val_ce": val_ce,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "batch_size": batch_size,
            })
            atomic_write_csv(history_path, pd.DataFrame(history))
            atomic_torch_save(
                checkpoint_path,
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "no_improve": no_improve,
                    "batch_size": batch_size,
                    "history": history,
                    "condition_seed": condition_seed,
                },
            )
            logger.info(
                "rep=%d | %s | %s d=%d seed=%d | epoch=%d train=%.4f val=%.4f",
                dataset_replicate, hmm_id, architecture, dimension, training_seed,
                epoch, train_ce, val_ce,
            )
            if epoch >= cfg.MIN_EPOCHS and no_improve >= cfg.EARLY_STOP_PATIENCE:
                break

        if best_path.exists():
            model.load_state_dict(safe_torch_load(best_path, map_location=device))
        test_curve = evaluate_model_ce(
            model, test_sequences, cfg, device, max(batch_size, 32), condition_seed + 400000
        )
        result: Dict[str, Any] = {
            "status": "done",
            "dataset_replicate": int(dataset_replicate),
            "hmm_id": hmm_id,
            "architecture": architecture,
            "dimension": int(dimension),
            "seed": int(training_seed),
            "n_train": int(cfg.N_TRAIN),
            "best_epoch": int(best_epoch),
            "best_val_ce": float(best_val),
            "final_batch_size": int(batch_size),
            "epochs_completed": int(history[-1]["epoch"] if history else 0),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "device": str(device),
            "condition_seed": int(condition_seed),
            "dataset_file": str(dataset_path),
        }
        for h, value in test_curve.items():
            result[f"test_ce_h{h}"] = float(value)
        atomic_write_json(done_path, result)
        if failed_path.exists():
            failed_path.unlink()
        if cfg.CLEAN_SUCCESSFUL_CHECKPOINTS:
            for p in (checkpoint_path, best_path):
                if p.exists():
                    p.unlink()
        return result

    except KeyboardInterrupt:
        logger.warning("Interrupted; checkpoint preserved at %s", checkpoint_path)
        raise
    except Exception as exc:
        atomic_write_json(
            failed_path,
            {
                "status": "failed",
                "dataset_replicate": dataset_replicate,
                "hmm_id": hmm_id,
                "architecture": architecture,
                "dimension": dimension,
                "seed": training_seed,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def collect_done_json(base: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not base.exists():
        return pd.DataFrame()
    for path in base.rglob("DONE.json"):
        try:
            row = json.load(path.open("r", encoding="utf-8"))
            row["result_file"] = str(path)
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def load_existing_reference_results(
    selected: pd.DataFrame,
    cfg: Config,
    files: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    if not cfg.REUSE_EXISTING_REFERENCE_RUNS:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []

    done_df = collect_done_json(files["runs"])
    if not done_df.empty:
        done_df["source"] = "existing_confirmatory_DONE"
        done_df["source_priority"] = 2
        frames.append(done_df)

    csv_path = files.get("reference_results_csv")
    if csv_path is not None and Path(csv_path).exists():
        csv_df = pd.read_csv(csv_path)
        if not csv_df.empty:
            csv_df["source"] = "existing_confirmatory_summary_csv"
            csv_df["source_priority"] = 1
            frames.append(csv_df)

    if not frames:
        logger.warning(
            "No existing confirmatory DONE.json files or run-level summary CSV were found"
        )
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    required = ["hmm_id", "architecture", "dimension", "seed", "n_train"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Existing confirmatory run results are missing columns: {missing}")

    df["architecture"] = (
        df["architecture"].astype(str).str.strip().str.lower()
        .map({"gru": "GRU", "transformer": "Transformer"})
        .fillna(df["architecture"].astype(str))
    )
    for col in ("dimension", "seed", "n_train"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    wanted = set(selected["hmm_id"].astype(str))
    df = df[
        df["hmm_id"].astype(str).isin(wanted)
        & df["architecture"].isin(cfg.ARCHITECTURES)
        & df["dimension"].astype("Int64").isin(cfg.DIMENSIONS)
        & df["seed"].astype("Int64").isin(cfg.TRAINING_SEEDS)
        & (df["n_train"].astype("Int64") == cfg.N_TRAIN)
    ].copy()

    keys = ["hmm_id", "architecture", "dimension", "seed"]
    df = (
        df.sort_values(keys + ["source_priority"])
        .drop_duplicates(keys, keep="last")
        .drop(columns=["source_priority"], errors="ignore")
    )
    df["dataset_replicate"] = cfg.REFERENCE_REPLICATE

    logger.info(
        "Reusing %d reference-replicate neural runs from %s",
        len(df),
        ", ".join(sorted(df["source"].astype(str).unique())),
    )
    return df


def validate_reference_replicate_availability(
    selected: pd.DataFrame,
    reference_results: pd.DataFrame,
    cfg: Config,
    logger: logging.Logger,
) -> None:
    expected = {
        expected_condition_key(
            cfg.REFERENCE_REPLICATE, hmm_id, arch, int(dim), int(seed)
        )
        for hmm_id in selected["hmm_id"].astype(str)
        for arch in cfg.ARCHITECTURES
        for dim in cfg.DIMENSIONS
        for seed in cfg.TRAINING_SEEDS
    }
    available = set()
    if not reference_results.empty:
        available = {
            expected_condition_key(
                int(r.dataset_replicate), str(r.hmm_id), str(r.architecture),
                int(r.dimension), int(r.seed)
            )
            for r in reference_results.itertuples()
        }
    missing = sorted(expected - available)

    missing_cache_hmms = [
        str(r.hmm_id)
        for r in selected.itertuples()
        if not Path(str(r.reference_dataset_file)).exists()
    ]

    if missing and missing_cache_hmms:
        preview = "\n".join(
            f"rep={rep} {hmm_id} {arch} d={dim} seed={seed}"
            for rep, hmm_id, arch, dim, seed in missing[:20]
        )
        raise FileNotFoundError(
            "The original cached confirmatory datasets are unavailable, and some requested "
            "replicate-0 run summaries are also missing. These missing conditions cannot be "
            "retrained on the original dataset. Missing examples:\n" + preview
        )

    if missing:
        logger.warning(
            "%d replicate-0 conditions are missing from the summaries; they will be trained "
            "from the original cached datasets.",
            len(missing),
        )
    elif missing_cache_hmms:
        logger.info(
            "Original cached datasets are absent for %d selected HMMs, but all %d requested "
            "replicate-0 run summaries are available. Replicate 0 will therefore be reused "
            "from the summary CSV without retraining.",
            len(missing_cache_hmms), len(expected),
        )

def all_conditions(selected: pd.DataFrame, cfg: Config) -> List[Tuple[int, str, str, int, int]]:
    reps = [cfg.REFERENCE_REPLICATE] + list(cfg.NEW_DATASET_REPLICATES)
    selected_order = selected.sort_values(["K", "subset_cluster"])["hmm_id"].astype(str).tolist()
    conditions = [
        (rep, hmm_id, arch, int(dim), int(seed))
        for rep in reps
        for seed in cfg.TRAINING_SEEDS
        for dim in cfg.DIMENSIONS
        for arch in cfg.ARCHITECTURES
        for hmm_id in selected_order
    ]
    return conditions


def expected_condition_key(
    rep: int, hmm_id: str, arch: str, dim: int, seed: int
) -> Tuple[int, str, str, int, int]:
    return int(rep), str(hmm_id), str(arch), int(dim), int(seed)


def train_grid(
    selected: pd.DataFrame,
    dataset_paths: Dict[Tuple[int, str], Path],
    reference_results: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[int, int]:
    conditions = all_conditions(selected, cfg)
    reference_keys = set()
    if not reference_results.empty:
        reference_keys = {
            expected_condition_key(
                row.dataset_replicate, row.hmm_id, row.architecture, row.dimension, row.seed
            )
            for row in reference_results.itertuples()
        }

    scheduled: List[Tuple[int, str, str, int, int]] = []
    for condition in conditions:
        rep, hmm_id, arch, dim, seed = condition
        key = expected_condition_key(*condition)
        if rep == cfg.REFERENCE_REPLICATE and key in reference_keys:
            continue
        if rep == cfg.REFERENCE_REPLICATE and not cfg.TRAIN_MISSING_REFERENCE_RUNS:
            continue
        done = run_directory(paths, rep, hmm_id, arch, dim, seed, cfg.N_TRAIN) / "DONE.json"
        if done.exists() and not cfg.FORCE_RETRAIN:
            continue
        scheduled.append(condition)

    if cfg.MAX_NEW_CONDITIONS_PER_EXECUTION is not None:
        scheduled = scheduled[:int(cfg.MAX_NEW_CONDITIONS_PER_EXECUTION)]

    expected = len(conditions)
    already = expected - len(scheduled)
    logger.info(
        "Grid total=%d, already available/skipped=%d, scheduled now=%d",
        expected, already, len(scheduled),
    )

    for idx, (rep, hmm_id, arch, dim, seed) in enumerate(scheduled, start=1):
        logger.info(
            "Condition %d/%d: rep=%d %s %s d=%d seed=%d",
            idx, len(scheduled), rep, hmm_id, arch, dim, seed,
        )
        try:
            dataset_key = (rep, hmm_id)
            if dataset_key not in dataset_paths:
                raise FileNotFoundError(
                    f"No dataset file is available for scheduled condition rep={rep}, "
                    f"hmm_id={hmm_id}. The original reference cache may have been removed."
                )
            train_one_condition(
                rep, hmm_id, arch, dim, seed,
                dataset_paths[dataset_key], cfg, paths, device, logger,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            record_failure(paths, "training", f"r{rep}_{hmm_id}_{arch}_d{dim}_s{seed}", exc)
            logger.error("Condition failed: %s", exc)
            if not cfg.CONTINUE_AFTER_ERROR:
                raise
        status = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "condition_completed_in_this_execution": idx,
            "conditions_scheduled_in_this_execution": len(scheduled),
        }
        atomic_write_json(paths["root"] / "RUN_STATUS.json", status)

    new_results = collect_done_json(paths["runs"])
    new_keys = set()
    if not new_results.empty:
        new_keys = {
            expected_condition_key(r.dataset_replicate, r.hmm_id, r.architecture, r.dimension, r.seed)
            for r in new_results.itertuples()
        }
    available = len(reference_keys | new_keys)
    return available, expected


# =============================================================================
# 7. Outcome reconstruction and balanced data validation
# =============================================================================

def combined_run_results(
    selected: pd.DataFrame,
    reference_results: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
) -> pd.DataFrame:
    new_results = collect_done_json(paths["runs"])
    if not new_results.empty:
        new_results["source"] = "dataset_replication_project"
    frames = [df for df in (reference_results, new_results) if not df.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    keys = ["dataset_replicate", "hmm_id", "architecture", "dimension", "seed"]
    # Prefer newly generated project results if a missing reference run was retrained.
    df["source_priority"] = (df["source"] == "dataset_replication_project").astype(int)
    df = df.sort_values(keys + ["source_priority"]).drop_duplicates(keys, keep="last")
    wanted = set(selected["hmm_id"].astype(str))
    df = df[
        df["hmm_id"].astype(str).isin(wanted)
        & df["architecture"].isin(cfg.ARCHITECTURES)
        & df["dimension"].astype(int).isin(cfg.DIMENSIONS)
        & df["seed"].astype(int).isin(cfg.TRAINING_SEEDS)
    ].copy()
    return df.drop(columns=["source_priority"], errors="ignore")


def attach_recovery_metrics(
    results: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Attach exact Bayes curves and reconstruct recovery metrics.

    Existing replicate-0 summary rows may already contain ``bayes_ce_h*``
    columns, while newly trained rows do not.  A direct merge therefore
    creates pandas suffixes (``_x``/``_y``) and makes the unsuffixed column
    unavailable.  We merge the selected-HMM values under temporary names and
    fill only missing values, so both row types can be analyzed together.
    """
    H = cfg.MAX_CONTEXT
    required_bayes = [f"bayes_ce_h{h}" for h in range(1, H + 1)]
    required_test = [f"test_ce_h{h}" for h in range(1, H + 1)]

    missing_selected = [c for c in ["hmm_id", *required_bayes] if c not in selected.columns]
    if missing_selected:
        raise KeyError(f"Selected-HMM table is missing Bayes columns: {missing_selected}")
    missing_test = [c for c in required_test if c not in results.columns]
    if missing_test:
        raise KeyError(f"Combined run results are missing test-CE columns: {missing_test}")

    lookup = (
        selected[["hmm_id", *required_bayes]]
        .drop_duplicates("hmm_id")
        .rename(columns={c: f"{c}__selected" for c in required_bayes})
    )
    merged = results.copy()

    # Repair dataframes produced by an earlier partial analysis, should any
    # suffixed columns already be present.  Prefer the original run-summary
    # value, then the selected-HMM lookup below.
    for c in required_bayes:
        if c not in merged.columns:
            candidates = [x for x in (f"{c}_x", f"{c}_y") if x in merged.columns]
            if candidates:
                merged[c] = merged[candidates].bfill(axis=1).iloc[:, 0]

    merged = merged.merge(lookup, on="hmm_id", how="left", validate="many_to_one")
    for c in required_bayes:
        selected_col = f"{c}__selected"
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(
                pd.to_numeric(merged[selected_col], errors="coerce")
            )
        else:
            merged[c] = pd.to_numeric(merged[selected_col], errors="coerce")
        merged = merged.drop(columns=[selected_col])

    missing_rows = merged[required_bayes].isna().any(axis=1)
    if missing_rows.any():
        examples = merged.loc[missing_rows, "hmm_id"].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(
            "Bayes cross-entropy values remain missing after lookup for HMMs: "
            + ", ".join(examples)
        )

    shape, excess, recovery = [], [], []
    for _, row in merged.iterrows():
        model_shape = np.array([
            float(row[f"test_ce_h{h}"]) - float(row[f"test_ce_h{H}"])
            for h in range(1, H + 1)
        ])
        bayes_shape = np.array([
            float(row[f"bayes_ce_h{h}"]) - float(row[f"bayes_ce_h{H}"])
            for h in range(1, H + 1)
        ])
        shape.append(float(np.sqrt(np.mean((model_shape - bayes_shape) ** 2))))
        excess.append(float(row[f"test_ce_h{H}"]) - float(row[f"bayes_ce_h{H}"]))
        gap = float(row["bayes_ce_h1"]) - float(row[f"bayes_ce_h{H}"])
        model_gap = float(row["test_ce_h1"]) - float(row[f"test_ce_h{H}"])
        recovery.append(model_gap / gap if abs(gap) > 1e-12 else np.nan)
    merged["shape_rmse"] = shape
    merged["excess_ce_H"] = excess
    merged["recovery_ratio"] = recovery
    return merged


def expected_grid_dataframe(selected: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dataset_replicate": rep,
            "hmm_id": hmm_id,
            "architecture": arch,
            "dimension": dim,
            "seed": seed,
        }
        for rep, hmm_id, arch, dim, seed in all_conditions(selected, cfg)
    ])


def grid_audit(run_metrics: pd.DataFrame, selected: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    expected = expected_grid_dataframe(selected, cfg)
    keys = ["dataset_replicate", "hmm_id", "architecture", "dimension", "seed"]
    observed = run_metrics[keys].drop_duplicates()
    audit = expected.merge(observed.assign(observed=True), on=keys, how="left")
    audit["observed"] = audit["observed"].fillna(False)
    return audit


def build_summaries(run_metrics: pd.DataFrame, selected: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    width = (
        run_metrics.groupby(
            ["dataset_replicate", "hmm_id", "architecture", "dimension"], as_index=False
        )
        .agg(
            shape_rmse=("shape_rmse", "mean"),
            shape_rmse_sd=("shape_rmse", "std"),
            excess_ce_H=("excess_ce_H", "mean"),
            excess_ce_H_sd=("excess_ce_H", "std"),
            best_val_ce=("best_val_ce", "mean"),
        )
    )
    rows: List[Dict[str, Any]] = []
    for (rep, hmm_id, arch), group in width.groupby(
        ["dataset_replicate", "hmm_id", "architecture"]
    ):
        group = group.sort_values("dimension")
        errors = group["shape_rmse"].to_numpy(float)
        dims = group["dimension"].to_numpy(int)
        best = float(np.min(errors))
        rows.append({
            "dataset_replicate": int(rep),
            "hmm_id": str(hmm_id),
            "architecture": str(arch),
            "A_width_shape": float(np.mean(errors)),
            "best_shape_rmse": best,
            "best_dimension": int(dims[int(np.argmin(errors))]),
            "mean_excess_ce_H": float(group["excess_ce_H"].mean()),
        })
    hmm_summary = pd.DataFrame(rows).merge(selected[["hmm_id", "K"]], on="hmm_id", how="left")
    return width, hmm_summary


# =============================================================================
# 8. Statistical analyses
# =============================================================================

def percentile_ci(values: Sequence[float], level: float = 0.95) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    alpha = 1.0 - level
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def pairwise_rank_stability(
    hmm_summary: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
    outcomes = ("A_width_shape", "mean_excess_ce_H")
    reps = sorted(hmm_summary["dataset_replicate"].unique())
    for arch in cfg.ARCHITECTURES:
        for outcome in outcomes:
            pivot = hmm_summary[hmm_summary["architecture"] == arch].pivot(
                index="hmm_id", columns="dataset_replicate", values=outcome
            ).dropna()
            for r1, r2 in combinations(reps, 2):
                if r1 not in pivot.columns or r2 not in pivot.columns or len(pivot) < 3:
                    continue
                x = pivot[r1].to_numpy(float)
                y = pivot[r2].to_numpy(float)
                rho = float(spearmanr(x, y).statistic)
                boot = []
                n = len(x)
                for _ in range(cfg.BOOTSTRAP_REPEATS):
                    idx = rng.integers(0, n, size=n)
                    val = spearmanr(x[idx], y[idx]).statistic
                    if np.isfinite(val):
                        boot.append(float(val))
                low, high = percentile_ci(boot) if boot else (np.nan, np.nan)
                rows.append({
                    "architecture": arch,
                    "outcome": outcome,
                    "replicate_1": int(r1),
                    "replicate_2": int(r2),
                    "n_hmms": int(n),
                    "spearman_rho": rho,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                })
    return pd.DataFrame(rows)


def one_way_icc(values: np.ndarray) -> Dict[str, float]:
    # values shape: processes x dataset replicates, balanced and complete.
    P, R = values.shape
    grand = values.mean()
    means = values.mean(axis=1)
    ss_between = R * np.sum((means - grand) ** 2)
    ss_within = np.sum((values - means[:, None]) ** 2)
    ms_between = ss_between / max(P - 1, 1)
    ms_within = ss_within / max(P * (R - 1), 1)
    sigma_process = max((ms_between - ms_within) / R, 0.0)
    sigma_dataset = max(ms_within, 0.0)
    denom = sigma_process + sigma_dataset
    icc = sigma_process / denom if denom > 0 else np.nan
    return {
        "ms_between": float(ms_between),
        "ms_within": float(ms_within),
        "process_variance": float(sigma_process),
        "dataset_realization_variance": float(sigma_dataset),
        "process_icc": float(icc),
    }


def hmm_level_icc(
    hmm_summary: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(cfg.BOOTSTRAP_SEED + 1)
    for arch in cfg.ARCHITECTURES:
        for outcome in ("A_width_shape", "mean_excess_ce_H"):
            pivot = hmm_summary[hmm_summary["architecture"] == arch].pivot(
                index="hmm_id", columns="dataset_replicate", values=outcome
            ).dropna()
            arr = pivot.to_numpy(float)
            if arr.shape[0] < 3 or arr.shape[1] < 2:
                continue
            point = one_way_icc(arr)
            boot = []
            for _ in range(cfg.BOOTSTRAP_REPEATS):
                idx = rng.integers(0, arr.shape[0], size=arr.shape[0])
                boot.append(one_way_icc(arr[idx])["process_icc"])
            low, high = percentile_ci([x for x in boot if np.isfinite(x)])
            rows.append({
                "architecture": arch,
                "outcome": outcome,
                "n_hmms": int(arr.shape[0]),
                "n_dataset_replicates": int(arr.shape[1]),
                **point,
                "icc_ci_low": low,
                "icc_ci_high": high,
            })
    return pd.DataFrame(rows)


def full_variance_components(
    run_metrics: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    # Model: Z_ircs = mu + P_i + C_c + PC_ic + D_ir + DC_irc + epsilon_ircs.
    # D is dataset realization nested within process.
    rows: List[Dict[str, Any]] = []
    condition = run_metrics["architecture"].astype(str) + "_d" + run_metrics["dimension"].astype(str)
    data = run_metrics.copy()
    data["condition"] = condition
    for outcome in ("shape_rmse", "excess_ce_H"):
        d = data[["hmm_id", "dataset_replicate", "condition", "seed", outcome]].copy()
        # Condition-wise population standardization over all HMM, dataset, and seed observations.
        d["z"] = d.groupby("condition")[outcome].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 1e-12 else 1.0)
        )
        processes = sorted(d["hmm_id"].unique())
        reps = sorted(d["dataset_replicate"].unique())
        conditions = sorted(d["condition"].unique())
        seeds = sorted(d["seed"].unique())
        P, R, C, S = len(processes), len(reps), len(conditions), len(seeds)
        expected = P * R * C * S
        if len(d) != expected:
            raise RuntimeError(
                f"Variance decomposition requires a balanced grid for {outcome}: "
                f"expected {expected}, found {len(d)}"
            )
        if P < 2 or R < 2 or C < 2 or S < 2:
            rows.append({
                "outcome": outcome,
                "P": P, "R": R, "C": C, "S": S,
                "note": "Variance components require at least two levels for P, R, C, and S.",
            })
            continue
        tensor = (
            d.set_index(["hmm_id", "dataset_replicate", "condition", "seed"])["z"]
            .reindex(pd.MultiIndex.from_product([processes, reps, conditions, seeds]))
            .to_numpy(float)
            .reshape(P, R, C, S)
        )
        grand = tensor.mean()
        mean_p = tensor.mean(axis=(1, 2, 3))
        mean_c = tensor.mean(axis=(0, 1, 3))
        mean_pr = tensor.mean(axis=(2, 3))
        mean_pc = tensor.mean(axis=(1, 3))
        mean_prc = tensor.mean(axis=3)

        effect_p = mean_p - grand
        effect_c = mean_c - grand
        effect_d = mean_pr - mean_p[:, None]
        effect_pc = mean_pc - mean_p[:, None] - mean_c[None, :] + grand
        effect_dc = mean_prc - mean_pr[:, :, None] - mean_pc[:, None, :] + mean_p[:, None, None]
        residual = tensor - mean_prc[:, :, :, None]

        ss_p = R * C * S * np.sum(effect_p ** 2)
        ss_c = P * R * S * np.sum(effect_c ** 2)
        ss_d = C * S * np.sum(effect_d ** 2)
        ss_pc = R * S * np.sum(effect_pc ** 2)
        ss_dc = S * np.sum(effect_dc ** 2)
        ss_e = np.sum(residual ** 2)

        df_p = P - 1
        df_c = C - 1
        df_d = P * (R - 1)
        df_pc = (P - 1) * (C - 1)
        df_dc = P * (R - 1) * (C - 1)
        df_e = P * R * C * (S - 1)

        ms_p = ss_p / df_p
        ms_c = ss_c / df_c
        ms_d = ss_d / df_d
        ms_pc = ss_pc / df_pc
        ms_dc = ss_dc / df_dc
        ms_e = ss_e / df_e

        sigma_e = max(ms_e, 0.0)
        sigma_dc = max((ms_dc - ms_e) / S, 0.0)
        sigma_d = max((ms_d - ms_dc) / (C * S), 0.0)
        sigma_pc = max((ms_pc - ms_dc) / (R * S), 0.0)
        sigma_p = max((ms_p - ms_d - ms_pc + ms_dc) / (R * C * S), 0.0)
        latent_total = sigma_p + sigma_d + sigma_pc + sigma_dc
        rows.append({
            "outcome": outcome,
            "P": P, "R": R, "C": C, "S": S,
            "MS_process": ms_p,
            "MS_condition": ms_c,
            "MS_dataset_within_process": ms_d,
            "MS_process_by_condition": ms_pc,
            "MS_dataset_by_condition": ms_dc,
            "MS_seed_residual": ms_e,
            "process_variance": sigma_p,
            "dataset_within_process_variance": sigma_d,
            "process_by_condition_variance": sigma_pc,
            "dataset_by_condition_variance": sigma_dc,
            "seed_residual_variance": sigma_e,
            "stable_process_share_vs_dataset": (
                sigma_p / (sigma_p + sigma_d) if sigma_p + sigma_d > 0 else np.nan
            ),
            "dataset_latent_share": (
                (sigma_d + sigma_dc) / latent_total if latent_total > 0 else np.nan
            ),
            "process_latent_share": (
                (sigma_p + sigma_pc) / latent_total if latent_total > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def exact_sign_flip_paired(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan
    observed = abs(values.mean())
    if n <= 20:
        count = 0
        total = 1 << n
        for mask in range(total):
            signs = np.array([1.0 if (mask >> j) & 1 else -1.0 for j in range(n)])
            if abs(np.mean(signs * values)) >= observed - 1e-15:
                count += 1
        return count / total
    rng = np.random.default_rng(2026080410)
    repeats = 100000
    signs = rng.choice([-1.0, 1.0], size=(repeats, n))
    return float((1 + np.sum(np.abs((signs * values).mean(axis=1)) >= observed)) / (repeats + 1))


def frozen_prediction_performance(
    hmm_summary: pd.DataFrame,
    files: Dict[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frozen = pd.read_csv(files["frozen_predictions"])
    frozen = frozen[["hmm_id", "architecture", "outcome", "model", "predicted"]].drop_duplicates()
    rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    for rep in sorted(hmm_summary["dataset_replicate"].unique()):
        observed_rep = hmm_summary[hmm_summary["dataset_replicate"] == rep]
        for arch in observed_rep["architecture"].unique():
            for outcome in ("A_width_shape", "mean_excess_ce_H"):
                obs = observed_rep[observed_rep["architecture"] == arch][
                    ["hmm_id", outcome]
                ].rename(columns={outcome: "observed"})
                pred = frozen[
                    (frozen["architecture"] == arch) & (frozen["outcome"] == outcome)
                ]
                merged = pred.merge(obs, on="hmm_id", how="inner")
                for model, group in merged.groupby("model"):
                    rmse = float(np.sqrt(mean_squared_error(group["observed"], group["predicted"])))
                    r2 = float(r2_score(group["observed"], group["predicted"]))
                    rows.append({
                        "dataset_replicate": int(rep),
                        "architecture": arch,
                        "outcome": outcome,
                        "model": model,
                        "n_hmms": int(len(group)),
                        "rmse": rmse,
                        "r2": r2,
                    })
                wide = merged.pivot(index="hmm_id", columns="model", values=["observed", "predicted"])
                # observed is duplicated by model; reconstruct directly.
                k = merged[merged["model"] == "K_only"].set_index("hmm_id")
                a = merged[merged["model"] == "augmented_profile"].set_index("hmm_id")
                common = k.index.intersection(a.index)
                if len(common):
                    y = k.loc[common, "observed"].to_numpy(float)
                    se_k = (y - k.loc[common, "predicted"].to_numpy(float)) ** 2
                    se_a = (y - a.loc[common, "predicted"].to_numpy(float)) ** 2
                    diff = se_k - se_a
                    rmse_k = float(np.sqrt(se_k.mean()))
                    rmse_a = float(np.sqrt(se_a.mean()))
                    paired_rows.append({
                        "dataset_replicate": int(rep),
                        "architecture": arch,
                        "outcome": outcome,
                        "n_hmms": int(len(common)),
                        "K_only_rmse": rmse_k,
                        "augmented_rmse": rmse_a,
                        "augmented_rmse_reduction_percent": 100 * (rmse_k - rmse_a) / rmse_k,
                        "mean_squared_error_advantage": float(diff.mean()),
                        "exact_two_sided_sign_flip_p": exact_sign_flip_paired(diff),
                    })
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def pca_stability(width_summary: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_frames: List[pd.DataFrame] = []
    loading_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for rep, group in width_summary.groupby("dataset_replicate"):
        group = group.copy()
        group["condition"] = group["architecture"] + "_d" + group["dimension"].astype(str)
        matrix = group.pivot(index="hmm_id", columns="condition", values="shape_rmse").dropna()
        X = matrix.to_numpy(float)
        X = (X - X.mean(axis=0)) / np.where(X.std(axis=0, ddof=0) > 1e-12, X.std(axis=0, ddof=0), 1.0)
        pca = PCA(n_components=min(2, X.shape[1], X.shape[0]))
        scores = pca.fit_transform(X)
        loadings = pca.components_.copy()
        if loadings[0].mean() < 0:
            loadings[0] *= -1
            scores[:, 0] *= -1
        summary_rows.append({
            "dataset_replicate": int(rep),
            "n_hmms": int(X.shape[0]),
            "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
            "pc1_pc2_explained_variance": float(pca.explained_variance_ratio_[:2].sum()),
        })
        for condition, loading in zip(matrix.columns, loadings[0]):
            loading_rows.append({
                "dataset_replicate": int(rep),
                "condition": condition,
                "pc1_loading": float(loading),
            })
        score_frames.append(pd.DataFrame({
            "hmm_id": matrix.index,
            "dataset_replicate": int(rep),
            "pc1_score": scores[:, 0],
        }))
    scores_df = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    pair_rows: List[Dict[str, Any]] = []
    if not scores_df.empty:
        pivot = scores_df.pivot(index="hmm_id", columns="dataset_replicate", values="pc1_score").dropna()
        for r1, r2 in combinations(pivot.columns, 2):
            pair_rows.append({
                "replicate_1": int(r1),
                "replicate_2": int(r2),
                "n_hmms": int(len(pivot)),
                "pc1_score_spearman": float(spearmanr(pivot[r1], pivot[r2]).statistic),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(loading_rows), pd.DataFrame(pair_rows)


# =============================================================================
# 9. Analysis outputs and figures
# =============================================================================

def save_analysis(
    run_metrics: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: Config,
    files: Dict[str, Path],
    paths: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    audit = grid_audit(run_metrics, selected, cfg)
    atomic_write_csv(paths["tables"] / "grid_audit.csv", audit)
    if not audit["observed"].all():
        missing = int((~audit["observed"]).sum())
        raise RuntimeError(f"Analysis requires the complete balanced grid; {missing} runs are missing")

    atomic_write_csv(paths["tables"] / "run_level_with_metrics.csv", run_metrics)
    width, hmm = build_summaries(run_metrics, selected)
    atomic_write_csv(paths["tables"] / "width_level_summary.csv", width)
    atomic_write_csv(paths["tables"] / "hmm_dataset_summary.csv", hmm)

    ranks = pairwise_rank_stability(hmm, cfg)
    icc = hmm_level_icc(hmm, cfg)
    variance = full_variance_components(run_metrics, cfg)
    frozen_perf, frozen_paired = frozen_prediction_performance(hmm, files)
    pca_summary, pca_loadings, pca_pairs = pca_stability(width, cfg)

    atomic_write_csv(paths["tables"] / "dataset_pair_rank_stability.csv", ranks)
    atomic_write_csv(paths["tables"] / "hmm_level_process_icc.csv", icc)
    atomic_write_csv(paths["tables"] / "full_variance_components.csv", variance)
    atomic_write_csv(paths["tables"] / "frozen_profile_performance_by_dataset.csv", frozen_perf)
    atomic_write_csv(paths["tables"] / "frozen_augmented_vs_K_by_dataset.csv", frozen_paired)
    atomic_write_csv(paths["tables"] / "pca_summary_by_dataset.csv", pca_summary)
    atomic_write_csv(paths["tables"] / "pca_pc1_loadings_by_dataset.csv", pca_loadings)
    atomic_write_csv(paths["tables"] / "pca_pc1_score_stability.csv", pca_pairs)

    # Figures: one figure per file, no custom color specification.
    for arch in cfg.ARCHITECTURES:
        for outcome in ("A_width_shape", "mean_excess_ce_H"):
            pivot = hmm[hmm["architecture"] == arch].pivot(
                index="hmm_id", columns="dataset_replicate", values=outcome
            ).dropna()
            for r1, r2 in combinations(pivot.columns, 2):
                plt.figure(figsize=(5.5, 5.0))
                plt.scatter(pivot[r1], pivot[r2], alpha=0.8)
                low = float(min(pivot[r1].min(), pivot[r2].min()))
                high = float(max(pivot[r1].max(), pivot[r2].max()))
                plt.plot([low, high], [low, high], linestyle="--")
                rho = spearmanr(pivot[r1], pivot[r2]).statistic
                plt.xlabel(f"Dataset replicate {r1}")
                plt.ylabel(f"Dataset replicate {r2}")
                plt.title(f"{arch}: {outcome} (Spearman={rho:.3f})")
                plt.tight_layout()
                plt.savefig(
                    paths["figures"] / f"rank_stability_{arch}_{outcome}_r{r1}_r{r2}.png",
                    dpi=220,
                )
                plt.close()

    if not frozen_perf.empty:
        for (arch, outcome), group in frozen_perf.groupby(["architecture", "outcome"]):
            pivot = group.pivot(index="dataset_replicate", columns="model", values="rmse")
            pivot.plot(kind="bar", figsize=(7.0, 4.5))
            plt.ylabel("RMSE")
            plt.title(f"Frozen prediction across dataset realizations: {arch}, {outcome}")
            plt.tight_layout()
            plt.savefig(paths["figures"] / f"frozen_rmse_{arch}_{outcome}.png", dpi=220)
            plt.close()

    # Human-readable compact summary.
    lines = [
        "# Dataset-realization robustness summary",
        "",
        f"Selected HMMs: {selected['hmm_id'].nunique()}",
        f"Dataset realizations: {sorted(run_metrics['dataset_replicate'].unique().tolist())}",
        f"Completed run-level conditions: {len(run_metrics)}",
        "",
        "## HMM-level process ICC",
        icc.to_markdown(index=False) if not icc.empty else "No ICC result.",
        "",
        "## Dataset-pair rank stability",
        ranks.to_markdown(index=False) if not ranks.empty else "No rank result.",
        "",
        "## Frozen augmented profile versus K-only",
        frozen_paired.to_markdown(index=False) if not frozen_paired.empty else "No frozen result.",
        "",
        "## Full variance components",
        variance.to_markdown(index=False) if not variance.empty else "No variance result.",
    ]
    (paths["analysis"] / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    atomic_write_json(
        paths["root"] / "PIPELINE_COMPLETE.json",
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "selected_hmms": int(selected["hmm_id"].nunique()),
            "dataset_replicates": sorted(run_metrics["dataset_replicate"].unique().tolist()),
            "run_level_conditions": int(len(run_metrics)),
            "tables_dir": str(paths["tables"]),
            "figures_dir": str(paths["figures"]),
        },
    )
    logger.info("Analysis completed. Results: %s", paths["analysis"])


# =============================================================================
# 10. Protocol, status, main entry point
# =============================================================================

def write_protocol(cfg: Config, paths: Dict[str, Path], files: Dict[str, Path]) -> None:
    all_reps = [cfg.REFERENCE_REPLICATE] + list(cfg.NEW_DATASET_REPLICATES)
    expected_new = (
        len(cfg.NEW_DATASET_REPLICATES) * len(cfg.K_VALUES) * cfg.HMMS_PER_K
        * len(cfg.ARCHITECTURES) * len(cfg.DIMENSIONS) * len(cfg.TRAINING_SEEDS)
    )
    protocol = {
        "study": "dataset_realization_robustness",
        "purpose": (
            "Separate HMM/process differences from finite train-validation-test dataset "
            "realization variability."
        ),
        "confirmatory_root_read_only": str(files["root"]),
        "new_project_root": cfg.PROJECT_ROOT,
        "outcome_blind_subset_selection": True,
        "subset_selection": {
            "K_values": list(cfg.K_VALUES),
            "HMMs_per_K": cfg.HMMS_PER_K,
            "features": list(TRANSFORMED_PROFILE_COLUMNS),
            "method": "within-K k-means, nearest observed HMM to each centroid",
        },
        "dataset_replicates": all_reps,
        "reference_replicate": {
            "id": cfg.REFERENCE_REPLICATE,
            "dataset": "existing cached confirmatory train/validation/test split",
            "runs": "existing seeds 1-3 reused when available",
        },
        "new_dataset_replicates": {
            "ids": list(cfg.NEW_DATASET_REPLICATES),
            "independent_train_validation_test_generation": True,
        },
        "neural_grid": {
            "architectures": list(cfg.ARCHITECTURES),
            "dimensions": list(cfg.DIMENSIONS),
            "training_seeds": list(cfg.TRAINING_SEEDS),
            "n_train": cfg.N_TRAIN,
            "n_val": cfg.N_VAL,
            "n_test": cfg.N_TEST,
            "sequence_length": cfg.SEQUENCE_LENGTH,
            "expected_new_runs": expected_new,
        },
        "analysis": {
            "dataset_pair_Spearman": True,
            "HMM_level_process_ICC": True,
            "full_nested_variance_components": True,
            "frozen_development_profile_prediction": True,
            "PCA_stability": True,
            "bootstrap_repeats": cfg.BOOTSTRAP_REPEATS,
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    protocol_path = paths["config"] / "DATASET_REPLICATION_PROTOCOL.json"
    if protocol_path.exists():
        old = json.load(protocol_path.open("r", encoding="utf-8"))
        # Protect design after training starts.
        old_design = {k: old.get(k) for k in (
            "subset_selection", "dataset_replicates", "new_dataset_replicates", "neural_grid"
        )}
        new_design = {k: protocol.get(k) for k in old_design}
        if old_design != new_design and runs_or_checkpoints_exist(paths):
            raise RuntimeError(
                "Protocol-defining settings changed after training began. Use a new PROJECT_ROOT."
            )
    atomic_write_json(protocol_path, protocol)


def main(cfg: Config = CFG) -> None:
    mount_drive_if_needed()
    if not in_colab() and cfg.CONFIRMATORY_ROOT.startswith("/content/drive/"):
        raise RuntimeError(
            "This notebook is configured for Google Colab. Set CONFIRMATORY_ROOT and "
            "PROJECT_ROOT to local paths for a local run."
        )
    paths = paths_for(cfg)
    logger = setup_logger(paths["logs"])
    files = confirmatory_files(cfg)
    device = resolve_device(cfg)
    set_global_seed(cfg.MASTER_SEED, cfg.DETERMINISTIC_TORCH)
    write_protocol(cfg, paths, files)
    atomic_write_json(
        paths["config"] / "RUN_ENVIRONMENT.json",
        {
            "config": asdict(cfg),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    logger.info("New project root: %s", paths["root"])
    logger.info("Existing confirmatory root (read only): %s", files["root"])
    logger.info("Device: %s", device)
    logger.info("Re-run the same cell after a disconnect; completed work is skipped.")

    all_metrics = load_confirmatory_metrics(cfg, files)
    selected = select_and_lock_subset(all_metrics, cfg, paths, logger)
    reference = load_existing_reference_results(selected, cfg, files, logger)
    validate_reference_replicate_availability(selected, reference, cfg, logger)
    datasets = prepare_dataset_replicates(selected, cfg, paths, logger)

    expected_new = (
        len(cfg.NEW_DATASET_REPLICATES) * selected["hmm_id"].nunique()
        * len(cfg.ARCHITECTURES) * len(cfg.DIMENSIONS) * len(cfg.TRAINING_SEEDS)
    )
    expected_total = (
        (1 + len(cfg.NEW_DATASET_REPLICATES)) * selected["hmm_id"].nunique()
        * len(cfg.ARCHITECTURES) * len(cfg.DIMENSIONS) * len(cfg.TRAINING_SEEDS)
    )
    logger.info(
        "Design: %d HMMs, %d dataset replicates, %d total conditions; %d new runs",
        selected["hmm_id"].nunique(), 1 + len(cfg.NEW_DATASET_REPLICATES),
        expected_total, expected_new,
    )

    if cfg.PREPARE_ONLY:
        atomic_write_json(paths["root"] / "PREPARATION_COMPLETE.json", {
            "status": "prepared",
            "selected_hmms": int(selected["hmm_id"].nunique()),
            "new_datasets": int(len(cfg.NEW_DATASET_REPLICATES) * selected["hmm_id"].nunique()),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        logger.info("PREPARE_ONLY=True; stopping before training")
        return

    if cfg.RUN_TRAINING and not cfg.ANALYZE_ONLY:
        available, expected = train_grid(
            selected, datasets, reference, cfg, paths, device, logger
        )
        logger.info("Available conditions after this execution: %d/%d", available, expected)

    combined = combined_run_results(selected, reference, cfg, paths)
    run_metrics = attach_recovery_metrics(combined, selected, cfg) if not combined.empty else combined
    audit = grid_audit(run_metrics, selected, cfg) if not run_metrics.empty else expected_grid_dataframe(selected, cfg).assign(observed=False)
    atomic_write_csv(paths["tables"] / "grid_audit.csv", audit)
    missing = int((~audit["observed"]).sum())
    atomic_write_json(paths["root"] / "RUN_STATUS.json", {
        "status": "complete_grid" if missing == 0 else "training_incomplete",
        "available_conditions": int(audit["observed"].sum()),
        "expected_conditions": int(len(audit)),
        "missing_conditions": missing,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    if missing > 0:
        logger.info(
            "Training is incomplete: %d conditions remain. Re-run the same cell to resume.",
            missing,
        )
        return

    if cfg.RUN_ANALYSIS:
        save_analysis(run_metrics, selected, cfg, files, paths, logger)


if __name__ == "__main__":
    main()
