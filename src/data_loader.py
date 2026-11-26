"""Dataset loading, task filtering, and cleaning for the BRMH EEG Psychiatric Disorders dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request
from typing import Literal

import numpy as np
import pandas as pd

TaskType = Literal["bipolar_vs_healthy", "bipolar_vs_depression", "bipolar_vs_all"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "EEG.machinelearing_data_BRMH.csv"
MIRROR_CSV_URL = (
    "https://raw.githubusercontent.com/jishnuvee/Psychiatric-diagnosis/"
    "main/EEG.machinelearing_data_BRMH.csv"
)

METADATA_COLS = ("no.", "eeg.date", "main.disorder", "specific.disorder")
DEMOGRAPHIC_COLS = ("sex", "age", "education", "IQ")


@dataclass
class LoadedDataset:
    """Container for cleaned features, binary target, and metadata."""

    X: pd.DataFrame
    y: pd.Series
    task: TaskType
    class_names: tuple[str, str]
    feature_names: list[str]
    raw_targets: pd.Series


def ensure_dataset_downloaded(data_path: Path = DEFAULT_DATA_PATH) -> Path:
    """Ensure the EEG CSV exists locally, downloading it from the open mirror if missing."""
    if data_path.exists():
        return data_path
    data_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MIRROR_CSV_URL, data_path)
    return data_path


def clean_feature_frame(df: pd.DataFrame, eeg_only: bool = False) -> pd.DataFrame:
    """Drop metadata/empty columns and convert sex to numeric while preserving EEG features."""
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    drop_cols = list(METADATA_COLS) + unnamed_cols
    if eeg_only:
        drop_cols.extend(DEMOGRAPHIC_COLS)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    if "sex" in X.columns:
        X["sex"] = X["sex"].map({"M": 1.0, "F": 0.0})

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def load_eeg_dataset(
    task: TaskType = "bipolar_vs_healthy",
    data_path: Path = DEFAULT_DATA_PATH,
    eeg_only: bool = False,
) -> LoadedDataset:
    """Load and filter the BRMH EEG dataset for the requested bipolar classification task."""
    csv_path = ensure_dataset_downloaded(data_path)
    df = pd.read_csv(csv_path)

    if "specific.disorder" not in df.columns:
        raise ValueError("Expected column 'specific.disorder' not found in dataset.")

    specific = df["specific.disorder"].astype(str).str.strip()

    if task == "bipolar_vs_healthy":
        mask = specific.isin(["Bipolar disorder", "Healthy control"])
        filtered = df.loc[mask].reset_index(drop=True)
        y = (filtered["specific.disorder"].str.strip() == "Bipolar disorder").astype(int)
        class_names = ("Healthy control", "Bipolar disorder")
    elif task == "bipolar_vs_depression":
        mask = specific.isin(["Bipolar disorder", "Depressive disorder"])
        filtered = df.loc[mask].reset_index(drop=True)
        y = (filtered["specific.disorder"].str.strip() == "Bipolar disorder").astype(int)
        class_names = ("Depressive disorder", "Bipolar disorder")
    elif task == "bipolar_vs_all":
        filtered = df.reset_index(drop=True)
        y = (filtered["specific.disorder"].str.strip() == "Bipolar disorder").astype(int)
        class_names = ("Other / Control", "Bipolar disorder")
    else:
        raise ValueError(f"Unsupported task: {task}")

    X = clean_feature_frame(filtered, eeg_only=eeg_only)
    raw_targets = filtered["specific.disorder"].astype(str).str.strip()

    return LoadedDataset(
        X=X,
        y=y,
        task=task,
        class_names=class_names,
        feature_names=list(X.columns),
        raw_targets=raw_targets,
    )
