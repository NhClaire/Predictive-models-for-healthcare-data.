from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def _find_csv(root: str | Path, preferred_name: Optional[str] = None) -> Path:
    root = Path(root)
    if root.is_file():
        return root
    if preferred_name:
        match = next(root.rglob(preferred_name), None)
        if match:
            return match
    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return csv_files[0]


def load_bcge_data(path: str | Path = "breast") -> Tuple[pd.DataFrame, pd.Series]:
    csv_path = _find_csv(path, "Breast_GSE45827.csv")
    df = pd.read_csv(csv_path)
    if "type" not in df.columns:
        raise ValueError("BCGE dataset must contain a 'type' target column.")
    drop_cols = [col for col in ["samples", "type"] if col in df.columns]
    X = df.drop(columns=drop_cols).apply(pd.to_numeric, errors="coerce")
    y = df["type"].astype(str)
    return X, y


def load_diabetes_data(
    path: str | Path = "diabetes",
    filename: str = "diabetes_binary_health_indicators_BRFSS2015.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    csv_path = _find_csv(path, filename)
    df = pd.read_csv(csv_path)
    target_col = "Diabetes_binary" if "Diabetes_binary" in df.columns else df.columns[0]
    X = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
    y = df[target_col].astype(int).astype(str)
    return X, y

