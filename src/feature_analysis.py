from __future__ import annotations

import pandas as pd


def class_distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts(dropna=False)
    return pd.DataFrame({"count": counts, "percent": counts / len(y) * 100})


def missing_values(X: pd.DataFrame) -> pd.DataFrame:
    missing = X.isna().sum()
    return pd.DataFrame({"missing_count": missing, "missing_percent": missing / len(X) * 100})


def feature_distributions(X: pd.DataFrame) -> pd.DataFrame:
    return X.describe().T


def correlation_matrix(X: pd.DataFrame, max_features: int = 50) -> pd.DataFrame:
    variances = X.var(numeric_only=True).sort_values(ascending=False)
    cols = variances.head(min(max_features, len(variances))).index
    return X[cols].corr()


def class_imbalance_ratio(y: pd.Series) -> float:
    counts = y.value_counts()
    if counts.empty or counts.min() == 0:
        return float("inf")
    return float(counts.max() / counts.min())


def eda_summary(X: pd.DataFrame, y: pd.Series) -> dict:
    return {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "class_distribution": class_distribution(y),
        "missing_values": missing_values(X),
        "feature_distributions": feature_distributions(X),
        "correlation_matrix": correlation_matrix(X),
        "imbalance_ratio": class_imbalance_ratio(y),
    }

