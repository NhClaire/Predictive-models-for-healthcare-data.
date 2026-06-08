from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def get_positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def compute_metrics(model, X_test, y_test, average: str = "binary") -> dict:
    y_pred = model.predict(X_test)
    scores = get_positive_scores(model, X_test)
    unique_classes = np.unique(y_test)
    f1_average = average if len(unique_classes) == 2 else "weighted"
    try:
        if len(unique_classes) == 2:
            roc_auc = roc_auc_score(y_test, scores)
        else:
            roc_auc = roc_auc_score(y_test, scores, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = np.nan
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average=f1_average, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        metrics = result.get("metrics", {})
        rows.append(
            {
                "experiment": result.get("experiment"),
                "data_version": result.get("data_version"),
                "model": result.get("model_name"),
                "best_params": result.get("best_params"),
                "cv_f1": result.get("best_cv_score"),
                "test_accuracy": metrics.get("accuracy"),
                "test_f1": metrics.get("f1"),
                "test_roc_auc": metrics.get("roc_auc"),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_f1", "test_roc_auc"], ascending=False)

