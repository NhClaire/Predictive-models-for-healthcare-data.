from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import StratifiedKFold


@dataclass
class FilterSelectionResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    variance_selector: VarianceThreshold
    kbest_selector: SelectKBest
    selected_features: list[str]
    scores: pd.DataFrame


@dataclass
class WrapperSelectionResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    selector: RFECV
    selected_features: list[str]
    ranking: pd.DataFrame


def filter_select_kbest(
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    k_values: Iterable[int] = (50, 100, 200, 500),
    variance_threshold: float = 0.0,
) -> dict[int, FilterSelectionResult]:
    vt = VarianceThreshold(threshold=variance_threshold)
    X_train_v = vt.fit_transform(X_train)
    X_test_v = vt.transform(X_test)
    vt_features = np.array(X_train.columns)[vt.get_support()]
    results: dict[int, FilterSelectionResult] = {}

    for requested_k in k_values:
        k = min(int(requested_k), X_train_v.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_k = selector.fit_transform(X_train_v, y_train)
        X_test_k = selector.transform(X_test_v)
        selected = list(vt_features[selector.get_support()])
        scores = pd.DataFrame(
            {"feature": vt_features, "score": selector.scores_, "p_value": selector.pvalues_}
        ).sort_values("score", ascending=False)
        results[requested_k] = FilterSelectionResult(
            pd.DataFrame(X_train_k, columns=selected, index=X_train.index),
            pd.DataFrame(X_test_k, columns=selected, index=X_test.index),
            vt,
            selector,
            selected,
            scores,
        )
    return results


def choose_filter_result(results: dict[int, FilterSelectionResult], preferred_k: int = 200):
    if preferred_k in results:
        return results[preferred_k]
    return results[sorted(results.keys())[-1]]


def wrapper_rfecv(
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    cv: Optional[StratifiedKFold] = None,
    min_features_to_select: int = 10,
    step: int | float = 0.1,
    random_state: int = 42,
    n_jobs: int = -1,
) -> WrapperSelectionResult:
    cv = cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    estimator = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    selector = RFECV(
        estimator=estimator,
        step=step,
        min_features_to_select=min(min_features_to_select, X_train.shape[1]),
        cv=cv,
        scoring="f1_weighted",
        n_jobs=n_jobs,
    )
    X_train_w = selector.fit_transform(X_train, y_train)
    X_test_w = selector.transform(X_test)
    selected = list(np.array(X_train.columns)[selector.get_support()])
    ranking = pd.DataFrame(
        {"feature": X_train.columns, "rank": selector.ranking_, "selected": selector.support_}
    ).sort_values(["rank", "feature"])
    return WrapperSelectionResult(
        pd.DataFrame(X_train_w, columns=selected, index=X_train.index),
        pd.DataFrame(X_test_w, columns=selected, index=X_test.index),
        selector,
        selected,
        ranking,
    )

