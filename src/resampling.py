from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE


@dataclass
class ResamplingResult:
    X: pd.DataFrame
    y: pd.Series
    sampler: object | None


def baseline_resample(X_train: pd.DataFrame, y_train) -> ResamplingResult:
    return ResamplingResult(X_train.copy(), pd.Series(y_train, index=X_train.index), None)


def smote_resample(X_train: pd.DataFrame, y_train, random_state: int = 42) -> ResamplingResult:
    sampler = SMOTE(random_state=random_state)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return ResamplingResult(pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res), sampler)


def smoteenn_resample(X_train: pd.DataFrame, y_train, random_state: int = 42) -> ResamplingResult:
    sampler = SMOTEENN(random_state=random_state)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return ResamplingResult(pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res), sampler)


def make_resampled_versions(X_train: pd.DataFrame, y_train, random_state: int = 42):
    return {
        "baseline": baseline_resample(X_train, y_train),
        "SMOTE": smote_resample(X_train, y_train, random_state),
        "SMOTE+ENN": smoteenn_resample(X_train, y_train, random_state),
    }


def get_resamplers(random_state: int = 42):
    return {
        "baseline": None,
        "SMOTE": SMOTE(random_state=random_state),
        "SMOTE+ENN": SMOTEENN(random_state=random_state),
    }
