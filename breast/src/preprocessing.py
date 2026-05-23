import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


def preprocess(X, config: dict, y=None, apply_selection: bool = True) -> tuple:
    """Remove zero-variance features, optionally select top-k by F-score, then scale.

    If apply_selection=False, SelectKBest is skipped entirely and all features
    after VarianceThreshold are returned.

    Returns (X_scaled: ndarray, feature_names: list).
    """
    n_before = X.shape[1]

    vt = VarianceThreshold(threshold=0)
    X_vt = vt.fit_transform(X)
    feature_names = [X.columns[i] for i in vt.get_support(indices=True)]

    n_select = config.get("n_select_features", None)
    if apply_selection and n_select is not None and y is not None:
        k = min(n_select, X_vt.shape[1])
        selector = SelectKBest(f_classif, k=k)
        X_vt = selector.fit_transform(X_vt, y)
        selected_idx = selector.get_support(indices=True)
        feature_names = [feature_names[i] for i in selected_idx]

    n_after = X_vt.shape[1]
    print(f"Feature selection: {n_before} → {n_after} features "
          f"({n_before - n_after} removed)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vt)
    return X_scaled, feature_names
