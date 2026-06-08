from __future__ import annotations

from sklearn.model_selection import StratifiedKFold, train_test_split


def make_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def make_cv(k=5, random_state=42):
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
