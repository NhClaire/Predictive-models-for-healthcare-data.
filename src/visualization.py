from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, learning_curve

from .classification import _configure_xgb_for_target, get_model_grids


def plot_class_distribution(y, title="Class distribution"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=pd.Series(y).astype(str), ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    return fig, ax


def plot_correlation_heatmap(corr: pd.DataFrame, title="Correlation heatmap"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax)
    ax.set_title(title)
    return fig, ax


def _sample_for_embedding(X, y, max_samples: int, random_state: int):
    if len(X) <= max_samples:
        return X, y
    idx = pd.Series(y).groupby(y).sample(frac=min(1, max_samples / len(X)), random_state=random_state).index
    return X.loc[idx], pd.Series(y).loc[idx]


def plot_tsne(X, y, title="t-SNE 2D", max_samples=2000, random_state=42):
    X_s, y_s = _sample_for_embedding(X, y, max_samples, random_state)
    perplexity = min(30, max(5, len(X_s) // 4))
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state).fit_transform(X_s)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=pd.Series(y_s).astype(str), s=25, ax=ax)
    ax.set_title(title)
    return fig, ax


def plot_umap(X, y, title="UMAP 2D", max_samples=3000, random_state=42):
    import umap

    X_s, y_s = _sample_for_embedding(X, y, max_samples, random_state)
    emb = umap.UMAP(n_components=2, random_state=random_state).fit_transform(X_s)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=pd.Series(y_s).astype(str), s=25, ax=ax)
    ax.set_title(title)
    return fig, ax


def plot_roc_curves(results, X_test_lookup=None, y_test=None, title="ROC curves"):
    fig, ax = plt.subplots(figsize=(7, 6))
    for result in results:
        estimator = result["best_estimator"]
        X_eval = X_test_lookup[result["data_version"]] if X_test_lookup else result.get("X_test")
        if X_eval is None:
            continue
        RocCurveDisplay.from_estimator(estimator, X_eval, y_test, name=result["experiment"], ax=ax)
    ax.set_title(title)
    return fig, ax


def plot_model_comparison(results_df: pd.DataFrame, metric="test_f1", title="Model comparison"):
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=results_df, x="model", y=metric, hue="data_version", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, min(1.0, max(0.1, results_df[metric].max() * 1.15)))
    return fig, ax


def plot_feature_importance(model, X, y=None, top_n=20, title="Feature importance", random_state=42):
    inner_model = model.named_steps["model"] if hasattr(model, "named_steps") and "model" in model.named_steps else model
    if hasattr(inner_model, "feature_importances_"):
        importances = inner_model.feature_importances_
    elif hasattr(inner_model, "coef_"):
        importances = np.abs(inner_model.coef_).mean(axis=0)
    else:
        if y is None:
            raise ValueError("y is required for permutation importance.")
        importances = permutation_importance(model, X, y, n_repeats=5, random_state=random_state, n_jobs=-1).importances_mean
    df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.25)))
    sns.barplot(data=df, y="feature", x="importance", ax=ax)
    ax.set_title(title)
    return fig, ax


def plot_learning_curves(X, y, cv=None, train_sizes=np.linspace(0.2, 1.0, 5), random_state=42):
    cv = cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grids = get_model_grids(random_state)
    xgb_model = _configure_xgb_for_target(grids["XGBoost"][0], y)
    models = {
        "LR": grids["Logistic Regression"][0],
        "RF": grids["Random Forest"][0].set_params(n_estimators=200, max_depth=None),
        "XGBoost": xgb_model.set_params(n_estimators=200, learning_rate=0.1, max_depth=3),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, model) in zip(axes, models.items()):
        sizes, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            scoring="f1_weighted",
            train_sizes=train_sizes,
            n_jobs=-1,
        )
        ax.plot(sizes, train_scores.mean(axis=1), marker="o", label="Train F1")
        ax.plot(sizes, val_scores.mean(axis=1), marker="o", label="Validation F1")
        ax.fill_between(
            sizes,
            val_scores.mean(axis=1) - val_scores.std(axis=1),
            val_scores.mean(axis=1) + val_scores.std(axis=1),
            alpha=0.15,
        )
        ax.set_title(name)
        ax.set_xlabel("Training samples")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Weighted F1")
    axes[-1].legend()
    fig.suptitle("Learning curves: LR vs RF vs XGBoost")
    fig.tight_layout()
    return fig, axes
