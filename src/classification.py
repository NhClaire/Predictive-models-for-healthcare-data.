from __future__ import annotations

from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

from .evaluation import compute_metrics


def get_model_grids(random_state: int = 42) -> dict:
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
            {"C": [0.1, 1, 10], "solver": ["lbfgs"]},
        ),
        "SVM RBF": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
            {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.001, 0.01]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1),
            {"n_estimators": [100, 200, 500], "max_depth": [None, 10, 20]},
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
            {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 6]},
        ),
        "MLP": (
            MLPClassifier(max_iter=600, early_stopping=True, random_state=random_state),
            {
                "hidden_layer_sizes": [(64,), (128,), (64, 64)],
                "learning_rate_init": [0.001, 0.01],
            },
        ),
    }


def _xgb_scale_pos_weight(y):
    counts = y.value_counts() if hasattr(y, "value_counts") else None
    if counts is None or len(counts) != 2 or counts.min() == 0:
        return 1
    return float(counts.max() / counts.min())


def _configure_xgb_for_target(model, y):
    n_classes = len(set(y))
    if n_classes > 2:
        model.set_params(objective="multi:softprob", eval_metric="mlogloss", num_class=n_classes)
    else:
        model.set_params(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=_xgb_scale_pos_weight(y),
        )
    return model


def tune_model(
    model,
    param_grid: dict,
    X_train,
    y_train,
    cv: StratifiedKFold,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
    sampler=None,
    fit_params: dict | None = None,
):
    estimator = model
    search_grid = param_grid
    if sampler is not None:
        estimator = ImbPipeline([("sampler", sampler), ("model", model)])
        search_grid = {f"model__{key}": value for key, value in param_grid.items()}
    search = GridSearchCV(
        estimator=estimator,
        param_grid=search_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    return search.fit(X_train, y_train, **(fit_params or {}))


def train_models_for_versions(
    data_versions: dict,
    X_test,
    y_test,
    cv: Optional[StratifiedKFold] = None,
    random_state: int = 42,
    scoring: str = "f1_weighted",
    experiment_prefix: str = "",
) -> list[dict]:
    cv = cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = []
    for version_name, version in data_versions.items():
        X_train = version["X_train"]
        y_train = version["y_train"]
        X_eval = version.get("X_test", X_test)
        sampler = version.get("sampler")
        grids = get_model_grids(random_state)
        if "XGBoost" in grids:
            xgb, params = grids["XGBoost"]
            xgb = _configure_xgb_for_target(xgb, y_train)
            grids["XGBoost"] = (xgb, params)

        for model_name, (model, param_grid) in grids.items():
            fit_params = None
            if model_name == "XGBoost" and len(set(y_train)) > 2:
                weights = compute_sample_weight("balanced", y_train)
                key = "model__sample_weight" if sampler is not None else "sample_weight"
                fit_params = {key: weights}
            search = tune_model(model, param_grid, X_train, y_train, cv, scoring=scoring, sampler=sampler, fit_params=fit_params)
            metrics = compute_metrics(search.best_estimator_, X_eval, y_test)
            results.append(
                {
                    "experiment": f"{experiment_prefix}{version_name} | {model_name}",
                    "data_version": version_name,
                    "model_name": model_name,
                    "best_estimator": search.best_estimator_,
                    "best_params": search.best_params_,
                    "best_cv_score": search.best_score_,
                    "metrics": metrics,
                    "cv_results": search.cv_results_,
                }
            )
    return results
