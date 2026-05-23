import time
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier


def _make_pipeline(model, config: dict) -> Pipeline:
    n_select = config.get("n_select_features", None)
    steps = []
    if n_select is not None:
        steps.append(("select", SelectKBest(f_classif, k=n_select)))
    steps.append(("model", model))
    return Pipeline(steps)


def _evaluate(model, X: np.ndarray, y, config: dict) -> dict:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    skf = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=config.get("random_state", 42)
    )
    y_pred_all = np.empty(len(y_enc), dtype=int)
    y_prob_all = np.empty((len(y_enc), n_classes), dtype=float)
    _needs_sw = isinstance(model, (XGBClassifier, MLPClassifier))
    pipe = _make_pipeline(model, config)

    t0 = time.time()
    for train_idx, test_idx in skf.split(X, y_enc):
        p = clone(pipe)
        if _needs_sw:
            sw = compute_sample_weight("balanced", y_enc[train_idx])
            p.fit(X[train_idx], y_enc[train_idx], model__sample_weight=sw)
        else:
            p.fit(X[train_idx], y_enc[train_idx])
        y_pred_all[test_idx] = p.predict(X[test_idx])
        y_prob_all[test_idx] = p.predict_proba(X[test_idx])
    fit_time = time.time() - t0

    f1  = f1_score(y_enc, y_pred_all, average="macro")
    auc = roc_auc_score(y_enc, y_prob_all, multi_class="ovr", average="macro")

    # Final full-data fit — used only for SHAP / feature importance
    if _needs_sw:
        sw_full = compute_sample_weight("balanced", y_enc)
        pipe.fit(X, y_enc, model__sample_weight=sw_full)
    else:
        pipe.fit(X, y_enc)

    # Extract fitted inner model and selector (for SHAP)
    fitted_model    = pipe.named_steps["model"]
    fitted_selector = pipe.named_steps.get("select", None)

    return {
        "model":           fitted_model,
        "selector":        fitted_selector,   # ← NEW
        "f1":              float(f1),
        "auc":             float(auc),
        "fit_time":        float(fit_time),
        "y_true":          y_enc,
        "y_pred":          y_pred_all,
        "y_prob":          y_prob_all,
        "label_encoder":   le,
    }


def run_logistic(X: np.ndarray, y, config: dict) -> dict:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=config.get("random_state", 42),
    )
    return _evaluate(model, X, y, config)


def run_svm_rbf(X: np.ndarray, y, config: dict) -> dict:
    model = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=config.get("random_state", 42),
    )
    return _evaluate(model, X, y, config)


def run_random_forest(X: np.ndarray, y, config: dict) -> dict:
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=config.get("random_state", 42),
        n_jobs=-1,
    )
    return _evaluate(model, X, y, config)


def run_xgboost(X: np.ndarray, y, config: dict) -> dict:
    model = XGBClassifier(
        random_state=config.get("random_state", 42),
        eval_metric="mlogloss",
        verbosity=0,
    )
    return _evaluate(model, X, y, config)


def run_mlp(X: np.ndarray, y, config: dict) -> dict:
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        random_state=config.get("random_state", 42),
    )
    return _evaluate(model, X, y, config)
