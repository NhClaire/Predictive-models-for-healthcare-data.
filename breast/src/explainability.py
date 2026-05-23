import warnings
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder


def shap_analysis(model, X: np.ndarray, feature_names: list, model_type: str):
    """Compute SHAP values and plot summary bar + beeswarm + top-10 dependence.

    model_type: 'rf' | 'xgboost' | 'svm' | 'mlp' | 'logistic'
    """
    X_sample = X if len(X) <= 200 else X[:200]

    if model_type in ("rf", "xgboost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    elif model_type == "logistic":
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    else:
        # KernelExplainer: limit to 50 samples for speed
        X_bg = shap.kmeans(X_sample, min(50, len(X_sample)))
        X_explain = X_sample[:50]
        explainer = shap.KernelExplainer(model.predict_proba, X_bg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X_explain)
        X_sample = X_explain

    # Normalise to (n_samples, n_features, n_classes) regardless of SHAP version:
    #   RF TreeExplainer  → list of (n_samples, n_features)
    #   XGB TreeExplainer → (n_samples, n_features, n_classes)  [SHAP ≥ 0.40]
    #   binary / linear   → (n_samples, n_features)
    if isinstance(shap_values, list):
        sv_3d = np.stack(shap_values, axis=-1)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv_3d = shap_values
    else:
        sv_3d = shap_values[:, :, np.newaxis]

    # mean |SHAP| over samples and classes → (n_features,)
    mean_abs = np.abs(sv_3d).mean(axis=(0, 2))
    top10_idx = np.argsort(mean_abs)[::-1][:10]
    top10_names = [feature_names[i] for i in top10_idx]

    # 2-D slice for beeswarm / dependence (use first class)
    sv_plot = sv_3d[:, :, 0]

    # Summary bar
    plt.figure(figsize=(9, 5))
    plt.barh(top10_names[::-1], mean_abs[top10_idx][::-1], color="steelblue")
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"SHAP Feature Importance (Top 10) — {model_type.upper()}", fontweight="bold")
    plt.tight_layout()
    plt.show()
    shap.summary_plot(
        sv_plot[:, top10_idx],
        X_sample[:, top10_idx],
        feature_names=top10_names,
        show=True,
    )

    # Dependence plot for the single most important feature
    best_feat_idx = int(top10_idx[0])
    shap.dependence_plot(
        best_feat_idx,
        sv_plot,
        X_sample,
        feature_names=feature_names,
        show=True,
    )

    return {"shap_values": shap_values, "top10_names": top10_names, "mean_abs": mean_abs}


def plot_nonlinearity_evidence(y, X_pca: np.ndarray, X_tsne: np.ndarray,
                                X_cluster: np.ndarray):
    """Three-panel plot providing evidence for non-linearity in the data.

    Panel 1: Silhouette score (class-based) in PCA-2D vs t-SNE-2D space.
    Panel 2: CV accuracy gap between Linear SVM and RBF SVM.
    Panel 3: Logistic Regression learning curve (shows underfitting).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    sil_pca = silhouette_score(X_pca, y_enc)
    sil_tsne = silhouette_score(X_tsne, y_enc)

    # Linear vs RBF SVM (use PCA-50D feature space for accurate comparison)
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_linear = cross_val_score(
        SVC(kernel="linear", class_weight="balanced"),
        X_cluster, y_enc, cv=skf, scoring="f1_macro"   # ← X_cluster thay X_pca
    ).mean()
    acc_rbf = cross_val_score(
        SVC(kernel="rbf", class_weight="balanced", probability=False),
        X_cluster, y_enc, cv=skf, scoring="f1_macro"   # ← X_cluster thay X_pca
    ).mean()

    # Logistic Regression learning curve (on PCA-50D data)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        lr, X_cluster, y_enc, cv=skf, scoring="f1_macro",   # ← X_cluster thay X_pca
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1
    axes[0].bar(["PCA 2D", "t-SNE 2D"], [sil_pca, sil_tsne], color=["#5DADE2", "#E74C3C"], edgecolor="white")
    axes[0].set_title("Silhouette Score: PCA vs t-SNE\n(higher = better class separation)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Silhouette Score")
    for i, v in enumerate([sil_pca, sil_tsne]):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")

    # Panel 2
    axes[1].bar(["Linear SVM", "RBF SVM"], [acc_linear, acc_rbf], color=["#F39C12", "#27AE60"], edgecolor="white")
    axes[1].set_title("Linear SVM vs RBF SVM\n(macro F1 on PCA-50D, 5-fold CV)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate([acc_linear, acc_rbf]):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    # Panel 3
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    axes[2].plot(train_sizes, train_mean, "o-", color="#2980B9", label="Train F1")
    axes[2].plot(train_sizes, val_mean, "s--", color="#E74C3C", label="Val F1")
    axes[2].fill_between(train_sizes, train_scores.min(axis=1), train_scores.max(axis=1), alpha=0.15, color="#2980B9")
    axes[2].fill_between(train_sizes, val_scores.min(axis=1), val_scores.max(axis=1), alpha=0.15, color="#E74C3C")
    axes[2].set_title("Logistic Regression Learning Curve\n(underfitting = gap between train & val)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Training samples")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Evidence of Non-Linearity in Gene Expression Data", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
