import sys
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from .data_loader import load_data
from .preprocessing import preprocess
from .feature_analysis import run_pca, run_tsne, run_umap, compare_linear_vs_nonlinear
from .clustering import run_kmeans, run_hierarchical, run_dbscan, run_gmm, run_spectral
from .classification import (
    run_logistic, run_svm_rbf, run_random_forest, run_xgboost, run_mlp,
)
from .evaluation import compute_cluster_metrics
from .explainability import shap_analysis
from sklearn.decomposition import PCA


def run_pipeline(config: dict) -> dict:
    """Execute full analysis pipeline and return all intermediate results.

    Order: load → preprocess → feature_analysis → clustering →
           classification → evaluation → explainability (RF + XGBoost).
    """
    print(f"[1/7] Loading data ({config['dataset']}) ...")
    X, y = load_data(config)
    print(f"      {X.shape[0]} samples, {X.shape[1]} raw features")

    print("[2/7] Preprocessing (VarianceThreshold + StandardScaler) ...")
    X_scaled, feature_names = preprocess(X, config)
    print(f"      {X_scaled.shape[1]} features after filtering")

    print("[3/7] Feature analysis (PCA, t-SNE, UMAP) ...")
    X_pca, pca_cumvar, pca_explained = run_pca(X_scaled, n_components=2)
    X_tsne = run_tsne(X_scaled)
    X_umap = run_umap(X_scaled, config)
    nonlinear_evidence = compare_linear_vs_nonlinear(
        X_scaled, y, X_pca_2d=X_pca, X_tsne_2d=X_tsne
    )
    pca = PCA(n_components=50, random_state=42)
    X_cluster = pca.fit_transform(X_scaled)
    print(f"      Silhouette — PCA: {nonlinear_evidence['pca_silhouette']:.3f}  "
          f"t-SNE: {nonlinear_evidence['tsne_silhouette']:.3f}")

    print("[4/7] Clustering ...")
    n_clusters = config["n_clusters"]
    cluster_results = {
        "kmeans":       run_kmeans(X_scaled, n_clusters, config),
        "hierarchical": run_hierarchical(X_scaled, n_clusters, config),
        "dbscan":       run_dbscan(X_scaled, n_clusters, config),
        "gmm":          run_gmm(X_cluster, n_clusters, config),
        "spectral":     run_spectral(X_scaled, n_clusters, config),
    }

    print("[5/7] Classification (5-fold StratifiedKFold) ...")
    clf_results = {}
    for name, fn in [
        ("logistic",  run_logistic),
        ("svm_rbf",   run_svm_rbf),
        ("rf",        run_random_forest),
        ("xgboost",   run_xgboost),
        ("mlp",       run_mlp),
    ]:
        print(f"      training {name} ...")
        clf_results[name] = fn(X_scaled, y, config)
        print(f"      {name}: F1={clf_results[name]['f1']:.4f}  AUC={clf_results[name]['auc']:.4f}")

    print("[6/7] Evaluation (cluster metrics) ...")
    cluster_metrics = {
        name: compute_cluster_metrics(X_scaled, y, labels)
        for name, labels in cluster_results.items()
    }

    print("[7/7] Explainability (SHAP for RF and XGBoost) ...")
    shap_results = {}
    for model_key in ("rf", "xgboost"):
        res = clf_results[model_key]
        shap_results[model_key] = shap_analysis(
            res["model"], X_scaled, feature_names, model_key
        )

    print("Pipeline complete.")
    return {
        "config": config,
        "X": X,
        "y": y,
        "X_scaled": X_scaled,
        "feature_names": feature_names,
        "X_pca": X_pca,
        "pca_cumvar": pca_cumvar,
        "pca_explained_ratio": pca_explained,
        "X_tsne": X_tsne,
        "X_umap": X_umap,
        "nonlinear_evidence": nonlinear_evidence,
        "cluster_results": cluster_results,
        "cluster_metrics": cluster_metrics,
        "clf_results": clf_results,
        "shap_results": shap_results,
    }
