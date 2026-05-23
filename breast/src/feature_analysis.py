import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import umap as umap_lib


def run_pca(X_scaled, n_components: int = 2):
    """Fit full PCA, return 2-D projection + full cumulative variance.

    Returns (X_2d, cumvar, explained_ratio).
    """
    # Full PCA needed to return complete cumvar for variance explained plot.
    n_full = min(X_scaled.shape)
    pca = PCA(n_components=n_full)
    X_full = pca.fit_transform(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return X_full[:, :n_components], cumvar, pca.explained_variance_ratio_


def run_tsne(X_scaled, n_pca_pre: int = 50):
    """Run t-SNE with optional PCA pre-reduction when n_features > 50."""
    if X_scaled.shape[1] > n_pca_pre:
        n_pre = min(n_pca_pre, X_scaled.shape[0] - 1)
        X_pre = PCA(n_components=n_pre).fit_transform(X_scaled)
    else:
        X_pre = X_scaled
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
        max_iter=1000,
    )
    return tsne.fit_transform(X_pre)


def run_umap(X_scaled, config: dict):
    """Run UMAP with parameters from config."""
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=config.get("umap_n_neighbors", 15),
        min_dist=config.get("umap_min_dist", 0.1),
        random_state=config.get("random_state", 42),
    )
    return reducer.fit_transform(X_scaled)


def compare_linear_vs_nonlinear(
    X_scaled, y, X_pca_2d=None, X_tsne_2d=None
) -> dict:
    """Silhouette score of class labels in PCA-2D vs t-SNE-2D space.

    Accepts pre-computed projections to avoid redundant computation.
    Higher silhouette in t-SNE space is evidence for non-linearity.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if X_pca_2d is None:
        X_pca_2d = PCA(n_components=2).fit_transform(X_scaled)
    if X_tsne_2d is None:
        X_tsne_2d = run_tsne(X_scaled)

    sil_pca = silhouette_score(X_pca_2d, y_enc)
    sil_tsne = silhouette_score(X_tsne_2d, y_enc)

    return {
        "pca_silhouette": float(sil_pca),
        "tsne_silhouette": float(sil_tsne),
    }
