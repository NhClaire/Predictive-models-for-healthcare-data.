import numpy as np
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture


def run_kmeans(X_scaled, n_clusters: int, config: dict):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=config.get("random_state", 42),
        n_init=10,
    )
    return model.fit_predict(X_scaled)


def run_hierarchical(X_scaled, n_clusters: int, config: dict):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(X_scaled)


def run_dbscan(X_scaled, n_clusters: int, config: dict):
    # n_clusters ignored; DBSCAN determines it automatically
    model = DBSCAN(
        eps=config.get("dbscan_eps", 5.0),
        min_samples=config.get("dbscan_min_samples", 5),
    )
    return model.fit_predict(X_scaled)


def run_gmm(X_scaled, n_clusters: int, config: dict):
    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        random_state=config.get("random_state", 42),
    )
    return model.fit_predict(X_scaled)


def run_spectral(X_scaled, n_clusters: int, config: dict):
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        random_state=config.get("random_state", 42),
        n_jobs=-1,
    )
    return model.fit_predict(X_scaled)
