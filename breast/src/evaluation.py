import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder, label_binarize


def compute_cluster_metrics(X_scaled: np.ndarray, y_true, cluster_labels) -> dict:
    le = LabelEncoder()
    y_enc = le.fit_transform(y_true)
    valid = cluster_labels >= 0  # DBSCAN marks noise as -1
    sil = (
        silhouette_score(X_scaled[valid], cluster_labels[valid])
        if valid.sum() > 1
        else float("nan")
    )
    ari = adjusted_rand_score(y_enc, cluster_labels)
    nmi = normalized_mutual_info_score(y_enc, cluster_labels)
    n_found = len(set(cluster_labels[valid]))
    return {"silhouette": sil, "ari": ari, "nmi": nmi, "n_clusters_found": n_found}


def plot_clusters(X_2d: np.ndarray, labels, title: str, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = list(dict.fromkeys(labels))
    palette = sns.color_palette("tab10", len(unique_labels))
    for lbl, color in zip(unique_labels, palette):
        mask = np.array(labels) == lbl
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=str(lbl), color=color, s=40, alpha=0.8, edgecolors="white", linewidths=0.3,
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(title="Label", fontsize=8, markerscale=1.2)
    ax.grid(True, linestyle="--", alpha=0.4)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_roc_auc_multiclass(y_true, y_prob: np.ndarray, class_names):
    n_classes = len(class_names)
    Y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(Y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def compare_models_table(results_dict: dict) -> pd.DataFrame:
    rows = []
    for name, res in results_dict.items():
        rows.append({
            "Model": name,
            "F1 (macro)": round(res["f1"], 4),
            "AUC (macro)": round(res["auc"], 4),
            "Fit Time (s)": round(res["fit_time"], 2),
        })
    df = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    return df

def plot_crosstab(y_true, cluster_labels, class_names, ax=None):
    le = LabelEncoder()
    le.fit(class_names)
    y_enc = le.transform(y_true)

    ct = pd.crosstab(
        pd.Series(y_enc, name="True Label"),
        pd.Series(cluster_labels, name="Cluster"),
    )
    ct.index = [le.classes_[i] for i in ct.index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, ct.shape[1] + 2), max(4, ct.shape[0])))

    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("True Label vs Cluster Assignment", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("True Label")

    if ax is None:
        plt.tight_layout()
        plt.show()
