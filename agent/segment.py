"""Segmentation / clustering: find natural groups in the data via KMeans on
standardized numeric columns, with a plain-numbers summary of what
distinguishes each cluster (e.g. "Cluster 2: high income, low spending") — the
unsupervised complement to "What predicts this?" (agent.predict). Zero AI
cost — scikit-learn only.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from agent.charts import CATEGORICAL, OTHER_COLOR, SURFACE, fig_to_base64_png, new_figure

MIN_ROWS = 20
MAX_CLUSTERS = 8


def find_segments(df: pd.DataFrame, columns: list[str] | None = None, n_clusters: int = 4) -> dict:
    """Standardizes the selected numeric columns, runs KMeans, and profiles
    each cluster by which columns deviate most from the overall mean (in
    standard-deviation units) — i.e. what actually defines that group, not
    just its raw averages. Returns {"applicable": False, "reason": ...} if
    there isn't enough usable data."""
    numeric_df = (df[columns] if columns else df).select_dtypes(include="number")
    working = numeric_df.dropna()

    if working.shape[1] < 2:
        return {"applicable": False, "reason": "Need at least 2 numeric columns to find segments."}
    if len(working) < MIN_ROWS:
        return {"applicable": False, "reason": f"Need at least {MIN_ROWS} complete rows across the selected columns."}

    n_clusters = max(2, min(n_clusters, MAX_CLUSTERS, len(working) // 2))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(working)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)

    overall_mean = working.mean()
    overall_std = working.std().replace(0, 1)
    cluster_profiles = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        size = int(mask.sum())
        if size == 0:
            continue
        cluster_mean = working[mask].mean()
        z_diff = ((cluster_mean - overall_mean) / overall_std).sort_values(key=lambda s: s.abs(), ascending=False)
        top_features = [
            {
                "column": col,
                "cluster_mean": round(float(cluster_mean[col]), 4),
                "overall_mean": round(float(overall_mean[col]), 4),
                "z_diff": round(float(z_diff[col]), 3),
            }
            for col in z_diff.index[:5]
        ]
        cluster_profiles.append(
            {
                "cluster": cluster_id,
                "size": size,
                "pct_of_rows": round(100 * size / len(working), 2),
                "top_features": top_features,
            }
        )

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    return {
        "applicable": True,
        "n_clusters": n_clusters,
        "n_rows_used": len(working),
        "columns_used": list(working.columns),
        "clusters": cluster_profiles,
        "pca_x": [round(float(v), 4) for v in coords[:, 0]],
        "pca_y": [round(float(v), 4) for v in coords[:, 1]],
        "pca_labels": [int(v) for v in labels],
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
    }


def plot_segments(result: dict) -> str:
    """2D PCA projection of the rows, colored by cluster assignment. Returns base64 PNG."""
    labels = result["pca_labels"]
    xs = result["pca_x"]
    ys = result["pca_y"]

    fig, ax = new_figure(figsize=(7, 5), grid="both")
    for cluster_id in sorted(set(labels)):
        color = CATEGORICAL[cluster_id] if cluster_id < len(CATEGORICAL) else OTHER_COLOR
        idx = [i for i, label in enumerate(labels) if label == cluster_id]
        ax.scatter(
            [xs[i] for i in idx],
            [ys[i] for i in idx],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
            s=24,
            color=color,
            edgecolor=SURFACE,
            linewidth=0.3,
        )
    ax.legend(fontsize=8, frameon=False)

    variance = result["pca_explained_variance"]
    x_label = f"PC1 ({variance[0]:.0%} of variance)" if variance else "PC1"
    y_label = f"PC2 ({variance[1]:.0%} of variance)" if len(variance) > 1 else "PC2"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{result['n_clusters']} segments found")
    fig.tight_layout()
    return fig_to_base64_png(fig)
