"""
Latent cluster-separation metrics.

Quantify how well pooled latent embeddings separate by interaction law - the
numeric companion to the t-SNE / UMAP visualisations. Pure numpy (no sklearn).
"""

from __future__ import annotations

import numpy as np


def silhouette_score(latents, labels) -> float:
    """
    Mean silhouette coefficient over all samples.

    latents : (M, D) embeddings.
    labels  : (M,) integer cluster/class ids.

    Returns a value in [-1, 1]; higher means tighter, better-separated clusters.
    Falls back to 0.0 when there are fewer than two populated clusters.
    """
    X = np.asarray(latents, dtype=np.float64)
    y = np.asarray(labels)
    M = X.shape[0]
    unique = np.unique(y)
    if M < 2 or unique.size < 2:
        return 0.0

    # Pairwise Euclidean distances.
    sq = (X ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    dist = np.sqrt(np.clip(d2, 0.0, None))

    sil = np.zeros(M)
    for i in range(M):
        same = (y == y[i])
        same[i] = False
        if same.sum() == 0:
            sil[i] = 0.0
            continue
        a = dist[i, same].mean()
        b = np.inf
        for c in unique:
            if c == y[i]:
                continue
            other = (y == c)
            if other.sum() == 0:
                continue
            b = min(b, dist[i, other].mean())
        denom = max(a, b)
        sil[i] = (b - a) / denom if denom > 0 else 0.0

    return float(sil.mean())


def class_separation_ratio(latents, labels) -> float:
    """
    Ratio of between-class to within-class mean distance.

    > 1 means classes are, on average, farther apart than points within a
    class - a quick, cheap separability signal.
    """
    X = np.asarray(latents, dtype=np.float64)
    y = np.asarray(labels)
    centroids = {c: X[y == c].mean(axis=0) for c in np.unique(y)}

    within = []
    for c in np.unique(y):
        pts = X[y == c]
        if pts.shape[0] > 0:
            within.append(np.linalg.norm(pts - centroids[c], axis=1).mean())
    within_mean = float(np.mean(within)) if within else 0.0

    cs = list(centroids.values())
    between = []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            between.append(float(np.linalg.norm(cs[i] - cs[j])))
    between_mean = float(np.mean(between)) if between else 0.0

    return between_mean / within_mean if within_mean > 0 else 0.0
