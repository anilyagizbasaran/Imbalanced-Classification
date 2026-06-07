import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_obn_scores(X, k_neighbors=5):
    """
    OBN(xi) = W[Nr(xi)] / sum_{yj in Nr(xi)} W[Nr(yj)]
    W[Nr(xi)] = sum of distances from xi to its k-NN
    """
    X = np.asarray(X)
    n = len(X)
    k = min(k_neighbors, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nn.kneighbors(X)

    distances = distances[:, 1:]
    indices   = indices[:, 1:]

    W = distances.sum(axis=1)

    obn = np.zeros(n)
    for i in range(n):
        W_nbrs_sum = W[indices[i]].sum()
        if W_nbrs_sum > 0:
            obn[i] = W[i] / W_nbrs_sum
    return obn


def apply_obn(X, y, k_neighbors=5, verbose=False):
    """
    Flags majority class outliers using OBN — Peng & Park (2022) / Gupta et al. (2019).
    Threshold = mean OBN score across all majority samples.
    Returns bool mask (True = remove).
    """
    if verbose:
        print("  [OBN] started...")

    outlier_mask = np.zeros(len(X), dtype=bool)
    maj_indices  = np.where(y == 0)[0]

    if len(maj_indices) < k_neighbors + 1:
        return outlier_mask

    scores    = compute_obn_scores(X[maj_indices], k_neighbors)
    threshold = np.mean(scores)
    is_outlier = scores > threshold
    outlier_mask[maj_indices[is_outlier]] = True

    if verbose:
        print(f"  [OBN] threshold (mean): {threshold:.6f}")
        print(f"  [OBN] marked for removal: {is_outlier.sum()} / {len(maj_indices)} majority")

    return outlier_mask
