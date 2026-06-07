import numpy as np
from sklearn.cluster import DBSCAN

SAFE       = 0  # core points  → dense cluster interior, apply RUS
BORDERLINE = 1  # border points → cluster edge, apply RUS
RARE       = 2  # noise with at least one neighbor within eps → protected
OUTLIER    = 3  # fully isolated noise → protected


def apply_dbscan_clustering(features, eps=0.5, min_samples=5):
    """
    Categorises majority-class points into SAFE / BORDERLINE / RARE / OUTLIER
    as defined in Peng & Park (2022).
    Returns (categories, core_samples_mask).
    """
    if len(features) == 0:
        return np.array([]), np.array([])

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = db.labels_

    core_mask = np.zeros(len(features), dtype=bool)
    core_mask[db.core_sample_indices_] = True

    categories = np.full(len(features), OUTLIER, dtype=int)

    for i in range(len(features)):
        if labels[i] != -1:
            categories[i] = SAFE if core_mask[i] else BORDERLINE
        else:
            dists = np.linalg.norm(features - features[i], axis=1)
            dists[i] = np.inf
            if np.any(dists <= eps):
                categories[i] = RARE

    return categories, core_mask
