import numpy as np
from sklearn.cluster import DBSCAN

def apply_dbscan_clustering(features, eps=0.5, min_samples=5):
    """
    Applies DBSCAN clustering to identify Safe (Cluster) and Rare (Noise) regions.
    
    Args:
        features (np.array): The feature matrix (typically the cleaned majority class).
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns:
        labels (np.array): Cluster labels for each point. -1 indicates noise (Rare).
        core_samples_mask (np.array): A boolean mask indicating core samples.
    """
    if len(features) == 0:
        return np.array([]), np.array([])

    # Apply DBSCAN clustering to separate safe (cluster) regions from rare (noise) regions
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(features)
    labels = db.labels_
    
    # Create boolean mask for core samples (points that have at least min_samples neighbors)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    return labels, core_samples_mask