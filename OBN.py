import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_obn_scores(X, k_neighbors=5):
    """
    Computes Outlier Based on Noise (OBN) scores.
    The score is defined as the mean distance to the k-nearest neighbors.
    Higher score means the point is likely an outlier.
    """
    X = np.asarray(X)
    # Compute distances to k-nearest neighbors for each point
    nn = NearestNeighbors(n_neighbors=k_neighbors+1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    
    # Exclude the point itself (distance at index 0 is always 0)
    distances = distances[:, 1:]
    
    # OBN score: mean distance to k neighbors (higher = more likely outlier)
    obn_scores = np.mean(distances, axis=1)
    return obn_scores

def apply_obn(X, y, k_neighbors=5, multiplier=1.8, verbose=False):
    """
    Detects and removes outliers from the Majority class using OBN scores.
    Threshold = Mean + (Multiplier * Std_Dev).
    
    Args:
        X: Feature matrix
        y: Labels
        k_neighbors: Number of neighbors for distance calculation
        multiplier: Sensitivity for threshold (typically 1.5 - 3.0)
        verbose: Print progress logs
        
    Returns:
        X_clean, y_clean: Dataset with outliers removed.
    """
    if verbose:
        print("OBN Algorithm Started...")

    # Apply OBN only to majority class samples (label 0)
    majority_class = 0 
    maj_indices = np.where(y == majority_class)[0]
    X_maj = X[maj_indices]
    
    # Safety check: need at least k+1 samples to compute k-nearest neighbors
    if len(X_maj) < k_neighbors + 1:
        return X, y
        
    # Compute OBN scores for majority class samples
    scores = compute_obn_scores(X_maj, k_neighbors)
    
    # Calculate threshold using mean + (multiplier * standard deviation)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + (multiplier * std_score)
    
    # Identify outliers: samples with OBN score above threshold
    outlier_mask = scores > threshold
    outlier_indices = maj_indices[outlier_mask]
    
    if verbose:
        print(f"  OBN Stats -> Mean: {mean_score:.4f}, Std: {std_score:.4f}, Threshold: {threshold:.4f}")
        print(f"  Outliers Removed: {len(outlier_indices)}")
        
    # Create boolean mask to keep non-outlier samples
    keep_mask = np.ones(len(X), dtype=bool)
    keep_mask[outlier_indices] = False
    
    return X[keep_mask], y[keep_mask]