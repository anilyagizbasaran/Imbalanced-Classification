import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

def apply_bnf(X, y, k=5, patience=3, verbose=False):
    """
    Applies Borderline Noise Filtering (BNF).
    It identifies 'borderline' examples in the majority class and iteratively removes them
    to maximize the AUC score.

    Args:
        X (np.array): Feature matrix
        y (np.array): Target vector (0: Majority, 1: Minority)
        k (int): Number of neighbors to check for borderline detection
        patience (int): How many steps to continue without improvement
        verbose (bool): Whether to print progress logs

    Returns:
        X_final, y_final: Cleaned dataset
    """
    if verbose:
        print(f"BNF Algorithm Started... Total Candidates: {len(X)}")

    # Establish baseline AUC score using Gaussian Naive Bayes classifier
    clf = GaussianNB()
    clf.fit(X, y)
    initial_score = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    
    if verbose:
        print(f"Initial AUC Score: {initial_score:.4f}")

    current_X = X.copy()
    current_y = y.copy()
    best_score = initial_score
    no_improv_count = 0
    
    # Identify borderline majority samples (candidates for removal)
    # A majority sample is considered borderline if most of its k-nearest neighbors are from minority class
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(current_X)
    distances, indices = nbrs.kneighbors(current_X)
    
    candidates_indices = []
    
    for i in range(len(current_X)):
        if current_y[i] == 0:  # Process only majority class samples
            neighbors_idx = indices[i][1:]  # Exclude the point itself (indices[i][0])
            minority_neighbor_count = np.sum(current_y[neighbors_idx] == 1)
            
            # Candidate threshold: if >= 50% of neighbors are minority, mark as borderline
            if minority_neighbor_count >= (k / 2):
                candidates_indices.append(i)
    
    if verbose:
        print(f"Number of Borderline Candidates found: {len(candidates_indices)}")

    # Iterative removal process: test each candidate removal and keep if AUC improves
    mask = np.ones(len(current_X), dtype=bool)  # Boolean mask to track which samples to keep
    removed_count = 0
    
    for idx in candidates_indices:
        # Temporarily remove this sample
        mask[idx] = False
        
        X_temp = current_X[mask]
        y_temp = current_y[mask]
        
        # Safety check: ensure both classes remain after removal
        if len(np.unique(y_temp)) < 2:
            mask[idx] = True
            continue

        # Evaluate AUC score after candidate removal
        clf.fit(X_temp, y_temp)
        try:
            new_score = roc_auc_score(y_temp, clf.predict_proba(X_temp)[:, 1])
        except ValueError:
            new_score = 0
        
        # Decision: keep removal if AUC improved, otherwise revert
        if new_score > best_score:
            best_score = new_score
            no_improv_count = 0
            removed_count += 1
            if verbose:
                print(f"  [Removed] Idx:{idx} -> New AUC: {new_score:.4f} (Improved)")
        else:
            mask[idx] = True  # Revert removal
            no_improv_count += 1
        
        # Early stopping: stop if no improvement for 'patience' consecutive iterations
        if no_improv_count > patience:
            if verbose:
                print(f"  Patience limit reached ({patience} steps without improvement). Stopping BNF.")
            break
            
    X_final = current_X[mask]
    y_final = current_y[mask]
    
    if verbose:
        print(f"BNF Completed. Removed: {removed_count} samples. Final Size: {len(X_final)}")
        
    return X_final, y_final