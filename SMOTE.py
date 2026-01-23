import numpy as np
from imblearn.over_sampling import SMOTE

def apply_custom_smote(X, y, k_neighbors=5, random_state=42):
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE).
    Includes safety checks for very small datasets where k_neighbors > n_samples.
    
    Args:
        X (np.array): Feature matrix (balanced via RUS usually).
        y (np.array): Labels.
        k_neighbors (int): Number of nearest neighbors for SMOTE generation.
        random_state (int): Seed for reproducibility.
        
    Returns:
        X_res, y_res: Augmented dataset including synthetic samples.
    """
    try:
        # Safety check: ensure we have enough minority samples for SMOTE
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = min(counts)
        
        # Adjust k_neighbors: must be less than number of minority samples
        effective_k = min(k_neighbors, min_class_count - 1)
        
        if effective_k < 1:
            # Insufficient samples for SMOTE (need at least 2 minority samples)
            return X, y
            
        # Apply SMOTE to generate synthetic minority samples
        smote = SMOTE(k_neighbors=effective_k, random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        
        return X_res, y_res
        
    except Exception as e:
        print(f"Warning: SMOTE failed ({str(e)}). Returning original data.")
        return X, y