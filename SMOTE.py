import numpy as np
from imblearn.over_sampling import SMOTE


def apply_custom_smote(X, y, k_neighbors=5, random_state=42):
    try:
        _, counts = np.unique(y, return_counts=True)
        effective_k = min(k_neighbors, min(counts) - 1)
        if effective_k < 1:
            return X, y
        smote = SMOTE(k_neighbors=effective_k, random_state=random_state)
        return smote.fit_resample(X, y)
    except Exception as e:
        print(f"Warning: SMOTE failed ({e}). Returning original data.")
        return X, y
