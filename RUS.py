import numpy as np

def apply_rus_majority(X, y, n_target=None, random_state=42):
    """
    Applies Random Under Sampling (RUS) to the given dataset.
    Typically used to reduce the 'Safe Majority' samples to match the minority count.
    
    Args:
        X (np.array): Feature matrix.
        y (np.array): Labels (usually all 0 since this is applied to majority subset).
        n_target (int): The exact number of samples to keep.
        random_state (int): Seed for reproducibility.
        
    Returns:
        X_res, y_res: Resampled features and labels.
    """
    # Return original data if no target specified or input is empty
    if n_target is None or len(X) == 0:
        return X, y
        
    n_samples = len(X)
    
    # Return original data if already at or below target size
    if n_samples <= n_target:
        return X, y
        
    # Randomly select n_target samples using stratified random sampling
    np.random.seed(random_state)
    selected_indices = np.random.choice(n_samples, n_target, replace=False)
    
    return X[selected_indices], y[selected_indices]