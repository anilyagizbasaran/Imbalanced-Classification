import numpy as np


def apply_rus_majority(X, y, n_target=None, random_state=42):
    if n_target is None or len(X) == 0:
        return X, y
    if len(X) <= n_target:
        return X, y
    np.random.seed(random_state)
    idx = np.random.choice(len(X), n_target, replace=False)
    return X[idx], y[idx]
