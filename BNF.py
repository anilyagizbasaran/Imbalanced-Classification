import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def apply_bnf(X, y, k=5, delta=0.5, verbose=False):
    """
    BNF (Borderline Noise Factor) — Yang & Gao (2013), adopted in Peng & Park (2022).

    BNF(x) = alpha*(Ks+delta)/(|kNS(x)|+delta) + beta*|kND(x)|
    alpha=0.3, beta=0.7, delta=0.5, Ks=k

    SOVR: a sample enters SOVR because it is a k-NN of an opposite-class sample.
    Iteratively removes the highest-BNF majority member of SOVR while AUC improves
    on a held-out 20% stratified validation split.
    """
    alpha, beta = 0.3, 0.7
    n = len(X)

    if verbose:
        print(f"  [BNF] n={n}, k={k}")

    min_cls = min(np.sum(y == 0), np.sum(y == 1))
    if min_cls < 5:
        return np.zeros(n, dtype=bool)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    proc_idx, val_idx = next(sss.split(X, y))
    X_proc, y_proc = X[proc_idx], y[proc_idx]
    X_val,  y_val  = X[val_idx],  y[val_idx]

    if len(np.unique(y_val)) < 2:
        return np.zeros(n, dtype=bool)

    nbrs_all = NearestNeighbors(n_neighbors=k + 1).fit(X_proc)
    _, nn_idx = nbrs_all.kneighbors(X_proc)

    sovr_set = set()
    for i in range(len(X_proc)):
        for nbr in nn_idx[i][1:]:
            if y_proc[nbr] != y_proc[i]:
                sovr_set.add(int(nbr))

    sovr = [i for i in sovr_set if y_proc[i] == 0]

    if verbose:
        print(f"  [BNF] SOVR majority size: {len(sovr)}")
    if not sovr:
        return np.zeros(n, dtype=bool)

    clf = SVC(kernel='rbf', random_state=42)
    clf.fit(X_proc, y_proc)
    best_auc = roc_auc_score(y_val, clf.decision_function(X_val))
    if verbose:
        print(f"  [BNF] baseline val AUC: {best_auc:.4f}")

    bnf_local   = np.zeros(len(X_proc), dtype=bool)
    active      = np.ones(len(X_proc),  dtype=bool)
    active_sovr = set(sovr)

    while active_sovr:
        X_cur   = X_proc[active]
        y_cur   = y_proc[active]
        act_idx = np.where(active)[0]
        local   = {g: l for l, g in enumerate(act_idx)}

        n_fit  = min(k + 1, len(X_cur))
        nn_cur = NearestNeighbors(n_neighbors=n_fit).fit(X_cur)
        _, nn_loc = nn_cur.kneighbors(X_cur)

        bnf_vals = {}
        for g in list(active_sovr):
            if not active[g]:
                active_sovr.discard(g)
                continue
            l       = local[g]
            nbr_g   = act_idx[nn_loc[l][1:]]
            kNS = int(np.sum(y_proc[nbr_g] == 0))
            kND = int(np.sum(y_proc[nbr_g] == 1))
            bnf_vals[g] = alpha * (k + delta) / (kNS + delta) + beta * kND

        if not bnf_vals:
            break

        best_g = max(bnf_vals, key=bnf_vals.get)
        active[best_g] = False
        X_try, y_try = X_proc[active], y_proc[active]

        if len(np.unique(y_try)) < 2:
            active[best_g] = True
            active_sovr.discard(best_g)
            continue

        clf.fit(X_try, y_try)
        try:
            new_auc = roc_auc_score(y_val, clf.decision_function(X_val))
        except Exception:
            active[best_g] = True
            active_sovr.discard(best_g)
            continue

        if new_auc >= best_auc:
            best_auc = new_auc
            bnf_local[best_g] = True
            active_sovr.discard(best_g)
            if verbose:
                print(f"  [BNF] removed proc_idx={best_g} "
                      f"BNF={bnf_vals[best_g]:.3f} val_AUC={new_auc:.4f}")
        else:
            active[best_g] = True
            if verbose:
                print(f"  [BNF] stopped. val_AUC would drop to {new_auc:.4f}")
            break

    bnf_mask = np.zeros(n, dtype=bool)
    bnf_mask[proc_idx[bnf_local]] = True
    if verbose:
        print(f"  [BNF] done. Removed: {bnf_mask.sum()}")
    return bnf_mask
