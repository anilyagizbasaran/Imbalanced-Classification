import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from keel_utils import load_keel_dat
from BNF import apply_bnf
from OBN import apply_obn
from DBSCAN import apply_dbscan_clustering, SAFE, BORDERLINE
from SMOTE import apply_custom_smote
from RUS import apply_rus_majority

warnings.filterwarnings("ignore")


def g_mean_score(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(np.sqrt(tpr * tnr))


def safe_auc(clf, X_test, y_test):
    try:
        return roc_auc_score(y_test, clf.decision_function(X_test))
    except Exception:
        return 0.5


def build_rus_dataset(X_tr, y_tr):
    n_min = int(np.sum(y_tr == 1))
    X_maj, y_maj = X_tr[y_tr == 0], y_tr[y_tr == 0]
    X_min, y_min = X_tr[y_tr == 1], y_tr[y_tr == 1]
    X_maj_r, y_maj_r = apply_rus_majority(X_maj, y_maj, n_target=n_min)
    return np.vstack([X_maj_r, X_min]), np.hstack([y_maj_r, y_min])


def build_dbscan_dataset(X_tr, y_tr):
    n_min   = int(np.sum(y_tr == 1))
    maj_idx = np.where(y_tr == 0)[0]
    X_maj   = X_tr[maj_idx]

    eps = 0.5 * np.sqrt(X_maj.shape[1])
    categories, _ = apply_dbscan_clustering(X_maj, eps=eps, min_samples=5)

    if len(categories) == 0:
        return X_tr, y_tr

    X_safe = X_maj[(categories == SAFE) | (categories == BORDERLINE)]
    X_min  = X_tr[y_tr == 1]
    y_min  = y_tr[y_tr == 1]

    if len(X_safe) > n_min:
        np.random.seed(42)
        X_safe = X_safe[np.random.choice(len(X_safe), n_min, replace=False)]

    return np.vstack([X_safe, X_min]), np.hstack([np.zeros(len(X_safe)), y_min])


def build_smote_dataset(X_tr, y_tr):
    return apply_custom_smote(X_tr, y_tr, k_neighbors=5)


def build_bod_dataset(X_tr, y_tr, verbose=False):
    """
    BOD under-sampling pipeline — Peng & Park (2022), Algorithm 2.
      1. BNF  : remove borderline-noise majority samples
      2. OBN  : remove outlier majority samples
      3. DBSCAN: Safe+Borderline -> apply RUS (target=n_min); Rare+Outlier -> protect
      4. Final: RUS(Safe) + Rare + Minority
    """
    maj_idx = np.where(y_tr == 0)[0]
    X_maj   = X_tr[maj_idx]
    n_min   = int(np.sum(y_tr == 1))

    bnf_mask = apply_bnf(X_tr, y_tr, k=5, verbose=verbose)
    obn_mask = apply_obn(X_tr, y_tr, k_neighbors=5, verbose=verbose)

    remove_local = (bnf_mask | obn_mask)[maj_idx]

    eps = 0.5 * np.sqrt(X_maj.shape[1])
    categories, _ = apply_dbscan_clustering(X_maj, eps=eps, min_samples=5)

    if len(categories) == 0:
        return X_tr, y_tr

    in_cluster  = (categories == SAFE) | (categories == BORDERLINE)
    safe_local  = in_cluster  & ~remove_local
    rare_local  = (~in_cluster) & ~remove_local

    X_safe = X_maj[safe_local]
    X_rare = X_maj[rare_local]

    if len(X_safe) > n_min:
        np.random.seed(42)
        X_safe = X_safe[np.random.choice(len(X_safe), n_min, replace=False)]

    X_min  = X_tr[y_tr == 1]
    y_min  = y_tr[y_tr == 1]

    parts = [(X_safe, np.zeros(len(X_safe))),
             (X_rare, np.zeros(len(X_rare))),
             (X_min,  y_min)]
    parts = [(xp, yp) for xp, yp in parts if len(xp) > 0]

    return np.vstack([xp for xp, _ in parts]), np.hstack([yp for _, yp in parts])


def run_analysis(dataset_name):
    report  = f"\n{'='*80}\n"
    report += f"DATASET: {dataset_name.upper()}\n"
    report += f"{'='*80}\n"

    try:
        X, y = load_keel_dat(f"datasets/{dataset_name}.dat")
    except Exception as e:
        msg = f"Error loading {dataset_name}: {e}\n"
        print(msg)
        return msg, {}

    n_maj = int(np.sum(y == 0))
    n_min = int(np.sum(y == 1))
    ir    = n_maj / max(n_min, 1)
    report += f"  Samples : {len(X)}  |  Features : {X.shape[1]}\n"
    report += f"  Majority: {n_maj}  |  Minority : {n_min}  |  IR : {ir:.2f}\n"
    report += f"  Classifier: SVM (RBF kernel, C=1, gamma='scale') + StandardScaler\n"
    report += f"  Evaluation: 5-fold Stratified Cross-Validation  (random_state=42)\n"
    report += f"{'-'*80}\n"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_names = ['Baseline', 'RUS', 'DBSCAN', 'SMOTE', 'BOD']
    results     = {m: {'auc': [], 'gmean': []} for m in model_names}
    fold_log    = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_test)) < 2:
            continue

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)

        clf      = SVC(kernel='rbf', random_state=42)
        fold_aucs = {}

        clf.fit(X_train, y_train)
        fold_aucs['Baseline'] = safe_auc(clf, X_test, y_test)
        results['Baseline']['auc'].append(fold_aucs['Baseline'])
        results['Baseline']['gmean'].append(g_mean_score(y_test, clf.predict(X_test)))

        X_r, y_r = build_rus_dataset(X_train, y_train)
        clf.fit(X_r, y_r)
        fold_aucs['RUS'] = safe_auc(clf, X_test, y_test)
        results['RUS']['auc'].append(fold_aucs['RUS'])
        results['RUS']['gmean'].append(g_mean_score(y_test, clf.predict(X_test)))

        X_d, y_d = build_dbscan_dataset(X_train, y_train)
        clf.fit(X_d, y_d)
        fold_aucs['DBSCAN'] = safe_auc(clf, X_test, y_test)
        results['DBSCAN']['auc'].append(fold_aucs['DBSCAN'])
        results['DBSCAN']['gmean'].append(g_mean_score(y_test, clf.predict(X_test)))

        X_s, y_s = build_smote_dataset(X_train, y_train)
        clf.fit(X_s, y_s)
        fold_aucs['SMOTE'] = safe_auc(clf, X_test, y_test)
        results['SMOTE']['auc'].append(fold_aucs['SMOTE'])
        results['SMOTE']['gmean'].append(g_mean_score(y_test, clf.predict(X_test)))

        print(f"  Fold {fold}: Running BOD pipeline...")
        X_b, y_b = build_bod_dataset(X_train, y_train)
        clf.fit(X_b, y_b)
        fold_aucs['BOD'] = safe_auc(clf, X_test, y_test)
        results['BOD']['auc'].append(fold_aucs['BOD'])
        results['BOD']['gmean'].append(g_mean_score(y_test, clf.predict(X_test)))

        fold_log.append(fold_aucs)
        print(f"  Fold {fold} complete.")

    report += f"  PER-FOLD AUC\n"
    header  = f"  {'Fold':<6}"
    for m in model_names:
        header += f" | {m:>9}"
    report += header + "\n"
    report += f"  {'-'*6}" + (f"-+-{'-'*9}" * len(model_names)) + "\n"
    for i, fa in enumerate(fold_log, 1):
        row = f"  {'F'+str(i):<6}"
        for m in model_names:
            row += f" | {fa[m]:>9.4f}"
        report += row + "\n"
    report += f"{'-'*80}\n"

    avg = lambda m, k: np.mean(results[m][k]) if results[m][k] else 0.0
    std = lambda m, k: np.std(results[m][k])  if results[m][k] else 0.0

    labels = [
        ('1. Baseline (SVM)',       'Baseline'),
        ('2. SVM + RUS',            'RUS'),
        ('3. SVM + DBSCAN',         'DBSCAN'),
        ('4. SVM + SMOTE',          'SMOTE'),
        ('5. SVM + BOD (proposed)', 'BOD'),
    ]

    base_auc      = avg('Baseline', 'auc')
    best_auc_key  = max(model_names, key=lambda m: avg(m, 'auc'))

    report += f"  {'METHOD':<28} | {'AUC (mean)':>10} | {'AUC (std)':>9} | {'G-MEAN':>8} | {'GAIN AUC':>10}\n"
    report += f"  {'-'*75}\n"
    for label, key in labels:
        a      = avg(key, 'auc')
        s      = std(key, 'auc')
        g      = avg(key, 'gmean')
        gain   = f"{a - base_auc:>+10.4f}" if key != 'Baseline' else f"{'Reference':>10}"
        marker = " (*)" if key == best_auc_key else "    "
        report += f"  {label:<28} | {a:>10.4f} | {s:>9.4f} | {g:>8.4f} | {gain}{marker}\n"
    report += f"  {'-'*75}\n"
    report += f"  (*) = best AUC for this dataset\n\n"

    bod_gain = avg('BOD', 'auc') - base_auc
    bod_rank = sorted(model_names, key=lambda m: avg(m, 'auc'), reverse=True).index('BOD') + 1
    report += f"  BOD rank   : {bod_rank} / {len(model_names)}\n"
    report += f"  BOD AUC gain vs Baseline: {bod_gain:+.4f}\n"
    report += ("  >> SUCCESS: BOD outperformed Baseline SVM.\n"
               if bod_gain > 0 else
               "  >> NOTE: BOD did not improve AUC for this dataset.\n")
    report += f"{'='*80}\n"

    print(report)
    return report, {
        'dataset':      dataset_name,
        'ir':           ir,
        'bod_auc':      avg('BOD', 'auc'),
        'base_auc':     base_auc,
        'bod_gain':     bod_gain,
        'bod_rank':     bod_rank,
        'best_method':  best_auc_key,
        'aucs':         {m: avg(m, 'auc')   for m in model_names},
        'gmeans':       {m: avg(m, 'gmean') for m in model_names},
    }


if __name__ == "__main__":
    datasets = [
        "glass1", "yeast1", "haberman", "ecoli1", "segment0", "glass6",
        "yeast2vs4", "glass0146vs2", "yeast1vs7", "glass4", "yeast5", "yeast6",
    ]
    output_file = "all_results_summary.txt"

    print(f"Starting BOD analysis (Peng & Park 2022)... Results -> '{output_file}'")
    final_report  = "BOD EXPERIMENTAL RESULTS  --  Peng & Park (2022)\n"
    final_report += "Classifier : SVM (RBF kernel), StandardScaler, 5-fold Stratified CV\n"
    final_report += "Metrics    : AUC (mean +/- std over 5 folds), G-mean\n"
    final_report += "Datasets   : 12 KEEL imbalanced benchmark sets\n"
    final_report += "=" * 80 + "\n"

    summaries = []
    for ds in datasets:
        print(f"\n>>> Dataset: {ds}")
        report_str, summary = run_analysis(ds)
        final_report += report_str
        summaries.append(summary)

    model_names = ['Baseline', 'RUS', 'DBSCAN', 'SMOTE', 'BOD']

    final_report += "\n" + "=" * 80 + "\n"
    final_report += "GLOBAL SUMMARY -- AUC per Dataset\n"
    final_report += "=" * 80 + "\n"
    hdr  = f"  {'Dataset':<18} | {'IR':>5} | "
    hdr += " | ".join(f"{m:>8}" for m in model_names)
    hdr += " | Best\n"
    final_report += hdr
    final_report += (f"  {'-'*18}-+-{'-'*5}-+-"
                     + "-+-".join("-"*8 for _ in model_names)
                     + "-+-" + "-"*8 + "\n")
    for s in summaries:
        row  = f"  {s['dataset']:<18} | {s['ir']:>5.2f} | "
        row += " | ".join(f"{s['aucs'][m]:>8.4f}" for m in model_names)
        row += f" | {s['best_method']}\n"
        final_report += row

    final_report += f"\n  {'METHOD':<12} | {'Wins (best AUC)':>15} | {'BOD SUCCESS':>12}\n"
    final_report += f"  {'-'*45}\n"
    for m in model_names:
        wins = sum(1 for s in summaries if s['best_method'] == m)
        if m == 'BOD':
            successes = sum(1 for s in summaries if s['bod_gain'] > 0)
            final_report += f"  {m:<12} | {wins:>15} | {successes:>10}/12\n"
        else:
            final_report += f"  {m:<12} | {wins:>15} |\n"

    final_report += "\n" + "=" * 80 + "\n"
    final_report += "BOD PERFORMANCE vs BASELINE (sorted by IR)\n"
    final_report += "=" * 80 + "\n"
    final_report += f"  {'Dataset':<18} | {'IR':>5} | {'Baseline':>9} | {'BOD':>9} | {'Gain':>8} | {'Rank':>5} | Result\n"
    final_report += f"  {'-'*72}\n"
    for s in sorted(summaries, key=lambda x: x['ir']):
        status = "SUCCESS" if s['bod_gain'] > 0 else "fail"
        final_report += (f"  {s['dataset']:<18} | {s['ir']:>5.2f} | "
                         f"{s['base_auc']:>9.4f} | {s['bod_auc']:>9.4f} | "
                         f"{s['bod_gain']:>+8.4f} | {s['bod_rank']:>5} | {status}\n")
    final_report += "=" * 80 + "\n"

    with open(output_file, "w") as f:
        f.write(final_report)

    print(f"\nDone! See '{output_file}'")
