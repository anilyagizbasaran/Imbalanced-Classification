import sys
import numpy as np
import warnings
import json
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             f1_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

from keel_utils import load_keel_dat
from BNF import apply_bnf
from OBN import apply_obn
from DBSCAN import apply_dbscan_clustering, find_adaptive_eps, SAFE, BORDERLINE
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
        if hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X_test)
        else:
            scores = clf.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, scores)
    except Exception:
        return 0.5


def make_classifier(clf_type):
    if clf_type == 'rf':
        return RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    return SVC(kernel='rbf', random_state=42, class_weight='balanced')


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

    eps = find_adaptive_eps(X_maj, k=5)
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
    maj_idx = np.where(y_tr == 0)[0]
    X_maj   = X_tr[maj_idx]
    n_min   = int(np.sum(y_tr == 1))

    bnf_mask = apply_bnf(X_tr, y_tr, k=5, verbose=verbose)
    obn_mask = apply_obn(X_tr, y_tr, k_neighbors=5, verbose=verbose)

    remove_local = (bnf_mask | obn_mask)[maj_idx]

    eps = find_adaptive_eps(X_maj, k=5)
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


def run_analysis(dataset_name, clf_type='svm'):
    clf_label = 'SVM (RBF, cost-sensitive)' if clf_type == 'svm' else 'Random Forest (100 trees, balanced)'
    report  = f"\n{'='*80}\n"
    report += f"DATASET: {dataset_name.upper()}  |  CLASSIFIER: {clf_label}\n"
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
    report += f"  Evaluation: 5-fold Stratified Cross-Validation  (random_state=42)\n"
    report += f"{'-'*80}\n"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_names = ['Baseline', 'RUS', 'DBSCAN', 'SMOTE', 'BOD']
    results     = {m: {'auc': [], 'gmean': [], 'f1': [], 'mcc': []} for m in model_names}
    fold_log    = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_test)) < 2:
            continue

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)

        fold_aucs = {}

        datasets_fold = {
            'Baseline': (X_train, y_train),
            'RUS':      build_rus_dataset(X_train, y_train),
            'DBSCAN':   build_dbscan_dataset(X_train, y_train),
            'SMOTE':    build_smote_dataset(X_train, y_train),
            'BOD':      build_bod_dataset(X_train, y_train),
        }

        for m in model_names:
            clf = make_classifier(clf_type)
            X_tr_m, y_tr_m = datasets_fold[m]
            clf.fit(X_tr_m, y_tr_m)
            y_pred = clf.predict(X_test)

            results[m]['auc'].append(safe_auc(clf, X_test, y_test))
            results[m]['gmean'].append(g_mean_score(y_test, y_pred))
            results[m]['f1'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            results[m]['mcc'].append(matthews_corrcoef(y_test, y_pred))
            fold_aucs[m] = results[m]['auc'][-1]

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

    base_auc     = avg('Baseline', 'auc')
    best_auc_key = max(model_names, key=lambda m: avg(m, 'auc'))

    report += (f"  {'METHOD':<28} | {'AUC':>7} | {'±':>6} | "
               f"{'G-MEAN':>7} | {'F1':>7} | {'MCC':>7} | {'GAIN':>7}\n")
    report += f"  {'-'*80}\n"
    labels = [
        ('1. Baseline',       'Baseline'),
        ('2. RUS',            'RUS'),
        ('3. DBSCAN',         'DBSCAN'),
        ('4. SMOTE',          'SMOTE'),
        ('5. BOD (proposed)', 'BOD'),
    ]
    for label, key in labels:
        a    = avg(key, 'auc')
        s    = std(key, 'auc')
        g    = avg(key, 'gmean')
        f1   = avg(key, 'f1')
        mcc  = avg(key, 'mcc')
        gain = f"{a - base_auc:>+7.4f}" if key != 'Baseline' else f"{'ref':>7}"
        mark = " (*)" if key == best_auc_key else "    "
        report += (f"  {label:<28} | {a:>7.4f} | {s:>6.4f} | "
                   f"{g:>7.4f} | {f1:>7.4f} | {mcc:>7.4f} | {gain}{mark}\n")
    report += f"  {'-'*80}\n"
    report += f"  (*) = best AUC\n\n"

    bod_gain = avg('BOD', 'auc') - base_auc
    bod_rank = sorted(model_names, key=lambda m: avg(m, 'auc'), reverse=True).index('BOD') + 1
    report += f"  BOD rank: {bod_rank}/{len(model_names)}  |  AUC gain vs Baseline: {bod_gain:+.4f}\n"
    report += f"{'='*80}\n"

    print(report)
    return report, {
        'dataset':     dataset_name,
        'clf_type':    clf_type,
        'ir':          ir,
        'bod_auc':     avg('BOD', 'auc'),
        'base_auc':    base_auc,
        'bod_gain':    bod_gain,
        'bod_rank':    bod_rank,
        'best_method': best_auc_key,
        'aucs':        {m: avg(m, 'auc')   for m in model_names},
        'gmeans':      {m: avg(m, 'gmean') for m in model_names},
        'f1s':         {m: avg(m, 'f1')    for m in model_names},
        'mccs':        {m: avg(m, 'mcc')   for m in model_names},
        'auc_folds':   {m: results[m]['auc'] for m in model_names},
    }


def wilcoxon_section(summaries, model_names):
    """BOD vs diğer yöntemler arası Wilcoxon signed-rank testi (veri seti başına ortalama AUC)."""
    lines  = "\n" + "=" * 80 + "\n"
    lines += "WILCOXON SIGNED-RANK TEST  (BOD vs diğer yöntemler, n=dataset sayısı)\n"
    lines += "H0: fark yok  |  p < 0.05 → BOD istatistiksel olarak farklı\n"
    lines += "=" * 80 + "\n"
    lines += f"  {'Karşılaştırma':<30} | {'W istatistiği':>14} | {'p-değeri':>10} | Sonuç\n"
    lines += f"  {'-'*65}\n"

    bod_aucs = [s['aucs']['BOD'] for s in summaries]
    for m in model_names:
        if m == 'BOD':
            continue
        other_aucs = [s['aucs'][m] for s in summaries]
        diff = np.array(bod_aucs) - np.array(other_aucs)
        if np.all(diff == 0):
            lines += f"  {'BOD vs ' + m:<30} | {'N/A':>14} | {'N/A':>10} | Fark yok\n"
            continue
        try:
            stat, p = wilcoxon(bod_aucs, other_aucs, alternative='greater')
            result = "BOD > " + m + " (p<0.05)" if p < 0.05 else "Anlamlı fark yok"
            lines += f"  {'BOD vs ' + m:<30} | {stat:>14.4f} | {p:>10.4f} | {result}\n"
        except Exception as e:
            lines += f"  {'BOD vs ' + m:<30} | {'hata':>14} | {'hata':>10} | {e}\n"
    lines += "=" * 80 + "\n"
    return lines


if __name__ == "__main__":
    datasets = [
        "glass1", "yeast1", "haberman", "ecoli1", "segment0", "glass6",
        "yeast2vs4", "glass0146vs2", "yeast1vs7", "glass4", "yeast5", "yeast6",
    ]
    model_names = ['Baseline', 'RUS', 'DBSCAN', 'SMOTE', 'BOD']

    for clf_type in ['svm', 'rf']:
        clf_label  = 'SVM' if clf_type == 'svm' else 'RandomForest'
        output_file = f"results_{clf_label}.txt"

        print(f"\n{'#'*80}")
        print(f"# CLASSIFIER: {clf_label}")
        print(f"{'#'*80}")

        final_report  = f"BOD EXPERIMENTAL RESULTS  --  Classifier: {clf_label}\n"
        final_report += "Metrics    : AUC, G-mean, F1 (macro), MCC\n"
        final_report += "Evaluation : 5-fold Stratified CV  |  Adaptive eps  |  Cost-sensitive weights\n"
        final_report += "Datasets   : 12 KEEL imbalanced benchmark sets\n"
        final_report += "=" * 80 + "\n"

        summaries = []
        for ds in datasets:
            print(f"\n>>> Dataset: {ds}  [{clf_label}]")
            report_str, summary = run_analysis(ds, clf_type=clf_type)
            final_report += report_str
            if summary:
                summaries.append(summary)

        # Global AUC tablosu
        final_report += "\n" + "=" * 80 + "\n"
        final_report += f"GLOBAL SUMMARY -- AUC per Dataset [{clf_label}]\n"
        final_report += "=" * 80 + "\n"
        hdr  = f"  {'Dataset':<18} | {'IR':>5} | "
        hdr += " | ".join(f"{m:>8}" for m in model_names) + " | Best\n"
        final_report += hdr
        final_report += (f"  {'-'*18}-+-{'-'*5}-+-"
                         + "-+-".join("-"*8 for _ in model_names)
                         + "-+-" + "-"*8 + "\n")
        for s in summaries:
            row  = f"  {s['dataset']:<18} | {s['ir']:>5.2f} | "
            row += " | ".join(f"{s['aucs'][m]:>8.4f}" for m in model_names)
            row += f" | {s['best_method']}\n"
            final_report += row

        # Global F1 tablosu
        final_report += "\n" + "=" * 80 + "\n"
        final_report += f"GLOBAL SUMMARY -- F1 (macro) per Dataset [{clf_label}]\n"
        final_report += "=" * 80 + "\n"
        final_report += hdr
        for s in summaries:
            row  = f"  {s['dataset']:<18} | {s['ir']:>5.2f} | "
            row += " | ".join(f"{s['f1s'][m]:>8.4f}" for m in model_names)
            row += f" | {max(model_names, key=lambda m: s['f1s'][m])}\n"
            final_report += row

        # Global MCC tablosu
        final_report += "\n" + "=" * 80 + "\n"
        final_report += f"GLOBAL SUMMARY -- MCC per Dataset [{clf_label}]\n"
        final_report += "=" * 80 + "\n"
        final_report += hdr
        for s in summaries:
            row  = f"  {s['dataset']:<18} | {s['ir']:>5.2f} | "
            row += " | ".join(f"{s['mccs'][m]:>8.4f}" for m in model_names)
            row += f" | {max(model_names, key=lambda m: s['mccs'][m])}\n"
            final_report += row

        # Kazanım tablosu
        final_report += "\n" + "=" * 80 + "\n"
        final_report += f"BOD PERFORMANCE vs BASELINE (sorted by IR) [{clf_label}]\n"
        final_report += "=" * 80 + "\n"
        final_report += (f"  {'Dataset':<18} | {'IR':>5} | {'Baseline':>9} | "
                         f"{'BOD':>9} | {'Gain':>8} | {'Rank':>5} | Sonuç\n")
        final_report += f"  {'-'*72}\n"
        for s in sorted(summaries, key=lambda x: x['ir']):
            status = "BAŞARILI" if s['bod_gain'] > 0 else "iyileşme yok"
            final_report += (f"  {s['dataset']:<18} | {s['ir']:>5.2f} | "
                             f"{s['base_auc']:>9.4f} | {s['bod_auc']:>9.4f} | "
                             f"{s['bod_gain']:>+8.4f} | {s['bod_rank']:>5} | {status}\n")
        final_report += "=" * 80 + "\n"

        # Wilcoxon testi
        if len(summaries) >= 5:
            final_report += wilcoxon_section(summaries, model_names)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_report)

        json_file = f"results_{clf_label}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

        print(f"\nTamamlandı! Sonuçlar: '{output_file}'  |  JSON: '{json_file}'")
