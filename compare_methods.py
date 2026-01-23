import argparse
import numpy as np
import warnings
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

# Import custom modules for data processing and resampling methods
from keel_utils import load_keel_dat
from BNF import apply_bnf
from DBSCAN import apply_dbscan_clustering
from RUS import apply_rus_majority

warnings.filterwarnings("ignore")

def get_safe_probs(model, X):
    """
    Prevents IndexError by checking the shape of predict_proba.
    Ensures a probability vector is returned even if only one class is predicted.
    """
    probs = model.predict_proba(X)
    if probs.shape[1] == 1:
        predicted_class = model.classes_[0]
        # If the only predicted class is 1, return 1s; otherwise return 0s
        return probs[:, 0] if predicted_class == 1 else np.zeros(probs.shape[0])
    return probs[:, 1]

def run_analysis(dataset_name):
    report = ""
    report += f"\n{'='*80}\n"
    report += f"RESEARCH ANALYSIS: {dataset_name.upper()}\n"
    report += f"{'='*80}\n"

    try:
        X, y = load_keel_dat(f"datasets/{dataset_name}.dat")
    except Exception as e:
        msg = f"Error: Could not load {dataset_name}: {e}\n"
        print(msg)
        return msg

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {
        'Baseline':  {'auc': [], 'f1': []},
        'Reference': {'auc': [], 'f1': []},
        'Proposed':  {'auc': [], 'f1': []}
    }

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip fold if test set doesn't contain both classes (required for AUC)
        if len(np.unique(y_test)) < 2: 
            continue

        clf = GaussianNB()

        # Method 1: Baseline - train on raw imbalanced data
        clf.fit(X_train, y_train)
        results['Baseline']['auc'].append(roc_auc_score(y_test, get_safe_probs(clf, X_test)))
        results['Baseline']['f1'].append(f1_score(y_test, clf.predict(X_test)))

        # Method 2: Reference method (Peng & Park) - BNF + DBSCAN + RUS (under-sampling only)
        X_bnf_ref, y_bnf_ref = apply_bnf(X_train, y_train, k=5, patience=0, verbose=False)
        X_maj = X_bnf_ref[y_bnf_ref == 0]
        labels, _ = apply_dbscan_clustering(X_maj, eps=0.5, min_samples=5)
        X_safe = X_maj[labels != -1]  # Keep only safe (clustered) majority samples
        n_min = np.sum(y_bnf_ref == 1)
        
        if len(X_safe) > 0:
            X_rus, y_rus = apply_rus_majority(X_safe, np.zeros(len(X_safe)), n_target=n_min)
            X_ref_f = np.vstack([X_rus, X_bnf_ref[y_bnf_ref == 1]])
            y_ref_f = np.hstack([y_rus, y_bnf_ref[y_bnf_ref == 1]])
        else:
            X_ref_f, y_ref_f = X_bnf_ref, y_bnf_ref

        clf.fit(X_ref_f, y_ref_f)
        results['Reference']['auc'].append(roc_auc_score(y_test, get_safe_probs(clf, X_test)))
        results['Reference']['f1'].append(f1_score(y_test, clf.predict(X_test)))

        # Method 3: Proposed method - Hybrid approach (BNF + DBSCAN + RUS + SMOTE)
        X_prop, y_prop = apply_bnf(X_train, y_train, k=5, patience=3, verbose=False)
        X_maj_p = X_prop[y_prop == 0]
        labels_p, _ = apply_dbscan_clustering(X_maj_p, eps=0.5, min_samples=5)
        X_safe_p = X_maj_p[labels_p != -1]
        n_min_p = np.sum(y_prop == 1)
        
        if len(X_safe_p) > 0:
            X_rus_p, y_rus_p = apply_rus_majority(X_safe_p, np.zeros(len(X_safe_p)), n_target=n_min_p)
            X_mid = np.vstack([X_rus_p, X_prop[y_prop == 1]])
            y_mid = np.hstack([y_rus_p, y_prop[y_prop == 1]])
        else:
            X_mid, y_mid = X_prop, y_prop
        
        # Apply SMOTE to balance the dataset after under-sampling
        try:
            k_smote = min(5, np.sum(y_mid == 1) - 1)
            if k_smote > 0:
                sm = SMOTE(k_neighbors=k_smote, random_state=42)
                X_f, y_f = sm.fit_resample(X_mid, y_mid)
            else: 
                X_f, y_f = X_mid, y_mid
        except: 
            X_f, y_f = X_mid, y_mid

        clf.fit(X_f, y_f)
        results['Proposed']['auc'].append(roc_auc_score(y_test, get_safe_probs(clf, X_test)))
        results['Proposed']['f1'].append(f1_score(y_test, clf.predict(X_test)))
        fold += 1

    # Calculate average metrics across all folds
    def get_avg(m, met): return np.mean(results[m][met]) if results[m][met] else 0

    m1_a, m1_f = get_avg('Baseline', 'auc'), get_avg('Baseline', 'f1')
    m2_a, m2_f = get_avg('Reference', 'auc'), get_avg('Reference', 'f1')
    m3_a, m3_f = get_avg('Proposed', 'auc'), get_avg('Proposed', 'f1')

    # Generate results table
    report += f"{'METHOD':<25} | {'AUC SCORE':<12} | {'F1-SCORE':<12} | {'GAIN (AUC)'}\n"
    report += f"{'-'*80}\n"
    report += f"{'1. Baseline':<25} | {m1_a:.4f}       | {m1_f:.4f}       | Reference\n"
    report += f"{'2. Reference (RUS)':<25} | {m2_a:.4f}       | {m2_f:.4f}       | {m2_a - m1_a:+.4f}\n"
    report += f"{'3. Proposed (Hybrid)':<25} | {m3_a:.4f}       | {m3_f:.4f}       | {m3_a - m1_a:+.4f}\n"
    report += f"{'-'*80}\n"

    # Compare proposed method (with SMOTE) against reference method (without SMOTE)
    hybrid_gain = m3_a - m2_a
    improvement_pct = (hybrid_gain / m2_a * 100) if m2_a != 0 else 0
    
    report += f"HYBRID ANALYSIS RESULT:\n"
    report += f">> Adding SMOTE resulted in a {improvement_pct:+.2f}% change in AUC score.\n"
    if hybrid_gain > 0:
        report += f">> SUCCESS: The hybrid approach outperformed the under-sampling-only method.\n"
    else:
        report += f">> NOTE: The hybrid approach did not provide additional gain for this dataset.\n"
    report += f"{'='*80}\n"
    
    print(report)
    return report

if __name__ == "__main__":
    # List of all datasets to test
    datasets = ["glass1", "ecoli1", "glass6", "haberman", "segment0", "yeast1"]
    output_filename = "all_results_summary.txt"
    
    print(f"Starting analysis... Results will be saved to '{output_filename}'.")
    
    final_report = "EXPERIMENTAL RESULTS SUMMARY REPORT\n"
    final_report += "===================================\n"
    
    for ds in datasets:
        print(f">>> Processing dataset: {ds}")
        ds_report = run_analysis(ds)
        final_report += ds_report
    
    # Write to file
    with open(output_filename, "w") as f:
        f.write(final_report)
            
    print(f"\nProcessing complete! Please check '{output_filename}' for the results.")