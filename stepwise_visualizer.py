import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Import custom modules for data processing and visualization
from keel_utils import load_keel_dat
from BNF import apply_bnf
from OBN import compute_obn_scores, apply_obn 
from RUS import apply_rus_majority
from SMOTE import apply_custom_smote as apply_smote 
from DBSCAN import apply_dbscan_clustering

def plot_step(X_pca, y, title, filename, 
              highlight_indices=None, highlight_label=None, highlight_color='black', highlight_marker='x',
              removed_indices=None,
              new_points_pca=None, new_points_label=None, 
              new_points_color='lime', new_points_marker='+', new_points_size=100, 
              show_removed=True):
    plt.figure(figsize=(12, 8))
    
    # 1. REMOVED SAMPLES (History)
    if show_removed and removed_indices is not None and len(removed_indices) > 0:
        valid_removed = [i for i in removed_indices if i < len(X_pca)]
        if valid_removed:
            plt.scatter(X_pca[valid_removed, 0], X_pca[valid_removed, 1], 
                       c='lightgray', marker='.', alpha=0.3, label='Removed (History)')

    # 2. ACTIVE DATA (Original points still in set)
    all_indices = np.arange(len(X_pca))
    if removed_indices is None: removed_indices = []
    
    active_mask = ~np.isin(all_indices, removed_indices)
    active_idx = all_indices[active_mask]
    
    maj_mask = (y[active_idx] == 0)
    min_mask = (y[active_idx] == 1)
    
    if highlight_indices is not None:
        hl_mask = np.isin(active_idx, highlight_indices) 
        normal_mask = ~hl_mask
        
        plt.scatter(X_pca[active_idx[normal_mask & maj_mask], 0], 
                   X_pca[active_idx[normal_mask & maj_mask], 1], 
                   c='dodgerblue', alpha=0.6, label='Majority')
        
        plt.scatter(X_pca[active_idx[normal_mask & min_mask], 0], 
                   X_pca[active_idx[normal_mask & min_mask], 1], 
                   c='crimson', alpha=0.8, label='Minority')
        
        hl_real_indices = active_idx[hl_mask]
        plt.scatter(X_pca[hl_real_indices, 0], X_pca[hl_real_indices, 1], 
                   c=highlight_color, marker=highlight_marker, s=100, linewidth=2, label=highlight_label)
    else:
        plt.scatter(X_pca[active_idx[maj_mask], 0], X_pca[active_idx[maj_mask], 1], 
                   c='dodgerblue', alpha=0.6, label='Majority')
        
        plt.scatter(X_pca[active_idx[min_mask], 0], X_pca[active_idx[min_mask], 1], 
                   c='crimson', alpha=0.8, label='Minority (Original)')

    # 3. SYNTHETIC SAMPLES (SMOTE)
    if new_points_pca is not None and len(new_points_pca) > 0:
        plt.scatter(new_points_pca[:, 0], new_points_pca[:, 1], 
                   c=new_points_color, marker=new_points_marker, 
                   s=new_points_size, 
                   alpha=0.8, label=new_points_label)

    plt.title(title, fontsize=14, fontweight='bold')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    print(f"--> Plot saved: {filename}")
    plt.close()

def run_stepwise_viz(args):
    if args.dataset.endswith(".dat"):
        path = args.dataset
        name = os.path.basename(args.dataset).replace('.dat', '')
    else:
        path = os.path.join(args.datadir, f"{args.dataset}.dat")
        name = args.dataset
    
    output_folder = f"plots_{name}"
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    print(f"\n{'='*60}\nVISUALIZATION STARTED: {name}\n{'='*60}")

    X, y = load_keel_dat(path)
    vals, counts = np.unique(y, return_counts=True)
    min_label = vals[np.argmin(counts)]
    y_encoded = np.where(y == min_label, 1, 0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    total_removed_indices = []

    # Step 0: Visualize raw data distribution
    plot_step(X_pca, y_encoded, f"Step 0: Raw Data Distribution ({name})", f"{output_folder}/{name}_step0_raw.png")

    # Step 1: Apply BNF (Borderline Noise Filtering) to remove noisy borderline samples
    print("Step 1: BNF Filtering...")
    X_bnf, y_bnf = apply_bnf(X, y_encoded, k=args.obn_k, patience=3)
    # Identify indices of samples removed by BNF
    removed_bnf = np.setdiff1d(np.arange(len(X)), np.where(np.isin(X, X_bnf).all(axis=1))[0])
    
    plot_step(X_pca, y_encoded, "Step 1: Noise Detection with BNF", f"{output_folder}/{name}_step1_bnf.png",
              highlight_indices=removed_bnf, highlight_label='To be Removed (BNF)', highlight_color='black')
    total_removed_indices.extend(removed_bnf)

    # Step 2: Apply OBN (Outlier Based on Noise) to remove outliers from majority class
    print("Step 2: OBN Outlier Detection...")
    X_obn, y_obn = apply_obn(X_bnf, y_bnf, k_neighbors=args.obn_k, multiplier=args.obn_multiplier)
    # Identify indices of samples removed by OBN (additional to BNF removals)
    removed_obn = np.setdiff1d(np.arange(len(X)), np.where(np.isin(X, X_obn).all(axis=1))[0])
    removed_obn = [idx for idx in removed_obn if idx not in total_removed_indices]

    plot_step(X_pca, y_encoded, "Step 2: Outlier Detection with OBN", f"{output_folder}/{name}_step2_obn.png",
              highlight_indices=removed_obn, highlight_label='To be Removed (OBN)', highlight_color='purple', removed_indices=total_removed_indices)
    total_removed_indices.extend(removed_obn)

    # Step 3: Apply DBSCAN clustering to separate safe (clustered) from rare (noise) majority samples
    print("Step 3: DBSCAN Grouping...")
    
    # Get currently active (not yet removed) majority class samples
    current_mask = np.ones(len(X), dtype=bool)
    current_mask[total_removed_indices] = False
    
    # Find indices of majority class samples that are still active
    indices_maj_current = np.where((y_encoded == 0) & current_mask)[0]
    X_maj_current = X[indices_maj_current]
    
    # Apply DBSCAN clustering to majority samples
    labels, _ = apply_dbscan_clustering(X_maj_current, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    
    # Map cluster labels back to global indices: safe (labels != -1) vs rare (labels == -1)
    safe_indices_global = indices_maj_current[labels != -1]
    rare_indices_global = indices_maj_current[labels == -1]
    
    # Visualize DBSCAN clustering results
    plt.figure(figsize=(12, 8))
    if total_removed_indices:
        plt.scatter(X_pca[total_removed_indices, 0], X_pca[total_removed_indices, 1], c='lightgray', marker='.', alpha=0.3, label='Removed History')
    min_indices = np.where(y_encoded == 1)[0]
    plt.scatter(X_pca[min_indices, 0], X_pca[min_indices, 1], c='crimson', alpha=0.8, label='Minority')
    plt.scatter(X_pca[safe_indices_global, 0], X_pca[safe_indices_global, 1], c='mediumseagreen', alpha=0.6, label='SAFE (Cluster)')
    plt.scatter(X_pca[rare_indices_global, 0], X_pca[rare_indices_global, 1], c='orange', marker='D', edgecolor='black', s=50, label='RARE (Protected)')
    plt.title("Step 3: DBSCAN Grouping (Safe vs Rare)", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{name}_step3_dbscan.png", dpi=300)
    plt.close()

    # Step 4: Apply RUS (Random Under Sampling) to reduce safe majority samples to match minority count
    print("Step 4: RUS Process...")
    n_min = np.sum(y_encoded == 1)
    X_safe = X[safe_indices_global]
    y_safe = y_encoded[safe_indices_global]
    
    X_rus, y_rus = apply_rus_majority(X_safe, y_safe, n_target=n_min)
    
    # Identify which safe samples were removed by RUS
    remaining_safe_indices = np.where(np.isin(X, X_rus).all(axis=1))[0]
    removed_rus = np.setdiff1d(safe_indices_global, remaining_safe_indices)

    plot_step(X_pca, y_encoded, "Step 4: Majority Reduction with RUS", f"{output_folder}/{name}_step4_rus.png",
              highlight_indices=removed_rus, highlight_label='To be Removed (RUS)', highlight_color='teal', removed_indices=total_removed_indices)
    total_removed_indices.extend(removed_rus)

    # Step 5: Apply SMOTE (Synthetic Minority Oversampling) to generate synthetic minority samples
    print("Step 5: Hybrid Enhancement (SMOTE)...")
    final_mask = np.ones(len(X), dtype=bool)
    final_mask[total_removed_indices] = False
    X_final_pre = X[final_mask]
    y_final_pre = y_encoded[final_mask]
    
    X_resampled, y_resampled = apply_smote(X_final_pre, y_final_pre, k_neighbors=5)
    
    # Identify newly generated SMOTE samples (added at the end of the dataset)
    X_new_points = X_resampled[len(X_final_pre):]
    
    if len(X_new_points) > 0:
        new_pts_pca = pca.transform(scaler.transform(X_new_points))
        
        plot_step(X_pca, y_encoded, 
                  f"Step 5: Proposed Contribution - Hybrid (Synthetic Generation)", 
                  f"{output_folder}/{name}_step5_hybrid_smote.png",
                  removed_indices=total_removed_indices,
                  new_points_pca=new_pts_pca, 
                  new_points_label='SMOTE Generated Points',
                  new_points_color='lime', new_points_marker='+', new_points_size=120)

        plot_step(X_pca, y_encoded, 
                  f"Step 6: Final Balanced Dataset (Model Input)", 
                  f"{output_folder}/{name}_step6_final_clean.png",
                  removed_indices=total_removed_indices, 
                  show_removed=False, 
                  new_points_pca=new_pts_pca, 
                  new_points_label='Minority (Augmented)', 
                  new_points_color='crimson', new_points_marker='o', new_points_size=40)
                  
    print(f"\nCOMPLETED! Plots are in '{output_folder}' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--datadir", type=str, default="./datasets")
    parser.add_argument("--obn_k", type=int, default=5)
    parser.add_argument("--obn_multiplier", type=float, default=1.8)
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=5)
    args = parser.parse_args()
    run_stepwise_viz(args)