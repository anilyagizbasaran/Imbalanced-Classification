"""
Stepwise BOD pipeline visualiser -- Peng & Park (2022).

Steps:
  0 : Raw data distribution
  1 : BNF  -- majority samples marked for removal (borderline noise)
  2 : OBN  -- majority samples marked for removal (outliers)
  3 : DBSCAN categorisation (Safe / Rare / BNF-removed / OBN-removed)
  4 : RUS on Safe category
  5 : Final balanced training set
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keel_utils import load_keel_dat
from BNF import apply_bnf
from OBN import apply_obn
from DBSCAN import apply_dbscan_clustering, SAFE, BORDERLINE


def _save(fig, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"--> Plot saved: {path}")


def plot_step(X_pca, y, title, filename,
              highlight_indices=None, highlight_label=None,
              highlight_color='black', highlight_marker='x',
              removed_indices=None, show_removed=True,
              new_points_pca=None, new_points_label=None,
              new_points_color='lime', new_points_marker='+',
              new_points_size=100):

    fig, ax = plt.subplots(figsize=(12, 8))
    removed_indices = [] if removed_indices is None else list(removed_indices)

    if show_removed and removed_indices:
        ax.scatter(X_pca[removed_indices, 0], X_pca[removed_indices, 1],
                   c='lightgray', marker='.', alpha=0.3,
                   label='Removed (history)', zorder=1)

    all_idx    = np.arange(len(X_pca))
    active_idx = all_idx[~np.isin(all_idx, removed_indices)]
    maj_mask   = y[active_idx] == 0
    min_mask   = y[active_idx] == 1

    if highlight_indices is not None:
        hl_mask = np.isin(active_idx, highlight_indices)

        ax.scatter(X_pca[active_idx[~hl_mask & maj_mask], 0],
                   X_pca[active_idx[~hl_mask & maj_mask], 1],
                   c='dodgerblue', alpha=0.6, label='Majority', zorder=2)

        hl_idx = active_idx[hl_mask]
        if len(hl_idx):
            ax.scatter(X_pca[hl_idx, 0], X_pca[hl_idx, 1],
                       c='dodgerblue', edgecolors=highlight_color,
                       linewidths=2.5, s=130, zorder=3,
                       label=f'{highlight_label} (majority)')

        ax.scatter(X_pca[active_idx[~hl_mask & min_mask], 0],
                   X_pca[active_idx[~hl_mask & min_mask], 1],
                   c='crimson', alpha=0.8, label='Minority', zorder=4)

        if len(hl_idx):
            ax.scatter(X_pca[hl_idx, 0], X_pca[hl_idx, 1],
                       c=highlight_color, marker=highlight_marker,
                       s=80, linewidths=1.5, zorder=5, alpha=0.9,
                       label='_nolegend_')
    else:
        ax.scatter(X_pca[active_idx[maj_mask], 0],
                   X_pca[active_idx[maj_mask], 1],
                   c='dodgerblue', alpha=0.6, label='Majority', zorder=2)
        ax.scatter(X_pca[active_idx[min_mask], 0],
                   X_pca[active_idx[min_mask], 1],
                   c='crimson', alpha=0.8, label='Minority', zorder=3)

    ax.set_title(title, fontsize=14, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(),
              dict(zip(labels, handles)).keys(), loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    _save(fig, filename)


def run_stepwise_viz(args):
    if args.dataset.endswith('.dat'):
        path = args.dataset
        name = os.path.basename(args.dataset).replace('.dat', '')
    else:
        path = os.path.join(args.datadir, f"{args.dataset}.dat")
        name = args.dataset

    base_dir = getattr(args, 'outdir', 'result_images')
    out = os.path.join(base_dir, f"plots_{name}")
    os.makedirs(out, exist_ok=True)
    print(f"\n{'='*60}\nVISUALISATION (BOD): {name}\n{'='*60}")

    X, y = load_keel_dat(path)
    vals, counts = np.unique(y, return_counts=True)
    y_enc = np.where(y == vals[np.argmin(counts)], 1, 0)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=2)
    X_pca  = pca.fit_transform(X_sc)

    eps   = 0.5 * np.sqrt(X.shape[1])
    n_min = int(np.sum(y_enc == 1))

    plot_step(X_pca, y_enc,
              f"Step 0: Raw Data Distribution ({name})",
              f"{out}/{name}_step0_raw.png")

    print("Step 1: BNF...")
    bnf_mask   = apply_bnf(X, y_enc, k=args.k, verbose=True)
    bnf_global = np.where(bnf_mask)[0]

    plot_step(X_pca, y_enc,
              "Step 1: BNF -- Borderline Noise Marked for Removal",
              f"{out}/{name}_step1_bnf.png",
              highlight_indices=bnf_global,
              highlight_label=f'BNF: to remove ({len(bnf_global)})',
              highlight_color='black', highlight_marker='x')

    print("Step 2: OBN...")
    obn_mask = apply_obn(X, y_enc, k_neighbors=args.k, verbose=True)
    obn_only = np.where(obn_mask & ~bnf_mask)[0]

    plot_step(X_pca, y_enc,
              "Step 2: OBN -- Outliers Marked for Removal",
              f"{out}/{name}_step2_obn.png",
              highlight_indices=obn_only,
              highlight_label=f'OBN: to remove ({len(obn_only)})',
              highlight_color='purple', highlight_marker='x')

    print(f"Step 3: DBSCAN (eps={eps:.4f})...")
    maj_idx  = np.where(y_enc == 0)[0]
    X_maj_sc = X_sc[maj_idx]

    categories, _ = apply_dbscan_clustering(X_maj_sc, eps=eps,
                                            min_samples=args.min_samples)
    in_cluster = (categories == SAFE) | (categories == BORDERLINE)

    remove_local = (bnf_mask | obn_mask)[maj_idx]
    safe_local   = in_cluster  & ~remove_local
    rare_local   = (~in_cluster) & ~remove_local

    safe_global = maj_idx[safe_local]
    rare_global = maj_idx[rare_local]
    bnf_rem     = np.where(bnf_mask)[0]
    obn_rem     = np.where(obn_mask & ~bnf_mask)[0]
    min_global  = np.where(y_enc == 1)[0]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X_pca[min_global, 0],    X_pca[min_global, 1],
               c='crimson', alpha=0.8, s=30, label='Minority')
    ax.scatter(X_pca[safe_global, 0],   X_pca[safe_global, 1],
               c='mediumseagreen', alpha=0.6, s=30,
               label=f'SAFE -- apply RUS ({len(safe_global)})')
    ax.scatter(X_pca[rare_global, 0],   X_pca[rare_global, 1],
               c='orange', marker='D', edgecolors='black', s=55,
               label=f'RARE -- protected ({len(rare_global)})')
    if len(bnf_rem):
        ax.scatter(X_pca[bnf_rem, 0],   X_pca[bnf_rem, 1],
                   c='black', marker='x', s=70, linewidths=2,
                   label=f'BNF -- removed ({len(bnf_rem)})')
    if len(obn_rem):
        ax.scatter(X_pca[obn_rem, 0],   X_pca[obn_rem, 1],
                   c='purple', marker='x', s=70, linewidths=2,
                   label=f'OBN -- removed ({len(obn_rem)})')
    ax.set_title("Step 3: DBSCAN Categorisation (Safe / Rare / BNF / OBN)",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    _save(fig, f"{out}/{name}_step3_dbscan.png")

    print("Step 4: RUS...")
    np.random.seed(42)
    if len(safe_global) > n_min:
        sel          = np.random.choice(len(safe_global), n_min, replace=False)
        safe_kept    = safe_global[sel]
        safe_removed = np.setdiff1d(safe_global, safe_kept)
    else:
        safe_kept    = safe_global
        safe_removed = np.array([], dtype=int)

    removed_bnf_obn = np.concatenate([bnf_rem, obn_rem])

    plot_step(X_pca, y_enc,
              f"Step 4: RUS -- Safe Reduced to {n_min} (= minority count)",
              f"{out}/{name}_step4_rus.png",
              highlight_indices=safe_removed,
              highlight_label=f'Safe pruned by RUS ({len(safe_removed)})',
              highlight_color='teal', highlight_marker='x',
              removed_indices=removed_bnf_obn)

    all_removed = np.concatenate([removed_bnf_obn, safe_removed]).astype(int)
    final_maj   = np.concatenate([safe_kept, rare_global])
    n_final_maj = len(final_maj)
    n_final_min = len(min_global)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X_pca[all_removed, 0], X_pca[all_removed, 1],
               c='lightgray', marker='.', alpha=0.3,
               label='Removed (history)', zorder=1)
    ax.scatter(X_pca[safe_kept, 0],   X_pca[safe_kept, 1],
               c='dodgerblue', alpha=0.7, s=35,
               label=f'Majority -- Safe after RUS ({len(safe_kept)})', zorder=2)
    if len(rare_global):
        ax.scatter(X_pca[rare_global, 0], X_pca[rare_global, 1],
                   c='deepskyblue', marker='D', s=45, alpha=0.8,
                   label=f'Majority -- Rare protected ({len(rare_global)})', zorder=2)
    ax.scatter(X_pca[min_global, 0],   X_pca[min_global, 1],
               c='crimson', alpha=0.8, s=35,
               label=f'Minority ({n_final_min})', zorder=3)
    ax.set_title(
        f"Step 5: Final Balanced Training Set\n"
        f"Majority={n_final_maj}  Minority={n_final_min}  "
        f"(IR = {n_final_maj/max(n_final_min,1):.2f})",
        fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    _save(fig, f"{out}/{name}_step5_final.png")

    print(f"\nDone! Plots saved to '{out}/'")
    print(f"  Final set: Majority={n_final_maj}, Minority={n_final_min}, "
          f"IR={n_final_maj/max(n_final_min,1):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BOD stepwise visualiser (Peng & Park 2022)")
    p.add_argument("--dataset",     type=str, required=True)
    p.add_argument("--datadir",     type=str, default="./datasets")
    p.add_argument("--k",           type=int, default=5)
    p.add_argument("--min_samples", type=int, default=5)
    args = p.parse_args()
    run_stepwise_viz(args)
