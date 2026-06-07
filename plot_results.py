"""
Karşılaştırma sonuçlarını görselleştirir.
compare_methods.py çalıştırıldıktan sonra üretilen JSON dosyalarını okur.

Üretilen grafikler (result_images/comparison/ klasörüne kaydedilir):
  1. grouped_bar_<metrik>_<clf>.png  -- veri seti başına yöntem karşılaştırması
  2. heatmap_<metrik>_<clf>.png      -- yöntem × veri seti ısı haritası
  3. avg_summary_<clf>.png           -- tüm veri setleri ortalaması özet çubuğu
  4. combined_summary.png            -- SVM vs RF BOD karşılaştırması
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR     = os.path.join("result_images", "comparison")
MODEL_NAMES = ['Baseline', 'RUS', 'DBSCAN', 'SMOTE', 'BOD']
COLORS      = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
METRICS     = [
    ('aucs',   'AUC'),
    ('gmeans', 'G-Mean'),
    ('f1s',    'F1 (macro)'),
    ('mccs',   'MCC'),
]


def load_json(clf_label):
    path = f"results_{clf_label}.json"
    if not os.path.exists(path):
        print(f"[UYARI] {path} bulunamadı. Önce compare_methods.py çalıştırın.")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def grouped_bar(summaries, metric_key, metric_label, clf_label):
    """Her veri seti için yöntemleri karşılaştıran gruplu çubuk grafik."""
    datasets = [s['dataset'] for s in summaries]
    n_ds     = len(datasets)
    n_m      = len(MODEL_NAMES)
    x        = np.arange(n_ds)
    width    = 0.15

    fig, ax = plt.subplots(figsize=(max(14, n_ds * 1.2), 5))

    for i, (m, color) in enumerate(zip(MODEL_NAMES, COLORS)):
        vals = [s[metric_key][m] for s in summaries]
        offset = (i - n_m / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=m, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f'{metric_label} per Dataset — {clf_label}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.set_ylim(0, 1.12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, f"grouped_bar_{metric_key}_{clf_label}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"--> {path}")


def heatmap(summaries, metric_key, metric_label, clf_label):
    """Yöntem × veri seti ısı haritası."""
    datasets = [s['dataset'] for s in summaries]
    data = np.array([[s[metric_key][m] for s in summaries] for m in MODEL_NAMES])

    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 0.9), 4))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(MODEL_NAMES)))
    ax.set_yticklabels(MODEL_NAMES, fontsize=10)

    for i in range(len(MODEL_NAMES)):
        for j in range(len(datasets)):
            val = data[i, j]
            color = 'black' if 0.3 < val < 0.75 else 'white'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7.5, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(f'{metric_label} Heatmap — {clf_label}', fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(OUT_DIR, f"heatmap_{metric_key}_{clf_label}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"--> {path}")


def avg_summary(summaries, clf_label):
    """Tüm veri setleri üzerinden ortalama metrik özeti."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f'Average Performance Across All Datasets — {clf_label}',
                 fontsize=13, fontweight='bold')

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        avgs = [np.mean([s[metric_key][m] for s in summaries]) for m in MODEL_NAMES]
        bars = ax.bar(MODEL_NAMES, avgs, color=COLORS, alpha=0.85)
        for bar, v in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(metric_label, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(MODEL_NAMES, rotation=25, ha='right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"avg_summary_{clf_label}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"--> {path}")


def combined_summary(svm_summaries, rf_summaries):
    """SVM ile RF için BOD AUC karşılaştırması yan yana."""
    datasets   = [s['dataset'] for s in svm_summaries]
    svm_bod    = [s['aucs']['BOD']      for s in svm_summaries]
    svm_base   = [s['aucs']['Baseline'] for s in svm_summaries]
    rf_bod     = [s['aucs']['BOD']      for s in rf_summaries]
    rf_base    = [s['aucs']['Baseline'] for s in rf_summaries]

    x     = np.arange(len(datasets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(14, len(datasets) * 1.2), 5))
    ax.bar(x - 1.5*width, svm_base, width, label='SVM Baseline',    color='#4C72B0', alpha=0.75)
    ax.bar(x - 0.5*width, svm_bod,  width, label='SVM + BOD',       color='#8172B3', alpha=0.9)
    ax.bar(x + 0.5*width, rf_base,  width, label='RF Baseline',     color='#DD8452', alpha=0.75)
    ax.bar(x + 1.5*width, rf_bod,   width, label='RF + BOD',        color='#55A868', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('BOD vs Baseline — SVM & Random Forest Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "combined_summary.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"--> {path}")


def rank_heatmap(summaries, clf_label):
    """Yöntemlerin her veri setindeki AUC sırasını gösteren ısı haritası."""
    datasets = [s['dataset'] for s in summaries]
    auc_data = np.array([[s['aucs'][m] for m in MODEL_NAMES] for s in summaries])
    ranks    = np.argsort(np.argsort(-auc_data, axis=1), axis=1) + 1  # 1=en iyi

    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 0.9), 4))
    im = ax.imshow(ranks.T, aspect='auto', cmap='RdYlGn_r', vmin=1, vmax=len(MODEL_NAMES))

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(MODEL_NAMES)))
    ax.set_yticklabels(MODEL_NAMES, fontsize=10)

    for i in range(len(MODEL_NAMES)):
        for j in range(len(datasets)):
            r = ranks[j, i]
            color = 'white' if r >= 4 else 'black'
            ax.text(j, i, str(r), ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Rank (1=best)')
    ax.set_title(f'AUC Rankings per Dataset — {clf_label}', fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(OUT_DIR, f"rank_heatmap_{clf_label}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"--> {path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    svm_data = load_json("SVM")
    rf_data  = load_json("RandomForest")

    for summaries, clf_label in [(svm_data, "SVM"), (rf_data, "RandomForest")]:
        if summaries is None:
            continue
        print(f"\n{'='*60}\nGrafik üretiliyor: {clf_label}\n{'='*60}")

        for metric_key, metric_label in METRICS:
            if metric_key not in summaries[0]:
                print(f"  [ATLA] {metric_key} bu JSON'da yok.")
                continue
            grouped_bar(summaries, metric_key, metric_label, clf_label)
            heatmap(summaries, metric_key, metric_label, clf_label)

        avg_summary(summaries, clf_label)
        rank_heatmap(summaries, clf_label)

    if svm_data and rf_data:
        print(f"\n{'='*60}\nSVM vs RF karşılaştırma grafiği\n{'='*60}")
        combined_summary(svm_data, rf_data)

    print(f"\nTüm grafikler '{OUT_DIR}/' klasörüne kaydedildi.")
