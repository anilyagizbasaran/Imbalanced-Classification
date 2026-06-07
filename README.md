# Imbalanced Classification with BOD Pipeline

A comparative study of resampling methods for imbalanced binary classification. Five methods are evaluated across 12 benchmark datasets using two classifiers and four performance metrics.

---

## Methods Compared

| Method | Type | Description |
|---|---|---|
| Baseline | — | No resampling, train directly |
| RUS | Under-sampling | Randomly removes majority samples |
| DBSCAN | Under-sampling | Removes majority samples based on density clustering |
| SMOTE | Over-sampling | Generates synthetic minority samples |
| **BOD** | Under-sampling | Removes borderline noise + outliers, protects rare samples |

**BOD** (Borderline Outlier Detection) is the main proposed method. It cleans the majority class in four stages:

```
1. BNF  — removes majority samples causing borderline noise (AUC-guided)
2. OBN  — removes isolated majority outliers
3. DBSCAN — labels remaining majority as Safe (removable) or Rare (protected)
4. RUS  — undersamples Safe majority down to minority class size
```

---

## Evaluation Setup

- **Classifiers:** SVM (RBF kernel) and Random Forest (100 trees), both cost-sensitive
- **Metrics:** AUC, G-mean, F1-score (macro), MCC
- **Validation:** 5-fold stratified cross-validation
- **Statistical test:** Wilcoxon signed-rank test (BOD vs each baseline)
- **DBSCAN eps:** Adaptive per dataset via k-distance elbow method

---

## Datasets

12 KEEL benchmark datasets spanning imbalance ratios from 1.82 to 41.40:

| Dataset | Samples | IR | Dataset | Samples | IR |
|---|---|---|---|---|---|
| glass1 | 214 | 1.82 | glass0146vs2 | 205 | 11.06 |
| yeast1 | 1484 | 2.46 | yeast1vs7 | 459 | 14.30 |
| haberman | 306 | 2.78 | glass4 | 214 | 15.46 |
| ecoli1 | 336 | 3.36 | yeast5 | 1484 | 32.73 |
| segment0 | 2308 | 6.02 | yeast6 | 1484 | 41.40 |
| glass6 | 214 | 6.38 | yeast2vs4 | 514 | 9.08 |

---

## Results

BOD is most effective at **high imbalance ratios (IR ≥ 14)**.

| Classifier | BOD beats Baseline | Best gain |
|---|---|---|
| SVM | 2 / 12 datasets | +0.019 (yeast6, IR = 41.40) |
| Random Forest | 4 / 12 datasets | +0.043 (yeast6, IR = 41.40) |

Full per-dataset results in `results_SVM.txt` and `results_RandomForest.txt`.

---

## Project Structure

```
├── BNF.py                 # Borderline Noise Factor removal
├── OBN.py                 # Outlier removal via neighborhood scoring
├── DBSCAN.py              # Density clustering + adaptive eps
├── RUS.py                 # Random Under-Sampling
├── SMOTE.py               # SMOTE wrapper
├── keel_utils.py          # KEEL .dat dataset loader
├── compare_methods.py     # Main experiment runner
├── plot_results.py        # Comparison chart generator
├── stepwise_visualizer.py # BOD pipeline step-by-step PCA plots
├── generate_all_plots.py  # Batch visualization for all datasets
├── datasets/              # 12 KEEL .dat files
└── result_images/         # All generated plots
```

---

## How to Run

```bash
pip install numpy scikit-learn imbalanced-learn matplotlib scipy

# Run all experiments
python compare_methods.py

# Generate comparison charts (bar charts, heatmaps, ranking plots)
python plot_results.py

# Generate BOD pipeline step-by-step visualizations
python generate_all_plots.py
```

---

## References

Peng, M., & Park, S. (2022). *A New Hybrid Under-sampling Approach to Imbalanced Classification Problems.* Turkish Journal of Electrical Engineering and Computer Sciences.
