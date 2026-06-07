# BOD: Borderline Outlier Detection for Imbalanced Classification

Python implementation of the **BOD (Borderline Outlier Detection)** under-sampling pipeline proposed in:

> Peng, M., & Park, S. (2022). *A New Hybrid Under-sampling Approach to Imbalanced Classification Problems.* Turk J Elec Eng & Comp Sci.

---

## Overview

BOD is a four-stage majority-class cleaning pipeline designed for imbalanced binary classification. Instead of blindly removing random majority samples, it identifies and removes *harmful* samples (borderline noise and outliers) while protecting structurally important ones (rare/isolated samples).

### Pipeline

```
Raw data
   |
   v
[Step 1] BNF  -- iteratively removes borderline majority noise (AUC-guided)
   |
   v
[Step 2] OBN  -- removes isolated majority outliers (OBN score > mean)
   |
   v
[Step 3] DBSCAN -- categorises remaining majority into Safe / Rare
   |
   v
[Step 4] RUS  -- random under-samples Safe majority to n_minority (IR = 1)
   |
   v
Final training set: RUS(Safe) + Rare + Minority
```

---

## Project Structure

```
gardprject/
├── BNF.py                  # Borderline Noise Factor (Yang & Gao 2013)
├── OBN.py                  # Outlierness Based on Neighborhood (Gupta et al. 2019)
├── DBSCAN.py               # DBSCAN clustering with Safe/Borderline/Rare/Outlier categories
├── RUS.py                  # Random Under-Sampling
├── SMOTE.py                # SMOTE wrapper (used in comparison baseline only)
├── keel_utils.py           # KEEL .dat file loader
├── compare_methods.py      # 5-method comparison across 12 datasets
├── stepwise_visualizer.py  # Per-dataset step-by-step PCA plots
├── generate_all_plots.py   # Batch plot generation for all datasets
├── datasets/               # 12 KEEL .dat files
└── result_images/          # Generated plots (plots_<dataset>/)
```

---

## Requirements

```
numpy>=1.19.0
scikit-learn>=0.24.0
imbalanced-learn>=0.8.0
matplotlib>=3.3.0
```

```bash
pip install numpy scikit-learn imbalanced-learn matplotlib
```

---

## Usage

### Run full comparison (all 12 datasets)

```bash
python compare_methods.py
```

Outputs `all_results_summary.txt` with per-fold AUC, mean ± std, G-mean, and rankings for five methods: Baseline, RUS, DBSCAN, SMOTE, BOD.

### Generate step-by-step visualizations

```bash
python generate_all_plots.py
```

Plots are saved to `result_images/plots_<dataset>/`.

Or for a single dataset:

```bash
python stepwise_visualizer.py --dataset ecoli1
```

---

## Datasets

12 KEEL benchmark datasets covering a wide range of imbalance ratios:

| Dataset | Samples | Features | Majority | Minority | IR |
|---|---|---|---|---|---|
| glass1 | 214 | 9 | 138 | 76 | 1.82 |
| yeast1 | 1484 | 8 | 1055 | 429 | 2.46 |
| haberman | 306 | 3 | 225 | 81 | 2.78 |
| ecoli1 | 336 | 7 | 259 | 77 | 3.36 |
| segment0 | 2308 | 19 | 1979 | 329 | 6.02 |
| glass6 | 214 | 9 | 185 | 29 | 6.38 |
| yeast2vs4 | 514 | 8 | 463 | 51 | 9.08 |
| glass0146vs2 | 205 | 9 | 188 | 17 | 11.06 |
| yeast1vs7 | 459 | 7 | 429 | 30 | 14.30 |
| glass4 | 214 | 9 | 201 | 13 | 15.46 |
| yeast5 | 1484 | 8 | 1440 | 44 | 32.73 |
| yeast6 | 1484 | 8 | 1449 | 35 | 41.40 |

---

## Step-by-Step Visualizations

Each dataset produces 6 PCA projection plots showing the pipeline state at each stage. Examples below.

### ecoli1 (IR = 3.36)

| Step 0: Raw Data | Step 1: BNF | Step 2: OBN |
|:---:|:---:|:---:|
| ![](result_images/plots_ecoli1/ecoli1_step0_raw.png) | ![](result_images/plots_ecoli1/ecoli1_step1_bnf.png) | ![](result_images/plots_ecoli1/ecoli1_step2_obn.png) |

| Step 3: DBSCAN | Step 4: RUS | Step 5: Final |
|:---:|:---:|:---:|
| ![](result_images/plots_ecoli1/ecoli1_step3_dbscan.png) | ![](result_images/plots_ecoli1/ecoli1_step4_rus.png) | ![](result_images/plots_ecoli1/ecoli1_step5_final.png) |

---

### yeast1vs7 (IR = 14.30)

| Step 0: Raw Data | Step 1: BNF | Step 2: OBN |
|:---:|:---:|:---:|
| ![](result_images/plots_yeast1vs7/yeast1vs7_step0_raw.png) | ![](result_images/plots_yeast1vs7/yeast1vs7_step1_bnf.png) | ![](result_images/plots_yeast1vs7/yeast1vs7_step2_obn.png) |

| Step 3: DBSCAN | Step 4: RUS | Step 5: Final |
|:---:|:---:|:---:|
| ![](result_images/plots_yeast1vs7/yeast1vs7_step3_dbscan.png) | ![](result_images/plots_yeast1vs7/yeast1vs7_step4_rus.png) | ![](result_images/plots_yeast1vs7/yeast1vs7_step5_final.png) |

---

### yeast6 (IR = 41.40)

| Step 0: Raw Data | Step 1: BNF | Step 2: OBN |
|:---:|:---:|:---:|
| ![](result_images/plots_yeast6/yeast6_step0_raw.png) | ![](result_images/plots_yeast6/yeast6_step1_bnf.png) | ![](result_images/plots_yeast6/yeast6_step2_obn.png) |

| Step 3: DBSCAN | Step 4: RUS | Step 5: Final |
|:---:|:---:|:---:|
| ![](result_images/plots_yeast6/yeast6_step3_dbscan.png) | ![](result_images/plots_yeast6/yeast6_step4_rus.png) | ![](result_images/plots_yeast6/yeast6_step5_final.png) |

---

## Experimental Results

Classifier: SVM (RBF kernel, C=1, gamma='scale') with StandardScaler.
Evaluation: 5-fold stratified cross-validation (random_state=42).

### BOD vs Baseline AUC (sorted by IR)

| Dataset | IR | Baseline | BOD | Gain | Result |
|---|---|---|---|---|---|
| glass1 | 1.82 | 0.7572 | 0.7578 | +0.0005 | SUCCESS |
| yeast1 | 2.46 | 0.7832 | 0.7813 | -0.0019 | -- |
| haberman | 2.78 | 0.7015 | 0.7103 | +0.0088 | SUCCESS |
| ecoli1 | 3.36 | 0.9490 | 0.9429 | -0.0062 | -- |
| segment0 | 6.02 | 0.9997 | 0.9993 | -0.0004 | -- |
| glass6 | 6.38 | 0.9883 | 0.9793 | -0.0090 | -- |
| yeast2vs4 | 9.08 | 0.9812 | 0.9743 | -0.0069 | -- |
| glass0146vs2 | 11.06 | 0.6632 | 0.7021 | +0.0389 | SUCCESS |
| yeast1vs7 | 14.30 | 0.6830 | 0.7967 | +0.1136 | SUCCESS |
| glass4 | 15.46 | 0.9868 | 0.9635 | -0.0233 | -- |
| yeast5 | 32.73 | 0.9798 | 0.9823 | +0.0025 | SUCCESS |
| yeast6 | 41.40 | 0.8438 | 0.9374 | +0.0936 | SUCCESS |

**BOD succeeds on 6/12 datasets.** Gains are most pronounced at high imbalance ratios (IR >= 14), consistent with the paper's findings.

### Win count (best AUC per dataset)

| Method | Wins |
|---|---|
| Baseline | 4 |
| SMOTE | 4 |
| RUS | 3 |
| BOD | 1 |
| DBSCAN | 0 |

> Note: Exact numerical values differ from the paper due to SVM hyperparameter differences. The paper uses optimized C and gamma (Liu et al. 2015); this implementation uses sklearn defaults. The algorithmic pipeline is identical to the paper's description.

---

## Algorithm Details

### BNF (Borderline Noise Factor)

Identifies majority samples in the overlap region (SOVR) and iteratively removes the one with the highest BNF score if removal improves validation AUC:

```
BNF(x) = alpha * (Ks + delta) / (|kNS(x)| + delta) + beta * |kND(x)|
alpha=0.3, beta=0.7, delta=0.5, Ks=k
```

SOVR is built by scanning all k-NN relationships: if sample `xtrn` is a neighbor of an opposite-class sample `xtr`, then `xtrn` enters SOVR.

### OBN (Outlierness Based on Neighborhood)

Scores each majority sample by how isolated it is relative to its neighbors:

```
OBN(xi) = W[Nr(xi)] / sum_{yj in Nr(xi)} W[Nr(yj)]
W[Nr(xi)] = sum of k-NN distances from xi
```

Samples with `OBN(xi) > mean(OBN)` are removed.

### DBSCAN Categories

| Category | Condition | Action |
|---|---|---|
| SAFE (0) | core point (in cluster) | apply RUS |
| BORDERLINE (1) | border point (in cluster, not core) | apply RUS |
| RARE (2) | noise point with at least one neighbor within eps | protect |
| OUTLIER (3) | fully isolated noise point | protect |

`eps = 0.5 * sqrt(n_features)`

---

## License

Provided as-is for research and educational purposes.
