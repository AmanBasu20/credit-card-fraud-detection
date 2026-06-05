# Credit Card Fraud Detection

> A rigorous end-to-end machine learning pipeline for detecting fraudulent credit card transactions on a severely imbalanced dataset (0.172% fraud prevalence). Built to demonstrate production-aware ML engineering practices for research internship applications.

---

## 🏆 Key Results

| Model | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | **AUPRC** |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.0563 | 0.9184 | 0.1061 | 0.7213 |
| MLP Neural Network (Keras) | 0.5629 | 0.8673 | 0.6827 | 0.8325 |
| **XGBoost (Best)** | **0.4915** | **0.8878** | **0.6327** | **0.8594** |

> **Reading the results:** Logistic Regression achieves high recall (0.9184) but catastrophically low precision (0.0563) — it flags nearly every transaction as fraud. XGBoost achieves the best AUPRC (0.8594), reflecting the most favorable precision-recall tradeoff across all decision thresholds, which is the operationally correct optimization target for this problem.

---

## 📌 Achievements (XYZ Format)

- **Engineered** a preprocessing pipeline (feature dropping, `StandardScaler` normalization) across 30 input dimensions, eliminating feature-scale asymmetry that causes elongated loss-surface geometry in gradient-based optimizers — a prerequisite for stable Logistic Regression and MLP convergence.

- **Eliminated data leakage** by applying SMOTE strictly within the training partition, synthesizing minority-class vectors via k-NN interpolation in feature space to achieve a balanced 1:1 class ratio without contaminating the 56,962-sample held-out evaluation set.

- **Benchmarked three architecturally distinct classifiers** (Logistic Regression, MLP, XGBoost) on a stratified test set, achieving a top AUPRC of **0.8594** with XGBoost — approximately **499× above the random classifier AUPRC baseline of 0.00172**.

- **Demonstrated the precision-recall tradeoff empirically**: Logistic Regression's recall of 0.9184 came at the cost of precision collapsing to 0.0563 (1 true fraud flagged per ~17 alerts), while XGBoost's AUPRC of 0.8594 reflects a substantially more deployable operating curve with precision of 0.4915 at recall 0.8878.

- **Identified AUPRC as the operationally correct evaluation metric** over ROC-AUC: with 56,864 legitimate transactions in the test set, the FPR denominator is large enough that thousands of false alarms still produce near-zero FPR, masking the real cost of false alerts that AUPRC's Precision term penalizes directly.

- **Implemented early stopping and Dropout regularization** (p=0.3) in the Keras MLP, achieving an AUPRC of 0.8325 and fraud F1 of 0.6827 — a **6.4× improvement in fraud F1** over the Logistic Regression baseline (0.1061), demonstrating the value of non-linear capacity on this PCA-transformed feature space.

---

## 🗂️ Project Structure

```
credit_card_fraud_detection/
├── credit_card_fraud_detection.ipynb  # Main Jupyter Notebook
├── precision_recall_curve.png      # Comparative PR curve visualization
└── README.md
```

---

## ⚙️ Pipeline Architecture

```
Raw Data (creditcard.csv)  [284,807 rows × 31 cols]
        │
        ▼
  Drop 'Time' column
  Scale 'Amount' via StandardScaler
        │
        ▼
  Stratified 80/20 Train/Test Split
  [Train: 227,845 | Test: 56,962 | Fraud rate preserved at 0.172%]
        │
        ▼
  SMOTE on X_train only  ◄── [Leakage prevention boundary]
  [Balanced: ~227,451 Legit + ~227,451 Fraud]
        │
        ▼
  ┌──────────────────────────────────────────────┐
  │  Logistic Regression  →  AUPRC: 0.7213       │
  │  MLP Neural Network   →  AUPRC: 0.8325       │
  │  XGBoost Classifier   →  AUPRC: 0.8594  ✓   │
  └──────────────────────────────────────────────┘
        │
        ▼
  Evaluation: Precision / Recall / F1 / AUPRC
  Precision-Recall Curve (comparative, all 3 models)
```

---

## 🔬 Technical Design Decisions

**Why drop `Time`?** The feature encodes elapsed seconds from the dataset's first transaction — a dataset-internal artifact with no causal relationship to fraud that would generalize to unseen data.

**Why SMOTE over class weighting?** Class weights adjust the loss function but do not alter the geometry of the training distribution. SMOTE synthesizes new points on the minority class manifold via k-NN interpolation, giving tree ensembles and neural networks richer decision-boundary signal in the fraud subspace.

**Why AUPRC over ROC-AUC?** With 56,864 legitimate transactions in the test partition alone, the FPR denominator is so large that even thousands of false alarms produce near-zero FPR. AUPRC's Precision term penalizes every false positive directly, providing an operationally honest view of model quality. A random classifier's AUPRC on this dataset equals the fraud prevalence ≈ **0.00172**; our best model achieves **0.8594**.

**Why does XGBoost win on AUPRC but not on Fraud-F1?** AUPRC integrates performance across *all* decision thresholds. XGBoost's probability calibration across the full curve is superior. At the default 0.5 threshold, MLP's fraud F1 (0.6827) edges XGBoost (0.6327) — but with threshold tuning toward recall parity, XGBoost's curve dominates. In production, the operating threshold is always tuned to business cost constraints, making AUPRC the correct selection criterion.

---

## 🚀 Reproducing Results

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost tensorflow matplotlib

# 2. Place creditcard.csv in the project root
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 3. Launch notebook
jupyter notebook fraud_detection_pipeline.ipynb
```
