# Model-Based Anomaly Detection

This repository contains **unsupervised model-based anomaly detection techniques** applied to the **Credit Card Fraud Detection** dataset (Kaggle). It includes implementations of:

* **Isolation Forest**
* **Local Outlier Factor (LOF)**
* (Future additions: One-Class SVM)

The goal is to detect fraudulent credit card transactions **without using labels for training**, highlighting both algorithm workflows and practical limitations.

---

## Dataset Overview

**Source:** Kaggle — Credit Card Fraud Detection

**Dataset statistics:**

| Metric          | Value     |
| --------------- | --------- |
| Total rows      | 284,807   |
| Fraud (Class=1) | 492       |
| Legit (Class=0) | 284,315   |
| Fraud Rate      | ≈ 0.1727% |

**Features include:**

* PCA-transformed components: V1–V28
* Time
* Amount
* Class (ground truth; only used for evaluation)

Fraud is extremely rare, making **supervised learning challenging** and motivating unsupervised anomaly detection methods.

---

## 1. Isolation Forest

**Isolation Forest** isolates anomalies by recursively partitioning the feature space. It is effective for datasets with extreme class imbalance.

### Steps Performed

1. **Import Libraries** – NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn (`IsolationForest`, PCA), metrics.
2. **Load Dataset**

```python
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
```

3. **Basic Exploration** – Checked datatypes, missing values, class distribution.
4. **Preprocessing** – Dropped target column and standardized features using `StandardScaler`.
5. **Compute Contamination** – Used the true fraud ratio:

```python
fraud_ratio = df["Class"].mean()  # ≈ 0.001727
```

6. **Train Isolation Forest**

```python
iso = IsolationForest(
    n_estimators=200,
    contamination=fraud_ratio,
    max_samples="auto",
    random_state=42
)
iso.fit(X_scaled)
```

7. **Prediction** – Converted model output to binary anomaly labels:

* 1 → anomaly
* 0 → normal

8. **Evaluation** (using true labels only for analysis)

* Confusion matrix
* Classification report: Precision & Recall (fraud = 0.25)
* Accuracy ≈ 99.74% (not meaningful due to imbalance)

9. **Visualization**

* Anomaly score distribution
* PCA 2D scatter of anomalies
* Crosstab comparison of fraud vs detected anomalies

10. **Save Results**

```python
df.to_csv("isolation_forest_results.csv", index=False)
```

**Conclusion:**
Isolation Forest provides a strong **unsupervised baseline**. While recall is limited, it is useful when labels are scarce or imbalanced.

---

## 2. Local Outlier Factor (LOF)

**LOF** detects anomalies based on **local density deviations** relative to neighbors.

### Steps Performed

1. **Import Libraries** – NumPy, Pandas, Matplotlib, Scikit-learn (`LocalOutlierFactor`, PCA, metrics), StandardScaler.
2. **Load Dataset**

```python
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
```

3. **Preprocessing**

* Dropped duplicates (important for distance-based calculations)
* Separated features (`X`) and labels (`y`)
* Standardized features with `StandardScaler`

4. **Estimate Contamination**

```python
contamination = y.mean()  # ≈ 0.001667
```

5. **Fit LOF Model**

```python
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination,
    novelty=False,
    metric="euclidean"
)
y_pred = lof.fit_predict(X_scaled)
y_pred_binary = (y_pred == -1).astype(int)
```

6. **Anomaly Scores**

```
scores = -lof.negative_outlier_factor_
df_results["lof_score"] = scores
df_results["predicted_anomaly"] = y_pred_binary
```

7. **Evaluation** (true labels only)

* Classification report
* ROC-AUC: 0.51
* Confirms LOF fails to detect fraud due to density assumptions

8. **Visualization**

* LOF score histogram
* PCA 2D scatter with LOF scores
* ROC curve

9. **Save Results**

```python
df_results.to_csv("lof_results.csv", index=False)
```

### Why LOF Performs Poorly

* **Fraud is not a density anomaly** – anomalies are not sparse; they overlap normal clusters.
* **High dimensionality** – distance metrics become less informative in 30+ dimensions.
* **Extreme imbalance** – contamination rate approximates true fraud, but density structure fails.
* **novelty=False** – LOF scores are training-only, limiting predictive power.

**Conclusion:**
LOF is effective **only when anomalies are local density outliers**. On this dataset, it performs near random, demonstrating the importance of algorithm choice for the problem type.

---

## Summary

| Model                | Use Case & Observations                                                                                                                             |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Isolation Forest** | Performs well as an unsupervised baseline; isolates anomalies effectively; limited recall on fraud.                                                 |
| **LOF**              | Fails on this dataset due to high dimensionality, overlapping anomalies, and density assumptions; educational for understanding method limitations. |

These notebooks provide a **comprehensive workflow** for model-based anomaly detection, including preprocessing, model fitting, evaluation, visualization, and saving results.
