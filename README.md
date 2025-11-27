# Anomaly Detection Repository

This repository contains multiple **unsupervised anomaly detection techniques** applied to various datasets. The goal is to **detect outliers or anomalies** in data using **clustering, model-based, and statistical approaches**.

The project is organized into three main directories:

* `clustering-based/` – Clustering-based anomaly detection
* `model-based/` – Model-based anomaly detection
* `statistical-methods/` – Statistical anomaly detection

Each directory contains notebooks with **full workflow**, including data preprocessing, model implementation, evaluation, visualization, and saving results.

---

## Table of Contents

1. [Clustering-Based Anomaly Detection](#clustering-based-anomaly-detection)
2. [Model-Based Anomaly Detection](#model-based-anomaly-detection)
3. [Statistical Methods for Anomaly Detection](#statistical-methods-for-anomaly-detection)
4. [Dependencies](#dependencies)

---

## Clustering-Based Anomaly Detection

This folder implements anomaly detection using **unsupervised clustering algorithms**.

### Project 1 — KMeans-Based Anomaly Detection

**Technique:**

* Form clusters of similar data points using KMeans
* Compute distance from cluster centroids
* Points in the top 5% distance percentile are marked as anomalies

**Dataset:**

* Mall Customer Segmentation Dataset (features: Annual Income, Spending Score)

**Steps Performed:**

1. Load dataset
2. Scale numeric features
3. Determine optimal cluster count using the Elbow Method
4. Fit KMeans (k=5)
5. Compute distances from centroids and determine anomaly threshold (95th percentile)
6. Mark anomalies
7. Visualize anomalies vs normal points

**Visualizations:**

* Elbow curve for optimal cluster selection
* Scatterplot showing anomalies (red) vs normal points (blue)

---

### Project 2 — DBSCAN-Based Anomaly Detection

**Technique:**

* DBSCAN detects dense regions as clusters and sparse regions as anomalies
* Works with irregular shapes and density

**Dataset:**

* UCI Wholesale Customers Dataset (features: spending on Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen)

**Steps Performed:**

1. Load and clean dataset
2. Scale features using `StandardScaler`
3. Plot K-distance graph to estimate `eps`
4. Apply DBSCAN (`eps ≈ 1.2`, `min_samples = 5`)
5. Label cluster = -1 as anomalies
6. Reduce to 2D using PCA
7. Visualize anomalies vs normal points

**Advantages of DBSCAN:**

* Automatically detects outliers
* Works for non-linear cluster shapes
* Handles irregular density patterns

**Next Projects:**

* Gaussian Mixture Model (GMM) Anomaly Detection

---

## Model-Based Anomaly Detection

This folder includes **unsupervised model-based techniques** applied to the **Credit Card Fraud Detection dataset (Kaggle)**.

**Dataset Statistics:**

| Metric          | Value     |
| --------------- | --------- |
| Total rows      | 284,807   |
| Fraud (Class=1) | 492       |
| Legit (Class=0) | 284,315   |
| Fraud Rate      | ≈ 0.1727% |

**Implemented Models:**

### 1. Isolation Forest

* **Goal:** Isolate anomalies in extremely imbalanced datasets

* **Steps:**

  1. Drop target column, standardize features
  2. Compute contamination from true fraud ratio
  3. Train `IsolationForest`
  4. Predict anomalies
  5. Evaluate with confusion matrix, classification report, PCA visualization
  6. Save results

* **Conclusion:** Provides a solid unsupervised baseline; useful when labels are scarce.

---

### 2. Local Outlier Factor (LOF)

* **Goal:** Detect anomalies using local density deviation

* **Steps:**

  1. Drop duplicates, separate features/labels, standardize features
  2. Fit `LocalOutlierFactor` with contamination
  3. Predict anomalies and compute LOF scores
  4. Evaluate with classification report and ROC-AUC
  5. Visualize LOF score distribution, PCA scatter, ROC curve
  6. Save results

* **Limitations:**

  * Fails on high-dimensional, extremely imbalanced data where anomalies are not density-based
  * Scores are training-only if `novelty=False`

* **Conclusion:** Useful only when anomalies are local density outliers.

---

## Statistical Methods for Anomaly Detection

This folder implements **classical statistical anomaly detection techniques**, including:

1. **Z-Score Based Detection** (`zscore_outlier_detection.ipynb`)

   * Detects points beyond a threshold number of standard deviations from the mean
2. **IQR-Based Detection** (`iqr_outlier_detection.ipynb`)

   * Detects points outside the interquartile range (1.5×IQR rule)
3. **Rolling Statistics for Time Series** (`rolling_stats_anomaly_detection.ipynb`)

   * Detects anomalies based on deviations from moving averages and rolling standard deviations

**Workflow:**

* Load and clean dataset
* Compute statistical metrics
* Apply thresholds to flag anomalies
* Visualize anomalies with plots
* Save results

**Advantages:**

* Simple and interpretable
* Fast for univariate and low-dimensional datasets
* Useful for exploratory analysis and baseline detection

---

## Dependencies

All projects rely on standard Python data science libraries:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

Additional dependencies per project may include:

* `scikit-learn` for clustering, isolation forest, LOF
* `PCA` for dimensionality reduction
* `StandardScaler` for feature scaling

---

## Summary

This repository provides a **comprehensive collection of anomaly detection approaches**, covering:

| Approach Type           | Techniques Implemented           | Use Case & Notes                                                                                                                  |
| ----------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Clustering-Based**    | KMeans, DBSCAN                   | Detect anomalies by distance from cluster centroids or density deviations; effective for multidimensional datasets with structure |
| **Model-Based**         | Isolation Forest, LOF            | Unsupervised detection in imbalanced datasets; Isolation Forest is robust, LOF shows limitations on high-dimensional rare events  |
| **Statistical Methods** | Z-Score, IQR, Rolling Statistics | Fast, interpretable, univariate or low-dimensional datasets; good for baselines                                                   |

This repository can serve as a **reference and hands-on practice resource** for understanding anomaly detection from multiple perspectives: clustering, model-based, and statistical approaches.
---
