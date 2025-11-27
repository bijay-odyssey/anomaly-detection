# Statistical Methods for Anomaly Detection

This directory contains a collection of classical, interpretable, and production-ready **statistical anomaly detection techniques**. These approaches require no machine learning models and are widely used in applications where transparency, consistency, and computational efficiency are essential.

The notebooks provided here demonstrate how to apply these methods to real-world time-series data using the Beijing PM2.5 Air Quality dataset.

---

## Directory Contents

### 1. Z-Score Based Outlier Detection

Identifies anomalies by measuring how many standard deviations a value deviates from the global mean.
Useful when the underlying data distribution is approximately normal.
**File:**
`zscore_outlier_detection.ipynb`

---

### 2. IQR (Interquartile Range) Outlier Detection

A distribution-independent method using Q1, Q3, and IQR to flag extreme deviations.
Effective for skewed, heavy-tailed, or non-normally distributed datasets.
**File:**
`iqr_outlier_detection.ipynb`

---

### 3. Rolling Statistics Anomaly Detection

A time-series focused technique using rolling mean and rolling standard deviation to identify local anomalies.
Suited for data exhibiting trends, seasonality, or structural shifts.
**File:**
`rolling_stats_anomaly_detection.ipynb`

---

## Dataset

All notebooks use the **Beijing PM2.5 Air Quality Dataset** (Kaggle).
This dataset provides hourly air-quality measurements across multiple years, making it ideal for demonstrating statistical anomaly detection due to its natural variability, periodic patterns, and occasional pollution spikes.

---

## Key Features

Each notebook includes:

* Full preprocessing and cleaning steps
* Computation of statistical thresholds
* Detection and labeling of anomalies
* Clear visualizations for diagnostic validation
* Lightweight, production-friendly code
* Methods that generalize well to operational monitoring systems

---

## Purpose of This Module

This folder is part of a broader anomaly detection repository. 
The statistical-methods module provides foundational baselines and interpretable approaches suitable for univariate or time-dependent data, and for environments where explainability is a priority.

---

## How to Navigate

```
statistical-methods/
│── zscore_outlier_detection.ipynb
│── iqr_outlier_detection.ipynb
│── rolling_stats_anomaly_detection.ipynb
```

Each notebook is stand-alone and includes all required logic, plots, and evaluation.

---
