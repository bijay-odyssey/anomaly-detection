# **Credit Card Fraud Detection Using Isolation Forest (Unsupervised Anomaly Detection)**

This project applies **Isolation Forest**, an unsupervised anomaly detection algorithm, on the popular **Credit Card Fraud Detection dataset (Kaggle)**.
The goal is to detect fraudulent transactions using only patterns in the data — without training on labels.

---

##  **Project Overview**

Credit card fraud is extremely rare and highly imbalanced in the dataset:

* **Total rows:** 284,807
* **Fraud (Class = 1):** 492
* **Legit (Class = 0):** 284,315
* **Fraud Rate:** ≈ 0.1727%

Since supervised learning struggles with extreme imbalance, we use **Isolation Forest**, which isolates anomalies by randomly partitioning the feature space.

---

##  **Dataset**

Source: Kaggle — *Credit Card Fraud Detection*
Features:

* PCA-transformed components: `V1` to `V28`
* `Time`
* `Amount`
* `Class` (ground-truth, only for evaluation — NOT used for training)

---

##  **Steps Performed**

### **1. Import Libraries**

NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn (IsolationForest, PCA), Metrics.

### **2. Load Dataset**

```python
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
```

### **3. Basic Exploration**

Checked datatypes, missing values, class distribution.

### **4. Preprocessing**

* Dropped target column
* Standardized features using `StandardScaler`

### **5. Compute Contamination**

Instead of manually guessing contamination, we use:

```python
fraud_ratio = df["Class"].mean()
```

This sets contamination ≈ 0.001727 (true fraud rate).

### **6. Train Isolation Forest**

```python
iso = IsolationForest(
    n_estimators=200,
    contamination=fraud_ratio,
    max_samples="auto",
    random_state=42,
)
iso.fit(X_scaled)
```

### **7. Prediction**

Isolation Forest predicts:

* `1` → normal
* `-1` → anomaly

Converted to:

* `0` → normal
* `1` → anomaly

Added anomaly scores and PCA components.

### **8. Evaluation (Using True Labels ONLY for Analysis)**

Confusion Matrix:

```
Predicted   0      1
Actual
0       283947   368
1          368   124
```

Classification Report:

* Precision (fraud): 0.25
* Recall (fraud): 0.25
* Accuracy: ~99.74% (not a good metric for imbalanced data)

### **9. Visualization**

* Anomaly Score Distribution
* PCA 2D Scatterplot of anomalies
* Crosstab comparison of fraud vs detected anomalies

### **10. Save Results**

```python
df.to_csv("isolation_forest_results.csv", index=False)
```

---


##  **Future Project**

- **Local Outlier Factor (LOF)**
- **One-Class SVM**
- Oversample anomalies using **SMOTE** for supervised models
- Add **threshold tuning** on anomaly scores

---

##  **Conclusion**

Isolation Forest provides a solid unsupervised baseline for anomaly detection.
It cannot match supervised ML in recall but is useful when labels are scarce or imbalanced.

---
