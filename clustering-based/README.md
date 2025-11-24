## **📂 Clustering-Based Anomaly Detection**

This folder contains anomaly detection projects using **unsupervised clustering algorithms**.
 

---

#  **Project 1 — KMeans-Based Anomaly Detection**

### **Technique**

KMeans is used to:

* Form clusters of similar customers
* Compute distance of each customer from its cluster center
* Data points far from the center are treated as anomalies

I defined anomalies as customers in the **top 5% distance percentile**.

---

# **Dataset**

**Mall Customer Segmentation Dataset**
Public dataset used widely for clustering analysis.
Features used:

* Annual Income (k$)
* Spending Score (1–100)

---

#  **Steps Performed**

1. Load dataset
2. Scale numeric features
3. Use Elbow Method to find optimal cluster count
4. Fit KMeans with k=5
5. Compute distance of each point from cluster centroid
6. Choose threshold = 95th percentile
7. Mark points above threshold as anomalies
8. Visualize anomalous vs normal points

---

#  **Visualization**

The generated notebook has:

* Elbow curve
* Scatterplot of anomalies (red) vs normal points (blue)
* Anomaly flags in the final dataset

---

## **Project 2 — DBSCAN-Based Anomaly Detection**

### **Technique**

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies:

* Dense regions → clusters
* Sparse regions → anomalies (label = -1)

It is excellent for anomaly detection because it **does not assume spherical clusters** like KMeans.

---

## **Dataset**

**UCI Wholesale Customers Dataset**
Features include yearly spending on:

* Fresh
* Milk
* Grocery
* Frozen
* Detergents_Paper
* Delicassen

We remove `Region` and `Channel` since they are categorical.

---

## **Steps Performed**

1. Load & clean dataset
2. Scale features using StandardScaler
3. Plot K-distance graph to estimate `eps`
4. Apply DBSCAN with `eps ≈ 1.2` and `min_samples = 5`
5. Label points with `cluster = -1` as anomalies
6. Reduce to 2D using PCA
7. Visualize anomalies (red) vs normal points (blue)

---

## **Why DBSCAN is Useful for Anomaly Detection?**

* Automatically detects outliers (no separate threshold needed)
* Works with any cluster shape (non-linear clusters)
* Good for irregular density patterns

---

#  **Next Projects (Coming Soon)**

* DBSCAN-based anomaly detection

 ---
 
 #  **Dependencies**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---
