## **📂 Clustering-Based Anomaly Detection**

This folder contains anomaly detection projects using **unsupervised clustering algorithms**.
 

---

#  **Project 1 — KMeans-Based Anomaly Detection**

### ** Technique**

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

#  **Visualization **

The generated notebook has:

* Elbow curve
* Scatterplot of anomalies (red) vs normal points (blue)
* Anomaly flags in the final dataset

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

#  **Next Projects (Coming Soon)**

* DBSCAN-based anomaly detection
* Gaussian Mixture Models (GMM) Anomaly Detection

 ---
