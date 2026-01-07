import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from matplotlib import pyplot as plt

df = pd.read_csv("combined_landmarks_no_label.csv")

y_true = df.iloc[:, 0].str.split("_").str[0].values

ids = df.iloc[:, 0]          # image names
X = df.iloc[:, 1:].values    # landmark features

# Scale features to ensure uniformity in distances
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = range(2, 21)
k_ari_scores = []
agg_ari_scores = []

# Run K-Means and Agglomerative clustering for K values 2...21
# Evaluate using adjusted rand index (ari)
for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    agg = AgglomerativeClustering(
        n_clusters=k,
        metric="euclidean",
        linkage="ward"
    )
    k_labels = kmeans.fit_predict(X_scaled)
    agg_labels = agg.fit_predict(X_scaled)

    k_ari = adjusted_rand_score(y_true, k_labels)
    k_ari_scores.append(k_ari)

    agg_ari = adjusted_rand_score(y_true, agg_labels)
    agg_ari_scores.append(agg_ari)

# Plot the results of K vs ARI score
plt.figure()
plt.plot(k_values, k_ari_scores, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Adjusted Rand Index (ARI)")
plt.title("K-means: Effect of K on ARI Score for K-Means Clustering")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(k_values, agg_ari_scores, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Adjusted Rand Index (ARI)")
plt.title("K-means: Effect of K on ARI Score for Agglomerative Clustering")
plt.grid(True)
plt.show()

# Find best K value for both clustering methods
best_k = k_values[np.argmax(k_ari_scores)]
best_ari = max(k_ari_scores)

print("K-Means Clustering: ")
print(f"Best K: {best_k}")
print(f"Best ARI: {best_ari:.4f}")

best_k = k_values[np.argmax(agg_ari_scores)]
best_ari = max(agg_ari_scores)

print("Agglomerative Clustering: ")
print(f"Best K: {best_k}")
print(f"Best ARI: {best_ari:.4f}")

pd.crosstab(agg_labels, y_true)






