import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

df = pd.read_csv("combined_landmarks_no_label.csv")

# Extract true labels from the image name
y_true = df.iloc[:, 0].str.split("_").str[0].values

# Extract landmark feature data
X = df.iloc[:, 1:].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use best found K value for agglomerative clustering
k = 19

agg = AgglomerativeClustering(
    n_clusters=k,
    metric="euclidean",
    linkage="ward"
)

labels = agg.fit_predict(X_scaled)
ari = adjusted_rand_score(y_true, labels)
print("ARI score: ", ari)

confusion_table = pd.crosstab(
    labels,
    y_true,
    rownames=["Cluster"],
    colnames=["True Label"]
)

# Plot confusion table normalized by cluster to see which clusters contain instances from which labels
confusion_norm = confusion_table.div(confusion_table.sum(axis=1), axis=0)

plt.figure(figsize=(12, 6))
plt.imshow(confusion_norm, aspect="auto")
plt.colorbar(label="Proportion")
plt.xlabel("True Hand Sign")
plt.ylabel("Cluster")
plt.title("Confusion-style Cluster Analysis (Agglomerative)")
plt.xticks(range(len(confusion_norm.columns)), confusion_norm.columns, rotation=90)
plt.yticks(range(len(confusion_norm.index)), confusion_norm.index)
plt.tight_layout()
plt.show()

# Plot confusion table normalized by label to see how labels are broken into clusters
confusion_label_norm = confusion_table.div(confusion_table.sum(axis=0), axis=1)

plt.figure(figsize=(14, 6))
plt.imshow(confusion_label_norm, aspect="auto")
plt.colorbar(label="Proportion of samples")
plt.xlabel("True Hand Sign")
plt.ylabel("Cluster")
plt.title("Confusion-style Cluster Analysis (Normalized by Label)")
plt.xticks(range(len(confusion_label_norm.columns)), confusion_label_norm.columns, rotation=90)
plt.yticks(range(len(confusion_label_norm.index)), confusion_label_norm.index)
plt.tight_layout()
plt.show()


