from knn import KNNClassifier
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from matplotlib import pyplot as plt

df = pd.read_csv("C:/Users/andyd/Downloads/combined_landmarks_clean.csv")

# Prepare features and labels
X = df.drop(columns=["image_name", "label"]).values
y = df["label"].values

# Split dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# Train KNN classifier on training set and evaluate on test set
# ----------------------------------
# knn = KNNClassifier(k=3, distance="euclidean", weighted=True)
# knn.fit(X_train, y_train)

# test_preds = knn.predict(X_test)

# print("Test accuracy:", accuracy_score(y_test, test_preds))
# print(
#     "Test sensitivity:",
#     recall_score(y_test, test_preds, average="macro", zero_division=0)
# )
# ----------------------------------

# Plot confusion matrix
# ----------------------------------
# class_names = np.unique(y_train)

# cm_test = confusion_matrix(
#     y_test,
#     test_preds,
#     labels=class_names
# )

# cm_norm = cm_test.astype(float) / cm_test.sum(axis=1, keepdims=True)

# def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
#     plt.figure(figsize=(8, 6))
#     plt.imshow(cm)
#     plt.title(title)
#     plt.colorbar()

#     tick_marks = np.arange(len(labels))
#     plt.xticks(tick_marks, labels, rotation=45)
#     plt.yticks(tick_marks, labels)

#     plt.xlabel("Predicted label")
#     plt.ylabel("True label")

#     # Add text annotations
#     for i in range(len(labels)):
#         for j in range(len(labels)):
#             value = cm[i, j]
#             plt.text(
#                 j, i,
#                 f"{value:.2f}",
#                 ha="center",
#                 va="center",
#                 color="white" if value > 0.5 else "black"
#             )

#     plt.tight_layout()
#     plt.show()

# plot_confusion_matrix(
#     cm_norm,
#     class_names,
#     title="Normalized Confusion Matrix (Test Set)"
# )
# ----------------------------------

# 5-FOLD Cross-validation function to compare different k and weighted options
# ----------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_score(X, y, k, weighted):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = []
    recall_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        knn = KNNClassifier(
            k=k,
            distance="euclidean",
            weighted=weighted
        )

        knn.fit(X_train.tolist(), y_train.tolist())

        preds = knn.predict(X_val.tolist())

        acc_scores.append(accuracy_score(y_val, preds))
        recall_scores.append(
            recall_score(y_val, preds, average="macro", zero_division=0)
        )

    return {
        "accuracy": np.mean(acc_scores),
        "sensitivity": np.mean(recall_scores)
    }


results = []

for k in [3, 5, 7, 9, 11, 15]:
    for weighted in [False, True]:
        metrics = cv_score(X_train, y_train, k, weighted)
        results.append((k, weighted, metrics))

        print(
            f"k={k}, weighted={weighted} | "
            f"accuracy={metrics['accuracy']:.3f}, "
            f"sensitivity={metrics['sensitivity']:.3f}"
        )

# Plot cross-validation results
k_values = [3, 5, 7, 9, 11, 15]
uniform_accuracies = []
weighted_accuracies = []

for k, weighted, metrics in results:
    if weighted:
        weighted_accuracies.append(metrics['accuracy'])
    else:
        uniform_accuracies.append(metrics['accuracy'])

plt.figure(figsize=(10, 6))
plt.plot(k_values, uniform_accuracies, marker='o', label='Uniform weights')
plt.plot(k_values, weighted_accuracies, marker='s', label='Weighted')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN Cross-Validation Accuracy vs k with/without Weights')
plt.legend()
plt.grid(True)
plt.show()

uniform_senstivities = []
weighted_sensitivies = []

for k, weighted, metrics in results:
    if weighted:
        weighted_sensitivies.append(metrics['sensitivity'])
    else:
        uniform_senstivities.append(metrics['sensitivity'])

plt.figure(figsize=(10, 6))
plt.plot(k_values, uniform_senstivities, marker='o', label='Uniform weights')
plt.plot(k_values, weighted_sensitivies, marker='s', label='Weighted')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validation Sensitivity')
plt.title('KNN Cross-Validation Sensitivity vs k with/without Weights')
plt.legend()
plt.grid(True)
plt.show()

best = max(
    results,
    key=lambda x: 0.6 * x[2]["accuracy"] + 0.4 * x[2]["sensitivity"]
)
best_k, best_weighted, best_metrics = best

final_knn = KNNClassifier(
    k=best_k,
    distance="euclidean",
    weighted=best_weighted
)

final_knn.fit(X_train.tolist(), y_train.tolist())

test_preds = final_knn.predict(X_test.tolist())

print("Test accuracy:", accuracy_score(y_test, test_preds))
print(
    "Test sensitivity:",
    recall_score(y_test, test_preds, average="macro", zero_division=0)
)
# ----------------------------------


