
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


import os
print("Current working directory:", os.getcwd())


# 1. Load Data
data = pd.read_csv('combined_landmarks_clean.csv')
print("Data shape:", data.shape)
print(data['label'].value_counts())


# 2. Prepare Features & Labels 
X = data.drop('label', axis=1)  # Only drop label – no instance_id error
y = data['label']


# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Class order (A-J):", le.classes_)


print("All columns:", data.columns.tolist())
print("First few rows:")
print(data.head())


# 2. Prepare Features & Labels and keep 63 landmarks
X = data.drop(['image_name', 'label'], axis=1)  # Removes string column + label
y = data['label']

print("Feature columns selected:", X.columns.tolist()[:5], "...", X.columns.tolist()[-5:])
print("Final features shape:", X.shape)  # Should be (3661, 63)

# Encode labels A-J → 0-9
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes (A-J order):", le.classes_)

# Normalize – MUST for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-Test Split (80-20, balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# 4. Hyperparameter Tuning (5-fold CV)
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("\n=== BEST MODEL RESULTS ===")
print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy: {:.3f}".format(grid.best_score_))

# Hyperparameter Table for Poster
results_df = pd.DataFrame(grid.cv_results_)
print("\nHyperparameter Tuning Table:")
print(results_df[['param_C', 'param_kernel', 'mean_test_score', 'std_test_score']]
      .sort_values('mean_test_score', ascending=False).round(3))

# 5. Final Test
best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nFINAL TEST ACCURACY: {test_acc:.3f} ({test_acc*100:.1f}%)")
print("\nPer-Class Performance:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 6. Confusion Matrix (Pure Matplotlib – Safe & Beautiful)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title(f'SVM Confusion Matrix\nTest Accuracy: {test_acc*100:.1f}% | Best: {grid.best_params_}', fontsize=16)
plt.colorbar()

classes = le.classes_
tick_marks = np.arange(10)
plt.xticks(tick_marks, classes, fontsize=12)
plt.yticks(tick_marks, classes, fontsize=12)

thresh = cm.max() / 2.
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i,j], ha="center", va="center",
                 color="white" if cm[i,j] > thresh else "black", fontsize=11)

plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig('svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nDone! 'svm_confusion_matrix.png' saved – Ready for poster!")





