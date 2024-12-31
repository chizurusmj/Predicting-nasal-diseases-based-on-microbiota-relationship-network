# -*- coding: utf-8 -*-

"""
1. Reading from 'sel_su.csv' which stores the sorted feature indices (descending order).
2. Incrementally adding features according to this ranking.
3. Output and plot both Accuracy and AUC for multiple models (SVM, RF, LR, KNN).
"""

import warnings
from main import kfold

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white')  # Set style for plots

from collections import defaultdict
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    accuracy_score, roc_curve, auc
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pandas import read_csv
# ========== 1. Read feature indices from 'su.csv' ==========
"""
'sel_su.csv' has a single row with feature indices sorted by importance
(from the most important to the least). 
"""
df = pd.read_csv("data/sel_su.csv", header=None)
su = df.values[0].tolist()  # Convert the row to a Python list
print("=== First 10 feature indices from su.csv ===")
print(su[:10])

# ========== 2. Prepare training/validation data and related variables ==========
"""
The same in main.py
"""
from sklearn.model_selection import train_test_split
import random
def fix_seed(seed=7):
    """
    Fix random seeds for reproducibility.
    """
    # Python's built-in random
    random.seed(seed)
    # Numpy random
    np.random.seed(seed)

seed = 7
fix_seed(seed)

filename = 'data/control0-all.csv'
dataset = read_csv(filename, header=None)
array = dataset.values
X = array[:, 0:780].astype(float)
Y = array[:, :780]

validation_size = 0.3
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y,
    test_size=validation_size,
    random_state=seed
)
# ========== 3. Create pipelines for multiple models and define parameter grids ==========
scaler = StandardScaler().fit(X_train)

svc = Pipeline([
    ("scaler", scaler),
    ("svc", SVC(probability=True, random_state=seed))
])

rand_forest = Pipeline([
    ("scaler", scaler),
    ("rf", RandomForestClassifier(random_state=seed))
])

LR = Pipeline([
    ("scaler", scaler),
    ("LR", LogisticRegression(random_state=seed))
])

knn = Pipeline([
    ("scaler", scaler),
    ("knn", KNeighborsClassifier())
])

# Parameter grids
svc_param = {
    "svc__C": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0],
    "svc__kernel": ["linear", "poly", "rbf", "sigmoid"]
}

rand_forest_param = {
    "rf__n_estimators": [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
}

LR_param = {
    "LR__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "LR__solver": ["liblinear", "sag", "lbfgs", "newton-cg"],
    "LR__class_weight": ["balanced", None],
    "LR__max_iter": [1, 10, 20, 50, 100, 200, 500]
}

knn_param = {
    "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
}

# ========== 4. Setup GridSearchCV objects ==========
"""
Here we demonstrate with AUC scoring, but you can switch to 'accuracy'.
We will actually compute both AUC and accuracy below, 
"""
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
gs_svm = GridSearchCV(
    svc, svc_param,
    scoring='roc_auc',
    cv=kfold,
    n_jobs=-1,
    verbose=1
)
gs_rand_forest = GridSearchCV(
    rand_forest, rand_forest_param,
    scoring='roc_auc',
    cv=kfold,
    n_jobs=-1,
    verbose=1
)
gs_LR = GridSearchCV(
    LR, LR_param,
    scoring='roc_auc',
    cv=kfold,
    n_jobs=-1,
    verbose=1
)
gs_knn = GridSearchCV(
    knn, knn_param,
    scoring='roc_auc',
    cv=kfold,
    n_jobs=-1,
    verbose=1
)

grids = {
    "gs_svm": gs_svm,
    "gs_rand_forest": gs_rand_forest,
    "gs_LR": gs_LR,
    "gs_knn": gs_knn
}

# ========== 5. Incrementally add features and evaluate models (output both Accuracy and AUC) ==========

records_auc = defaultdict(list)  # store AUC for each model at different #features
records_acc = defaultdict(list)  # store Accuracy for each model at different #features

# Suppose we try from 5 features up to 65, increment by 5, decided by the feature num
for i in range(5, 65, 5):
    # Select top i features from su
    x_train_sel = X_train[:, su[:i]]
    x_test_sel = X_validation[:, su[:i]]

    print(f"\n=== Using top {i} features ===")

    # Fit each GridSearchCV with these i features
    for model_name, grid_search in grids.items():
        grid_search.fit(x_train_sel, Y_train)
        best_est = grid_search.best_estimator_

        # Predictions and probabilities on the validation set
        preds = best_est.predict(x_test_sel)
        preds_prob = best_est.predict_proba(x_test_sel)[:, 1]

        # Compute Accuracy
        acc_score = accuracy_score(Y_validation, preds)

        # Compute AUC
        fpr, tpr, _ = roc_curve(Y_validation, preds_prob, pos_label=1)
        auc_score = auc(fpr, tpr)

        # Record
        records_auc[model_name].append(auc_score)
        records_acc[model_name].append(acc_score)

        print(f"{model_name}: ACC={acc_score:.4f}, AUC={auc_score:.4f}")

# ========== 6. Plot both Accuracy and AUC curves ==========

feature_range = list(range(5, 65, 5))

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # left subplot for Accuracy
plt.plot(feature_range, records_acc["gs_svm"], marker='o', linestyle='-', color='b', label="SVM")
plt.plot(feature_range, records_acc["gs_rand_forest"], marker='s', linestyle='--', color='r', label="Random Forest")
plt.plot(feature_range, records_acc["gs_LR"], marker='^', linestyle='-.', color='g', label="Logistic Regression")
plt.plot(feature_range, records_acc["gs_knn"], marker='x', linestyle=':', color='y', label="KNN")
plt.title('Model Comparison by Accuracy', fontsize=14)
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)

# Plot AUC
plt.subplot(1, 2, 2)  # right subplot for AUC
plt.plot(feature_range, records_auc["gs_svm"], marker='o', linestyle='-', color='b', label="SVM")
plt.plot(feature_range, records_auc["gs_rand_forest"], marker='s', linestyle='--', color='r', label="Random Forest")
plt.plot(feature_range, records_auc["gs_LR"], marker='^', linestyle='-.', color='g', label="Logistic Regression")
plt.plot(feature_range, records_auc["gs_knn"], marker='x', linestyle=':', color='y', label="KNN")
plt.title('Model Comparison by AUC', fontsize=14)
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('Model_Comparison_Acc_AUC.svg', format='svg', bbox_inches='tight')
plt.show()

# ========== 7. Final Output Check ==========
print("\n=== Final Records (Accuracy) ===")
for k, v in records_acc.items():
    print(k, ":", v)

print("\n=== Final Records (AUC) ===")
for k, v in records_auc.items():
    print(k, ":", v)
