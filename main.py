# -*- coding: utf-8 -*-

"""
This script demonstrates:
1. Fixing random seeds for reproducibility
2. Loading and exploring the dataset
3. Splitting into training and validation sets
4. Building base models and performing cross-validation
5. Model comparison (Accuracy, AUC, etc.)
6. Hyperparameter tuning with GridSearchCV (KNN, SVM, LR, RF)
7. Feature importance analysis (using RandomForest)
8. Model testing on the validation set with best hyperparameters and plotting ROC curves
"""

import warnings
warnings.filterwarnings("ignore")

# ========== 1. Fix Random Seeds ==========
import random
import numpy as np
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


# ========== 2. Load and Explore Data ==========
import pandas as pd
from pandas import read_csv, set_option

filename = 'data/control0-all.csv'
dataset = read_csv(filename, header=None)

print("=== Dataset Preview ===")
print(dataset.head())

print("\n=== Dataset Shape ===")
print(dataset.shape)

print("\n=== Data Types ===")
print(dataset.dtypes)

print("\n=== Statistical Summary ===")
print(dataset.describe())

# The label column is assumed to be at index 780/all, decided by OTUs
print("\n=== Label Distribution ===")
print(dataset.groupby(780).size())


# ========== 3. Split Training and Validation Sets ==========
from sklearn.model_selection import train_test_split
from collections import Counter

array = dataset.values
X = array[:, 0:780].astype(float)
Y = array[:, :780]

validation_size = 0.3
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y,
    test_size=validation_size,
    random_state=seed
)

print("\n=== Training Set Label Distribution ===")
print(Counter(Y_train))


# ========== 4. Build Base Models and Cross-Validation (Accuracy) ==========
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib import pyplot

# Create base models
models = {
    'LR': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'RF': RandomForestClassifier()
}

# Create pipelines with StandardScaler
pipelines = {
    'ScalerLR':  Pipeline([
        ('Scaler', StandardScaler()),
        ('LR', LogisticRegression())
    ]),
    'ScalerKNN': Pipeline([
        ('Scaler', StandardScaler()),
        ('KNN', KNeighborsClassifier())
    ]),
    'ScalerSVM': Pipeline([
        ('Scaler', StandardScaler()),
        ('SVM', SVC(random_state=seed))
    ]),
    'ScaledRF':  Pipeline([
        ('Scaler', StandardScaler()),
        ('RFR', RandomForestClassifier(random_state=seed))
    ])
}

num_folds = 10
scoring = 'accuracy'

print("\n=== 10-fold Cross-Validation Results (Accuracy) ===")
results_acc = []
for name, pipeline in pipelines.items():
    skfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(pipeline, X_train, Y_train,
                                 cv=skfold, scoring=scoring)
    results_acc.append(cv_results)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# Boxplot for Accuracy comparison
fig = pyplot.figure(figsize=(7, 5))
fig.suptitle('Algorithm Comparison (Accuracy)')
ax = fig.add_subplot(111)
pyplot.boxplot(results_acc)
ax.set_xticklabels(pipelines.keys())
pyplot.show()


# ========== 5. Model Comparison (AUC) ==========
print("\n=== 10-fold Cross-Validation Results (AUC) ===")
results_auc = []
for name, pipeline in pipelines.items():
    skfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results_auc = cross_val_score(pipeline, X_train, Y_train,
                                     cv=skfold, scoring='roc_auc')
    results_auc.append(cv_results_auc)
    print(f"{name}: {cv_results_auc.mean():.4f} ({cv_results_auc.std():.4f})")

# Boxplot for AUC comparison
fig_auc = pyplot.figure(figsize=(7, 5))
fig_auc.suptitle('Algorithm Comparison (AUC)')
ax_auc = fig_auc.add_subplot(111)
pyplot.boxplot(results_auc)
ax_auc.set_xticklabels(pipelines.keys())
pyplot.show()


# ========== 6. Hyperparameter Tuning (KNN, SVM, LR, RF) ==========
from sklearn.model_selection import GridSearchCV

# Scale training features once for all grid searches
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

# -- 6.1 KNN Tuning --
print("\n=== 6.1 KNN Tuning ===")
model_knn = KNeighborsClassifier()
param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
}
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
grid_knn = GridSearchCV(
    estimator=model_knn,
    param_grid=param_grid_knn,
    scoring=scoring,
    cv=kfold
)
grid_result_knn = grid_knn.fit(X_train_scaled, Y_train)
best_knn = grid_result_knn.best_estimator_
print(f"KNN Best Score: {grid_result_knn.best_score_:.4f}, Best Params: {grid_result_knn.best_params_}")

print("All Combinations:")
for mean, std, param in zip(
    grid_result_knn.cv_results_['mean_test_score'],
    grid_result_knn.cv_results_['std_test_score'],
    grid_result_knn.cv_results_['params']
):
    print(f"{mean:.4f} ({std:.4f}) with {param}")


# -- 6.2 SVM Tuning --
print("\n=== 6.2 SVM Tuning ===")
model_svm = SVC(probability=True, random_state=seed)
param_grid_svm = {
    'C': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}
grid_svm = GridSearchCV(
    estimator=model_svm,
    param_grid=param_grid_svm,
    scoring=scoring,
    cv=kfold
)
grid_result_svm = grid_svm.fit(X_train_scaled, Y_train)
best_svm = grid_result_svm.best_estimator_
print(f"SVM Best Score: {grid_result_svm.best_score_:.4f}, Best Params: {grid_result_svm.best_params_}")

print("All Combinations:")
for mean, std, param in zip(
    grid_result_svm.cv_results_['mean_test_score'],
    grid_result_svm.cv_results_['std_test_score'],
    grid_result_svm.cv_results_['params']
):
    print(f"{mean:.4f} ({std:.4f}) with {param}")


# -- 6.3 Logistic Regression Tuning --
print("\n=== 6.3 Logistic Regression Tuning ===")
model_lr = LogisticRegression(random_state=seed)
param_grid_lr = {
    'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'solver': ['liblinear', 'sag', 'lbfgs', 'newton-cg'],
    'class_weight': ['balanced', None],
    'max_iter': [1, 10, 20, 50, 100, 200, 500]
}
grid_lr = GridSearchCV(
    estimator=model_lr,
    param_grid=param_grid_lr,
    scoring=scoring,
    cv=kfold,
    n_jobs=-1
)
grid_result_lr = grid_lr.fit(X_train_scaled, Y_train)
best_lr = grid_result_lr.best_estimator_
print(f"LR Best Score: {grid_result_lr.best_score_:.4f}, Best Params: {grid_result_lr.best_params_}")

print("All Combinations:")
for mean, std, param in zip(
    grid_result_lr.cv_results_['mean_test_score'],
    grid_result_lr.cv_results_['std_test_score'],
    grid_result_lr.cv_results_['params']
):
    print(f"{mean:.4f} ({std:.4f}) with {param}")


# -- 6.4 RandomForest Tuning --
print("\n=== 6.4 Random Forest Tuning ===")
model_rf = RandomForestClassifier(random_state=seed)
param_grid_rf = {
    'n_estimators': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
}
grid_rf = GridSearchCV(
    estimator=model_rf,
    param_grid=param_grid_rf,
    scoring=scoring,
    cv=kfold,
    n_jobs=-1
)
grid_result_rf = grid_rf.fit(X_train_scaled, Y_train)
best_rf = grid_result_rf.best_estimator_
print(f"RF Best Score: {grid_result_rf.best_score_:.4f}, Best Params: {grid_result_rf.best_params_}")


# ========== 7. Feature Importance (Using RF as Example) ==========
print("\n=== 7. Feature Importance (RandomForest) ===")
rf_for_importance = RandomForestClassifier(n_estimators=200, random_state=seed)
rf_for_importance.fit(X_train, Y_train)

importances = rf_for_importance.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_for_importance.estimators_], axis=0)
indices = np.argsort(importances)[::-1]  # descending order

print("Feature ranking:")
for i in range(X_train.shape[1]):
    print(f"{i+1}. feature {indices[i]} ({importances[indices[i]]:.4f})")


# ========== 8. Model Testing (Validation Set) & ROC Curves with Best Params ==========
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score,
                             roc_curve, roc_auc_score)
from sklearn import metrics

# Scale validation features
X_validation_scaled = scaler.transform(X_validation)

# (1) KNN with best params
print("\n=== 8.1 KNN on Validation Set (Best Params) ===")
knn_predictions = best_knn.predict(X_validation_scaled)
knn_prob = best_knn.predict_proba(X_validation_scaled)[:, 1]

print("Accuracy:", accuracy_score(Y_validation, knn_predictions))
print("F1 Score:", f1_score(Y_validation, knn_predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_validation, knn_predictions))
print("Classification Report:\n", classification_report(Y_validation, knn_predictions))
fpr_knn, tpr_knn, _ = roc_curve(Y_validation, knn_prob, pos_label=1)
knn_auc = metrics.auc(fpr_knn, tpr_knn)
print("AUC =", knn_auc)


# (2) SVM with best params
print("\n=== 8.2 SVM on Validation Set (Best Params) ===")
svm_predictions = best_svm.predict(X_validation_scaled)
svm_prob = best_svm.predict_proba(X_validation_scaled)[:, 1]

print("Accuracy:", accuracy_score(Y_validation, svm_predictions))
print("F1 Score:", f1_score(Y_validation, svm_predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_validation, svm_predictions))
print("Classification Report:\n", classification_report(Y_validation, svm_predictions))
fpr_svm, tpr_svm, _ = roc_curve(Y_validation, svm_prob, pos_label=1)
svm_auc = metrics.auc(fpr_svm, tpr_svm)
print("AUC =", svm_auc)


# (3) Logistic Regression with best params
print("\n=== 8.3 Logistic Regression on Validation Set (Best Params) ===")
lr_predictions = best_lr.predict(X_validation_scaled)
lr_prob = best_lr.predict_proba(X_validation_scaled)[:, 1]

print("Accuracy:", accuracy_score(Y_validation, lr_predictions))
print("F1 Score:", f1_score(Y_validation, lr_predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_validation, lr_predictions))
print("Classification Report:\n", classification_report(Y_validation, lr_predictions))
fpr_lr, tpr_lr, _ = roc_curve(Y_validation, lr_prob, pos_label=1)
lr_auc = metrics.auc(fpr_lr, tpr_lr)
print("AUC =", lr_auc)


# (4) RandomForest with best params
print("\n=== 8.4 Random Forest on Validation Set (Best Params) ===")
rf_predictions = best_rf.predict(X_validation_scaled)
rf_prob = best_rf.predict_proba(X_validation_scaled)[:, 1]

print("Accuracy:", accuracy_score(Y_validation, rf_predictions))
print("F1 Score:", f1_score(Y_validation, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_validation, rf_predictions))
print("Classification Report:\n", classification_report(Y_validation, rf_predictions))
fpr_rf, tpr_rf, _ = roc_curve(Y_validation, rf_prob, pos_label=1)
rf_auc = metrics.auc(fpr_rf, tpr_rf)
print("AUC =", rf_auc)


# Unified ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC={knn_auc:.2f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={svm_auc:.2f})")
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={lr_auc:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={rf_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Validation Set)')
plt.legend()
plt.savefig('ROC_BestParams.svg', format='svg', bbox_inches='tight')
plt.show()
