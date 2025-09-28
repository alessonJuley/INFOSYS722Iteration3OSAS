#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 23:01:20 2025

@author: alessonabao
"""

# =====06-DMA=====
import pandas as pd

# load datasets
dataset_reduced = 'colorectal_cancer_reduced.csv'
df_reduced = pd.read_csv(dataset_reduced)
# 167497 rows, 24 cols
df_reduced.shape
df_reduced.info()

# =====6.3=====
# before you start, do: pip install xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier

# Define target and features
X = df_reduced.drop(columns=["Survival_5_yrs"])   # predictors
y = df_reduced["Survival_5_yrs"]                  # target

# Split into training and test sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Build a simple baseline XGBoost model
model = XGBClassifier(
    objective="binary:logistic",   # binary classification
    n_estimators=100,              # number of trees
    max_depth=3,                   # tree depth
    learning_rate=0.1,             # learning rate
    random_state=42,               # reproducibility
    eval_metric="auc"              # track AUC
)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate performance
print("=== Baseline XGBoost Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# =====7.3=====

# Feature importance (which predictors matter most)
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(len(sorted_idx[:15])), importances[sorted_idx[:15]])
plt.xticks(range(len(sorted_idx[:15])), X.columns[sorted_idx[:15]], rotation=90)
plt.title("Top 15 Feature Importances (XGBoost)")
plt.show()

# =====8=====

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label="XGBoost")
plt.plot([0,1],[0,1],"--",color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(rec, prec, label="XGBoost")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
