#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 01:10:42 2025

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
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Define features and target
X = df_reduced.drop(columns=["Survival_5_yrs"])   # predictors
y = df_reduced["Survival_5_yrs"].astype(int)      # target (0 = no, 1 = yes)

# =====experiments with different splits =====
for split in [0.2, 0.25]:   # 80/20 and 75/25
    print(f"\n=== Experiment with test_size={split} ({int((1-split)*100)}/{int(split*100)} split) ===")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=42, stratify=y
    )
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    print("=== Naive Bayes Results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    # =====7=====
    # Naive Bayes does not provide feature importance directly
    feature_means = pd.DataFrame(model.theta_, columns=X.columns, index=["Class 0", "Class 1"]).T
    print("\n=== Class-conditional feature means (top 10) ===")
    print(feature_means.head(10))
    
    # =====8=====
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Naive Bayes ({int((1-split)*100)}/{int(split*100)})")
    plt.plot([0,1],[0,1],"--",color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Naive Bayes)")
    plt.legend()
    plt.show()
    
    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(rec, prec, label=f"Naive Bayes ({int((1-split)*100)}/{int(split*100)})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Naive Bayes)")
    plt.legend()
    plt.show()