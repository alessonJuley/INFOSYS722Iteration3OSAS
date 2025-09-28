#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 21:33:01 2025

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

# iteration 2
dataset_projected_boxcox = 'colorectal_cancer_reduced_projected_boxcox.csv'
df_boxcox = pd.read_csv(dataset_projected_boxcox)
# 167497 rows, 25 cols
df_boxcox.shape
df_boxcox.info()
# Load relevant algorithms
# XGBoost, Random Surival Forest, Naive Bayes
# -----6.3 - building models-----

# ============XGBoost============
# iteration 1
# before you start, do: pip install xgboost

# =======6.3=======

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from xgboost import XGBClassifier

RNG = 42
np.random.seed(RNG)

# ==== Helper plotting functions ====
def plot_roc(y_true, y_proba, title="ROC Curve (Test)"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pr(y_true, y_proba, title="Precision–Recall Curve (Test)"):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def show_metrics(y_true, y_pred, y_proba, header="Metrics"):
    print(f"\n=== {header} ===")
    print(f"Accuracy:            {accuracy_score(y_true, y_pred):.3f}")
    print(f"Balanced Accuracy:   {balanced_accuracy_score(y_true, y_pred):.3f}")
    print(f"ROC AUC:             {roc_auc_score(y_true, y_proba):.3f}")
    print(f"Average Precision:   {average_precision_score(y_true, y_proba):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n",
          classification_report(y_true, y_pred, zero_division=0))




# =====07-DMA=====
# split data
from sklearn.model_selection import train_test_split

X = dataset_reduced.drop(columns='Survival_5_yrs')
y = dataset_reduced['Survival_5_yrs']

# split 70/30
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)