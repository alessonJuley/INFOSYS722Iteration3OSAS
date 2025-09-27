#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 23:15:53 2025

@author: alessonabao
"""

# =====04-DT=====
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
dataset = 'colorectal_cancer_final.csv'
df = pd.read_csv(dataset)
df.shape
df.info()

# check corr 
corr_final_matrix = df.corr()

# Heatmap 
plt.figure(figsize=(12, 8)) 
sns.heatmap(corr_final_matrix, annot=True, cmap="coolwarm", fmt=".2f", \
            annot_kws={"fontsize": 5}, square=True, linewidths=0.1) 
plt.title("Feature Correlation Heatmap") 
plt.show()

# -----4.1 - data reduction-----
# ---Separate predictors and target---
X = df.drop(columns=["Survival_5_yrs"])
y = df["Survival_5_yrs"].astype(int)

# ---chi2 + MI---
# Scale non-negative features for chi2
X_scaled = MinMaxScaler().fit_transform(
    X.select_dtypes(include=["int64", "Int64", "Int8", "boolean"])
)

chi2_selector = SelectKBest(score_func=chi2, k=10)
chi2_selector.fit(X_scaled, y)

mi_selector = SelectKBest(score_func=mutual_info_classif, k=10)
mi_selector.fit(X, y)

chi2_features = list(X.columns[chi2_selector.get_support()])
mi_features = list(X.columns[mi_selector.get_support()])

print("Top chi2 features:", chi2_features)
print("Top MI features:", mi_features)

# Chi2: 'Family_History', 'Alcohol_Consumption', 'BMI_Normal', 
# 'Screening_History_Irregular', 'Screening_History_Never', 
# 'Treatment_Type_Surgery', 'Urban', 'Rural', 'Insured', 'Uninsured'

# MI: Gender_F', 'Gender_M', 'Cancer_Stage_Regional', 'Smoking_History', 
# 'Alcohol_Consumption', 'BMI_Overweight', 'Screening_History_Regular', 
# 'Early_Detection', 'Urban', 'Insured'

# ---Logical Filtering (based on medical research)---
logical_features = [
    "Age", "Gender_F", "Cancer_Stage_Regional", "Cancer_Stage_Metastatic",
    "Tumor_mm", "Family_History", "Smoking_History", "Alcohol_Consumption",
    "BMI_Overweight", "BMI_Obese", "Diet_Risk", "Physical_Activity",
    "Diabetes", "IBD", "Genetic_Mutation", "Screening_History_Irregular",
    "Screening_History_Never", "Early_Detection", "Treatment_Type_Surgery",
    "Treatment_Type_Radiotherapy", "Treatment_Type_Chemotherapy", "Urban", "Insured"
]

# columns I want to keep: 'Age', 'Gender_F', 'Cancer_Stage_Regional',
# 'Cancer_Stage_Metastatic', 'Tumor_mm', 'Family_History', 'Smoking_History',
# 'Alcohol_Consumption', 'BMI_Overweight', 'BMI_Obese', 'Diet_Risk', 'Physical_Activity',
# 'Diabetes', 'IBD', 'Genetic_Mutation', 'Screening_History_Irregular',
# 'Screening_History_Never', 'Early_Detection', 'Treatment_Type_Surgery', 
# 'Treatment_Type_Radiotherapy', 'Treatment_Type_Chemotherapy', 'Urban', 'Insured'

# ---statistical + logical selections---
combined_features = sorted(set(chi2_features + mi_features + logical_features))

# Drop opposites/extras that can show up from chi2/MI
to_drop = [
    "Gender_M",               # keep Gender_F
    "Screening_History_Regular",
    "BMI_Normal",
    "Rural",                  # keep Urban
    "Uninsured",              # keep Insured
    "Cancer_Stage_Localized"  # not in keep list
]

final_features = [f for f in combined_features if f not in to_drop]

print("Final selected features:", final_features)

# ---reduced dataframe---
df_reduced = df[final_features + ["Survival_5_yrs"]]

# make corr matrix
corr_final_matrix = df_reduced.corr()

# Heatmap 
plt.figure(figsize=(12, 8)) 
sns.heatmap(corr_final_matrix, annot=True, cmap="coolwarm", fmt=".2f", \
            annot_kws={"fontsize": 5}, square=True, linewidths=0.1) 
plt.title("Feature Correlation Heatmap") 
plt.show()

# Save reduced dataset
df_reduced.to_csv("colorectal_cancer_reduced.csv", index=False)

# checks
print("Original dataset shape:", df.shape) # 167497, 33
print("Reduced dataset shape:", df_reduced.shape) # 167497, 24

# -----4.2 - data projection-----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import boxcox

# --- copy df_reduced ---
df_proj = df_reduced.copy()

# ----------helpers----------
def moments(x: pd.Series, name: str):
    x = pd.Series(x).dropna().astype(float)
    sk = stats.skew(x)
    ku = stats.kurtosis(x)
    print(f"{name:>20s}  |  Skewness: {sk:.3f}  |  Kurtosis: {ku:.3f}")
    return sk, ku

def plot_hist(x, title, xlabel):
    plt.figure(figsize=(6, 4))
    sns.histplot(x, kde=True, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_qq(x, title):
    plt.figure(figsize=(6, 6))
    sm.qqplot(pd.Series(x).dropna(), line="45")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ----------before transformation ----------
x0 = df_proj["Tumor_mm"].astype(float)
print("\nDistribution diagnostics (Tumor_mm):")
moments(x0, "Before")

plot_hist(x0, "Histogram of Tumor_mm (Before)", "Tumor size (mm)")
plot_qq(x0, "Q–Q Plot of Tumor_mm (Before)")

# ----------log----------
# Shift only if needed (to keep arguments strictly > 0)
log_shift = 0.0
min_x0 = float(x0.min())
if min_x0 <= 0:
    log_shift = (1 - min_x0) + 1e-9
x_log = np.log(x0 + log_shift)
df_proj["Tumor_mm_log"] = x_log

moments(x_log, "After (log)")
plot_hist(x_log, "Histogram of Tumor_mm (After log)", "log(Tumor size)")
plot_qq(x_log, "Q–Q Plot of Tumor_mm (After log)")

# ----------square root----------
# shift only if needed to ensure non-negativity
sqrt_shift = 0.0
if min_x0 < 0:
    sqrt_shift = -min_x0
x_sqrt = np.sqrt(x0 + sqrt_shift)
df_proj["Tumor_mm_sqrt"] = x_sqrt

moments(x_sqrt, "After (sqrt)")
plot_hist(x_sqrt, "Histogram of Tumor_mm (After √)", "√(Tumor size)")
plot_qq(x_sqrt, "Q–Q Plot of Tumor_mm (After √)")

# ----------box-cox----------
# Box–Cox requires strictly positive input. Shift if necessary.
bc_shift = 0.0
if min_x0 <= 0:
    bc_shift = (1 - min_x0) + 1e-9

x_for_bc = x0 + bc_shift
x_bc, lambda_bc = boxcox(x_for_bc)  # returns transformed data and λ
df_proj["Tumor_mm_boxcox"] = x_bc

print(f"\nBox–Cox λ (lambda): {lambda_bc:.4f}  |  shift used: {bc_shift}")
moments(x_bc, "After (Box–Cox)")
plot_hist(x_bc, "Histogram of Tumor_mm (After Box–Cox)", "Box–Cox(Tumor size)")
plot_qq(x_bc, "Q–Q Plot of Tumor_mm (After Box–Cox)")

# ----------summary----------
summary = pd.DataFrame(
    [
        ("Before",) + moments(x0, "Before"),
        ("log",)    + moments(x_log, "After (log)"),
        ("sqrt",)   + moments(x_sqrt, "After (sqrt)"),
        ("boxcox",) + moments(x_bc, "After (Box–Cox)")
    ],
    columns=["Transform", "Skewness", "Kurtosis"]
).drop_duplicates(subset=["Transform"]).reset_index(drop=True)

print("\nSkewness & Kurtosis Summary:")
print(summary.round(3).to_string(index=False))

# ----------save box-cox only----------
df_boxcox = df_reduced.copy()
df_boxcox["Tumor_mm_boxcox"] = df_proj["Tumor_mm_boxcox"]

output_path = "colorectal_cancer_reduced_projected_boxcox.csv"
df_boxcox.to_csv(output_path, index=False)

print(f"\nSaved: {output_path} (with Box–Cox only)")
