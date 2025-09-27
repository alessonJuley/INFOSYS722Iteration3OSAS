#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 14:50:21 2025

@author: alessonabao
"""

# 01-BU
"""
support clinical decision-making in colorectal cancer (CRC) care by 
developing a predictive model that estimates the likelihood of 
patient survival beyond five years following diagnosis
"""

# =====02-DU=====
import pandas as pd

# load dataset
Primary_Dataset = 'colorectal_cancer_dataset.csv'

# -----2.1 - Collecting initial data-----
df1 = pd.read_csv(Primary_Dataset)

# print first and last 2 columns with first and last 5 rows
print(df1)

# visualise top 5 rows
df1.head()


# -----2.2 - Describing data-----
# +++++++++++++++SCREENSHOT+++++++++++++++
# format
df1.dtypes

# quantity
print("Shape of dataset: ", df1.shape)

# fields
df1.info()

# features of the data
describe_data = df1.describe(include='all').T


# -----2.3 - Data exploration-----
df1['Age'].plot(kind='hist', bins=20, title="Age Distribution", legend=False)
df1['Country'].value_counts().plot.bar()
df1['Gender'].value_counts().plot.bar()
df1['Cancer_Stage'].value_counts().plot.bar()
df1['Family_History'].value_counts().plot.bar()

df1['Smoking_History'].value_counts().plot.bar()
df1['Alcohol_Consumption'].value_counts().plot.bar()
df1['Obesity_BMI'].value_counts().plot.bar()
df1['Diet_Risk'].value_counts().plot.bar()

df1['Urban_or_Rural'].value_counts().plot.bar()
df1['Physical_Activity'].value_counts().plot.bar()
df1['Diabetes'].value_counts().plot.bar()
df1['Inflammatory_Bowel_Disease'].value_counts().plot.bar()

df1['Genetic_Mutation'].value_counts().plot.bar()
df1['Screening_History'].value_counts().plot.bar()
df1['Early_Detection'].value_counts().plot.bar()
df1['Treatment_Type'].value_counts().plot.bar()

df1['Survival_5_years'].value_counts().plot.bar()
df1['Mortality'].value_counts().plot.bar()
df1['Economic_Classification'].value_counts().plot.bar()
df1['Healthcare_Access'].value_counts().plot.bar()

df1['Insurance_Status'].value_counts().plot.bar()
df1['Survival_Prediction'].value_counts().plot.bar()

import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(df1['Age'].sample(n=1000))
sns.distplot(df1['Tumor_Size_mm'].sample(n=1000))

# survival outcome by gender
sns.countplot(x="Gender", hue="Survival_5_years", data=df1)
plt.title("Survival Outcome by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survival (5 Years)", labels=["No", "Yes"])
plt.show()

# survival by age
sns.boxplot(x="Survival_5_years", y="Age", data=df1)
plt.title("Age Distribution by Survival Outcome")
plt.xlabel("Survival (5 Years)")
plt.ylabel("Age")
plt.show()

# cancer stage vs. survival 5 years
plt.figure(figsize=(8, 5))
sns.countplot(x='Cancer_Stage', hue='Survival_5_years', data=df1, palette="coolwarm")
plt.title("Cancer Stage vs 5-Year Survival")
plt.xlabel("Cancer Stage")
plt.ylabel("Count")
plt.show()

# -----2.3 - Data validation-----
# check missing values
# Count missing values per column
# +++++++++++++++SCREENSHOT+++++++++++++++
missing_per_col = df1.isnull().sum()
print(df1.isnull().sum())

# Percentage of missing values per column
# +++++++++++++++SCREENSHOT+++++++++++++++
percent_missing_per_col = (df1.isnull().mean() * 100).round(2)
print((df1.isnull().mean() * 100).round(2))

# Duplication check
# +++++++++++++++SCREENSHOT+++++++++++++++
print("Duplicate rows:", df1.duplicated().sum())

# check data imbalance
# Distribution of target
sns.countplot(x="Survival_5_years", data=df1)
plt.title("Class Distribution: Survival (5 Years)")
plt.show()

# check for outliers
def iqr_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers, lower_bound, upper_bound

# outliers in Age
outlier_age, lower, upper = iqr_outliers(df1["Age"])
print(f"Outliers in Age: {len(outlier_age)} values")
print(f"Lower bound: {lower}, Upper bound: {upper}")

outlier_tumor_size, lower, upper = iqr_outliers(df1["Tumor_Size_mm"])
print(f"Outliers in Tumor_Size_mm: {len(outlier_tumor_size)} values")
print(f"Lower bound: {lower}, Upper bound: {upper}")

# Visualise
sns.boxplot(x=df1["Age"])
plt.title("Outliers in Age (IQR Method)")
plt.show()

sns.boxplot(x=df1["Tumor_Size_mm"])
plt.title("Outliers in Tumor_Size_mm (IQR Method)")
plt.show()

