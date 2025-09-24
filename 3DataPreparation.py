#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 01:53:31 2025

@author: alessonabao
"""

# =====03-DP=====
# +++++++++++++++SCREENSHOT+++++++++++++++
import pandas as pd
# load dataset
Primary_Dataset = 'colorectal_cancer_dataset.csv'
df1 = pd.read_csv(Primary_Dataset)
df1.info()

# before doing the steps, I want to introduce null values to satisfy
# requirements for step 3.2.
import numpy as np

# copy dataset so that original dataset is not affected
df1_altered = df1.copy()
# df1_altered shape: (167497, 28)
df1_altered.shape

np.random.seed(42)

# ---add null values---
# These features are clinically relevant for survival prediction
cols_with_nulls = [
    'Age',               # age is a strong survival factor
    'Tumor_Size_mm',     # tumour size affects prognosis
    'Cancer_Stage',      # staging is critical in survival analysis
    'Obesity_BMI',       # BMI affects post-surgery outcomes
    'Physical_Activity'  # may impact recovery & survival
]

for col in cols_with_nulls:
    null_indices = df1_altered.sample(frac=0.05).index  # 5% missing values
    df1_altered.loc[null_indices, col] = np.nan

# ---save altered dataset---
df1_altered.to_csv("colorectal_cancer_dataset_altered.csv", index=False)
print("Altered dataset saved as colorectal_cancer_dataset_altered.csv")

Altered_Dataset = 'colorectal_cancer_dataset_altered.csv'
dfa = pd.read_csv(Altered_Dataset)

# NOTE: df1_altered will be used from this point on
# -----3.1 - data selection-----
TARGET = 'Survival_5_years'
KEY = 'Patient_ID'

drop_cols = [
    'Country',
    'Mortality',
    'Incidence_Rate_per_100K',
    'Mortality_Rate_per_100K',
    'Economic_Classification',
    'Survival_Prediction'
]

keep_cols = [c for c in dfa.columns if c not in drop_cols]

df_selected = dfa[keep_cols].copy()
# +++++++++++++++SCREENSHOT+++++++++++++++
# checks
print("Altered dataset shape: ", dfa.shape)
print("Selected shape: ", df_selected.shape)
print("Dropped columns: ", drop_cols)
print("Kept Patient_ID for splitting/merging in Step 3.3–3.4")
df_selected.head()
# -----3.2 - data cleaning-----
# checks
# ---nulls---
# +++++++++++++++SCREENSHOT+++++++++++++++
print("Missing values per column before imputation:")
print(df_selected[cols_with_nulls].isnull().sum())
print("Selected shape: ", df_selected.shape)

continuous_cols = ['Age', 'Tumor_Size_mm']
categorical_cols = ['Cancer_Stage', 'Obesity_BMI', 'Physical_Activity']

for col in continuous_cols:
    median_val = df_selected[col].median()
    df_selected.fillna({col: median_val}, inplace=True)
    print(f"Imputed {col} with median value {median_val}")
    
for col in categorical_cols:
    mode_val = df_selected[col].mode()[0]
    df_selected.fillna({col: mode_val}, inplace=True)
    print(f"Imputed {col} with mode value '{mode_val}'")

# +++++++++++++++SCREENSHOT+++++++++++++++
print("\nMissing values per column AFTER imputation:")
print(df_selected[cols_with_nulls].isnull().sum())
print("Selected shape: ", df_selected.shape)

# ---white spaces---
# +++++++++++++++SCREENSHOT+++++++++++++++
obj_cols = df_selected.select_dtypes(include=['object']).columns.tolist()

for col in obj_cols:
    # count how many entries have leading or trailing spaces
    space_issues = df_selected[col].astype(str).apply(
        lambda x: (x != x.strip())
    ).sum()

    if space_issues > 0:
        print(f"{col}: {space_issues} values with leading/trailing spaces")
    else:
        print(f"{col}: no space issues detected")

# ---duplicates---
# +++++++++++++++SCREENSHOT+++++++++++++++
# other columns will have repeated categorical values
for col in df_selected.columns:
    counts = df_selected[col].value_counts()
    duplicates = counts[counts > 1]
    print(f"\nColumn: {col}")
    if not duplicates.empty:
        print(f"  Number of duplicated values: {len(duplicates)}")
        print("  Top duplicated values:")
        print(duplicates.head(5))
    else:
        print("  No duplicates found (all values unique).")
# -----3.3 - data construct-----
# split data into different patient data categories
KEY, TARGET = "Patient_ID", "Survival_5_years"

patient_demographics = df_selected[[KEY, "Age", "Gender"]].copy()

patient_clinical = df_selected[[KEY,
    "Cancer_Stage", "Tumor_Size_mm", "Family_History", 
    "Diabetes", "Inflammatory_Bowel_Disease", "Genetic_Mutation"
]].copy()

patient_lifestyle = df_selected[[KEY,
    "Smoking_History", "Alcohol_Consumption", "Obesity_BMI",
    "Diet_Risk", "Physical_Activity"
]].copy()

patient_treatment = df_selected[[KEY, "Treatment_Type", "Screening_History", \
                                 "Early_Detection"]].copy()

# split Urban_or_Rural into urban and rural and Insurance_Status into insured and uninsured
# One-hot encode Urban_or_Rural into 'urban' and 'rural' columns
# +++++++++++++++SCREENSHOT+++++++++++++++
urban_dummies = pd.get_dummies(
    df_selected["Urban_or_Rural"].str.lower(),
    prefix="",       
    prefix_sep=""    
)[["urban", "rural"]]  

insured_dummies = pd.get_dummies(
    df_selected["Insurance_Status"].str.lower(),
    prefix="",       
    prefix_sep=""    
)[["insured", "uninsured"]]  # keep only these two columns

patient_access = pd.concat(
    [
        df_selected[[KEY, "Healthcare_Access"]],
        insured_dummies, urban_dummies
    ],
    axis=1
).copy()

patient_costs = df_selected[[KEY, "Healthcare_Costs"]].copy()

patient_outcomes = df_selected[[KEY, TARGET]].copy()
# -----3.4 - data integration-----
# merge patient_demographics, patient_clinical, patient_lifestyle,
# patient_treatment, patient_access, patient_costs
# patient_outcomes into one dataset again

df_integrated = patient_demographics.copy()

# merge all other patient tables using Patient_ID
df_integrated = df_integrated.merge(patient_clinical,  on=KEY, how="inner")
df_integrated = df_integrated.merge(patient_lifestyle, on=KEY, how="inner")
df_integrated = df_integrated.merge(patient_treatment, on=KEY, how="inner")
df_integrated = df_integrated.merge(patient_access,   on=KEY, how="inner")
df_integrated = df_integrated.merge(patient_costs,    on=KEY, how="inner")
df_integrated = df_integrated.merge(patient_outcomes, on=KEY, how="inner")

# reorder columns
final_columns = [
    "Patient_ID", "Age", "Gender", "Cancer_Stage", "Tumor_Size_mm",
    "Family_History", "Smoking_History", "Alcohol_Consumption",
    "Obesity_BMI", "Diet_Risk", "Physical_Activity", "Diabetes",
    "Inflammatory_Bowel_Disease", "Genetic_Mutation",
    "Screening_History", "Early_Detection", "Treatment_Type",
    "Survival_5_years", "Healthcare_Costs",
    "urban", "rural", "Healthcare_Access",
    "insured", "uninsured"
]

df_integrated = df_integrated[final_columns].copy()

# df_selected = merged tables with one-hot encoding = 167497 rows, 24 col
# df_selected = patient-specific columns only =  167497 rows, 22 col
# df1 = original dataset = 167497 rows, 28 col
print("Integrated dataset shape:", df_integrated.shape) 
print("Selected dataset shape: ", df_selected.shape)
print("Original dataset shape: ", df1.shape)

# checks if merged properly
n_rows = len(df_integrated)
n_unique_ids = df_integrated["Patient_ID"].nunique()
print(f"Rows in final dataset: {n_rows}")
print(f"Unique Patient_IDs:   {n_unique_ids}")

if n_rows == n_unique_ids:
    print("No duplicate Patient_IDs")
else:
    print("Duplicate found")
    
missing_ids = df_integrated["Patient_ID"].isna().sum()
print(f"Missing Patient_IDs: {missing_ids}")

# check if there's still null values
total_nulls = df_integrated.isnull().sum().sum()
print(f"\nTotal null values in dataset: {total_nulls}")
# -----3.5 - data reformatting-----

df_final = df_integrated.copy()

# remove Patient_ID since it's not needed anymore
df_final = df_final.drop(columns=["Patient_ID"])

# rename long columns
rename_cols = {
    "Tumor_Size_mm": "Tumor_mm",
    "Obesity_BMI": "BMI",
    "Inflammatory_Bowel_Disease": "IBD",
    "Survival_5_years": "Survival_5_yrs",
    "Healthcare_Costs": "HC_Costs",
    "urban": "Urban",
    "rural": "Rural",
    "insured": "Insured",
    "uninsured": "Uninsured"
} 

df_final.rename(columns={k: v for k, v in rename_cols.items() if k in df_final.columns}, inplace=True)

# change yes/no values to true/false for consistency
yes_no_cols = ['Family_History', 'Smoking_History', 'Alcohol_Consumption', \
               'Diabetes', 'IBD', 'Genetic_Mutation', 'Early_Detection', \
               'Survival_5_yrs']
    
def yesno_to_bool(series: pd.Series) -> pd.Series:
    # Map yes/no (and 1/0, true/false strings) to booleans
    s = series.astype(str).str.strip().str.lower()
    mapped = s.map({
        "yes": True, "y": True, "true": True, "1": True,
        "no": False, "n": False, "false": False, "0": False
    })
    # Use pandas Nullable Boolean dtype to allow missing values (pd.NA)
    return mapped.astype("boolean")

for col in yes_no_cols:
    if col in df_final.columns:
        df_final[col] = yesno_to_bool(df_final[col])
        
# check changes
print("Boolean dtypes check:")
print(df_final[yes_no_cols].dtypes)  # should show 'boolean'

print("\nValue checks (first 5 rows):")
print(df_final[yes_no_cols].head())

print("\nCounts of True/False/NA per column:")
for c in yes_no_cols:
    if c in df_final.columns:
        print(c, df_final[c].value_counts(dropna=False).to_dict())

# apply one-hot encoding
one_hot_encode_cols = ["Gender", "Cancer_Stage", "BMI", "Screening_History", "Treatment_Type"]

df_final = pd.get_dummies(df_final, columns=one_hot_encode_cols, drop_first=False)

dummy_cols = [c for c in df_final.columns if any(col + "_" in c for col in one_hot_encode_cols)]
df_final[dummy_cols] = df_final[dummy_cols].astype("Int64")

# apply label encoding
ordinal_map = {
    "low": 0,
    "moderate": 1,
    "high": 2
}

ordinal_cols = ["Diet_Risk", "Physical_Activity", "Healthcare_Access"]

for col in ordinal_cols:
    if col in df_final.columns:
        df_final[col] = (
            df_final[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(ordinal_map)
            .astype("Int64")  # nullable integer, preserves missing if any
        )


# reorder columns similar to original dataset
column_order = [
    "Age",
    # Gender dummies
    *[c for c in df_final.columns if c.startswith("Gender_")],
    # Cancer_Stage dummies
    *[c for c in df_final.columns if c.startswith("Cancer_Stage_")],
    "Tumor_mm",
    "Family_History",
    "Smoking_History",
    "Alcohol_Consumption",
    # BMI dummies
    *[c for c in df_final.columns if c.startswith("BMI_")],
    "Diet_Risk",
    "Physical_Activity",
    "Diabetes",
    "IBD",
    "Genetic_Mutation",
    # Screening_History dummies
    *[c for c in df_final.columns if c.startswith("Screening_History_")],
    "Early_Detection",
    # Treatment_Type dummies
    *[c for c in df_final.columns if c.startswith("Treatment_Type_")],
    "Survival_5_yrs",
    "HC_Costs",
    "Urban", "Rural", "Healthcare_Access", "Insured", "Uninsured"
]

column_order = [c for c in column_order if c in df_final.columns]

df_final = df_final[column_order].copy()

# make sure datatypes are set to the correct type
continuous_cols = [c for c in ["Age", "Tumor_mm", "HC_Costs"] if c in df_final.columns]
for col in continuous_cols:
    df_final[col] = pd.to_numeric(df_final[col], errors="coerce").astype("Int64")
    
ordinal_cols = [c for c in ["Diet_Risk", "Physical_Activity", "Healthcare_Access"] if c in df_final.columns]
for col in ordinal_cols:
    df_final[col] = pd.to_numeric(df_final[col], errors="coerce").astype("Int8")
    
bool_cols = [c for c in [
    "Family_History", "Smoking_History", "Alcohol_Consumption",
    "Diabetes", "IBD", "Genetic_Mutation", "Early_Detection",
    "Survival_5_yrs"
] if c in df_final.columns]

for col in bool_cols:
    # If any residual strings slipped through, normalise; else astype will be a no-op
    s = df_final[col]
    if s.dtype == object:
        s = (s.astype(str).str.strip().str.lower()
             .map({"yes": True, "y": True, "true": True, "1": True,
                   "no": False, "n": False, "false": False, "0": False}))
    df_final[col] = s.astype("boolean")  # pandas nullable Boolean (allows <NA>)
    
dummy_prefixes = ["Gender_", "Cancer_Stage_", "BMI_", "Screening_History_", "Treatment_Type_"]
dummy_cols = [c for c in df_final.columns if any(c.startswith(p) for p in dummy_prefixes)]
for col in dummy_cols:
    df_final[col] = pd.to_numeric(df_final[col], errors="coerce").astype("Int8")
    
for col in [c for c in ["Urban", "Rural", "Insured", "Uninsured"] if c in df_final.columns]:
    df_final[col] = pd.to_numeric(df_final[col], errors="coerce").astype("Int8")
    
# check if there are object type cols remaining
residual_object = [c for c in df_final.columns if df_final[c].dtype == object]
if residual_object:
    print("Note: residual object-typed columns:", residual_object)
    
# check all types and nulls
print("\n=== 3.5 DTYPE SUMMARY ===")
print(df_final.dtypes.sort_index())

print("\nNulls per column (top 20):")
print(df_final.isnull().sum().sort_values(ascending=False).head(20))

# # Assert dummy columns are truly 0/1 (or NA)
if dummy_cols:
    dmax = df_final[dummy_cols].max(numeric_only=True).max()
    dmin = df_final[dummy_cols].min(numeric_only=True).min()
    assert (dmin in (0, np.nan)) and (dmax in (1, np.nan)), \
        "Dummy columns contain values other than 0/1."

# check
print("Final shape: ", df_final.shape)
print("Final null: ", df_final.isnull().sum())
print("Final types: ", df_final.dtypes)

# export df_final as .csv
output_csv = "colorectal_cancer_final.csv"

# Export with UTF-8 encoding and without the index column
df_final.to_csv(output_csv, index=False, encoding="utf-8")

print(f"Final dataset exported successfully to {output_csv}")
print("Shape:", df_final.shape)

