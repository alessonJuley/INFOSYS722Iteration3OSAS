#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 00:34:29 2025

@author: alessonabao
"""

# =====06-DMA=====
# resouce: https://pypi.org/project/random-survival-forest/
# cannot apply RSF because dataset does not have time until event occurs column
import pandas as pd

# load datasets
dataset_reduced = 'colorectal_cancer_reduced.csv'
df_reduced = pd.read_csv(dataset_reduced)
# 167497 rows, 24 cols
df_reduced.shape
df_reduced.info()

# =====6.3=====
# before starting: pip install random-survival-forest
import time

from lifelines import datasets
from sklearn.model_selection import train_test_split

from random_survival_forest.models import RandomSurvivalForest
from random_survival_forest.scoring import concordance_index

# value next to Survival_5_yrs should be time until event occurs (I don't have it in my dataset)
y = df_reduced.loc[:, ["Survival_5_yrs", ""]]

# iterate to other algorithm