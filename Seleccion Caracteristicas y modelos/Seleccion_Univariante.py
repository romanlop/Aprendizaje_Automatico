#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:36:17 2019

@author: Ruman
"""
import pandas as pd
from sklearn.feature_selection import VarianceThreshold   #https://scikit-learn.org/stable/modules/feature_selection.html

wine = pd.read_csv('winequality-white.csv', sep = ';')

target = 'quality'
features = list(wine.columns)
features.remove('quality')

x = wine[features]
y = wine[target]

