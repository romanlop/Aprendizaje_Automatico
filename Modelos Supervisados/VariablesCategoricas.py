#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:10:34 2019

@author: Ruman
"""

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mtr
from sklearn.pipeline import Pipeline

# Conjunto de datos con las factorias
x = [[80, 'Factoria 1'], [79, 'Factoria 2'], [83, 'Factoria 3'],
     [84, 'Factoria 1'], [78, 'Factoria 2'], [60, 'Factoria 3'],
     [82, 'Factoria 1'], [85, 'Factoria 2'], [79, 'Factoria 3'],
     [84, 'Factoria 1'], [80, 'Factoria 2'], [62, 'Factoria 3']]
y = [[300], [302], [315], [330], [300], [250], [300], [340], [315], [330], [310], [240]]

# Conversion de las datos a DataFrame
x_0 = pd.DataFrame(x, columns = ['Horas', 'Factoria'])
y = pd.DataFrame(y)

# Vamos a crear las variables dummy correspondientes
x_dummy=pd.get_dummies(x_0)


#Vamos a hacer una regresión lineal con un polinomio de grado 2
model = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias= False)),
                  ('linear', LinearRegression(fit_intercept=False))])
model = model.fit(x_dummy,y)

y_pred=model.predict(x_dummy)
print('R^2', mtr.r2_score(y_pred, y))
print('Error cuadrático medio', mtr.mean_squared_error(y_pred, y))
print('Error absoluto medio', mtr.mean_absolute_error(y_pred, y))
print('Mediana del error absoluto', mtr.median_absolute_error(y_pred, y))

