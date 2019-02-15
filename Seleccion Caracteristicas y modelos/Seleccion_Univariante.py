#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:36:17 2019

@author: Ruman
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold   #https://scikit-learn.org/stable/modules/feature_selection.html
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression 

wine = pd.read_csv('data/winequality-white.csv', sep = ';')

target = 'quality'
features = list(wine.columns)
features.remove('quality')

x = wine[features]
y = wine[target]

#######################################
#Selección en función de la varianza.
selector = VarianceThreshold()
selection = selector.fit_transform(x) #aplicando el valor por defecto no elimina nada de la selección ya que solo elimina varianza = 0. 

#Vamos a echar un vistazo a las varianzas para ver que umbral podríamos tener en cuenta.
print("Varianzas:\n", pd.DataFrame.var(x,axis=0),"\n")

#Vemos que por ejemplo 0.01 puede ser un buen valor. Se cepilla un par de variables: densidad, chlorides.
selector = VarianceThreshold(threshold=0.01)
selection = selector.fit_transform(x)

print("Listado de variables por varianza ", np.asarray(list(x))[selector.get_support()])


#######################################
#Selección K_Beast.
X_new = SelectKBest(chi2, k=9)
selection = X_new.fit_transform(x, y)

print("\nListado de mediante KBeast CHI ", np.asarray(list(x))[X_new.get_support()])


X_new = SelectKBest(f_regression, k=9)
selection = X_new.fit_transform(x, y)

print("\nListado de mediante KBeast f-regression", np.asarray(list(x))[X_new.get_support()])


#######################################
#Percentile
var_sp = SelectPercentile(f_regression, percentile = 50)
x_sp = var_sp.fit_transform(x, y)

print("\nVariables finales ", x_sp.shape[1])
print("Listado de variables Percentile", np.asarray(list(x))[var_sp.get_support()])