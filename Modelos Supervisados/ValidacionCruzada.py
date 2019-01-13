#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:38:24 2019

@author: Ruman
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# import some data to play with
boston = load_boston()

#Entrenamiento del modelo
model = LinearRegression().fit(boston.data, boston.target)
print("R^2:", model.score(boston.data, boston.target))

#Dividimos los datos en dos grupos.
X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, test_size=0.33, random_state=4)

#Entrenamos el modelo con el subconjunto correspondiente
model = LinearRegression().fit(X_train, Y_train)
print("R^2 train:", model.score(X_train, Y_train))
print("R^2 test:", model.score(X_test, Y_test))