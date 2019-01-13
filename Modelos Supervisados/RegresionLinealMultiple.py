#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 18:59:14 2019

@author: Ruman
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.metrics as mtr
from sklearn.pipeline import Pipeline
import numpy as np

# Conjunto de datos
X = [[80], [79], [83], [84], [78], [60], [82], [85], [79], [84], [80], [62]]
y = [[300], [302], [315], [330], [300], [250], [300], [340], [315], [330], [310], [240]]

plt.title('Datos de entrenamiento')
plt.scatter(X,y)
plt.xlabel('Horas')
plt.ylabel('Producción')
plt.show()

#Creamos el modelo polinómico -> https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
#Lo que hacemos es transformar los datos para luego utilizarlos en un modelo lineal
poly = PolynomialFeatures(degree=2)
poly.fit_transform(X)


"""
Polinomio grado 2
"""
#Con las herramientas de Pipeline se puede simplificar, del siguiente modo.
#include_bias False, permite no incluir el termino independiente. En este problema concreto no nos interesa.
model = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias= False)),
                  ('linear', LinearRegression(fit_intercept=False))])
model = model.fit(X,y)

#Vamos a obtener los valores del modelo
print(model.named_steps['linear'].coef_)

y_pred=model.predict(X)
print('Otra forma de calcular el R^2', mtr.r2_score(y_pred, y))
print('Error cuadrático medio', mtr.mean_squared_error(y_pred, y))
print('Error absoluto medio', mtr.mean_absolute_error(y_pred, y))
print('Mediana del error absoluto', mtr.median_absolute_error(y_pred, y))

#Vamos a tratar de pintar la predicción. 
testdata = np.arange(0, 90, 0.5)
predicted = model.predict(testdata.reshape(-1, 1))

"""
Polinomio grado 3
"""
model3 = Pipeline([('poly', PolynomialFeatures(degree=3, include_bias= False)),
                  ('linear', LinearRegression(fit_intercept=False))])
model3 = model3.fit(X,y)

#Vamos a obtener los valores del modelo
print("Coeficientes con grado 3:", model3.named_steps['linear'].coef_)

y_pred=model3.predict(X)
print('Otra forma de calcular el R^2', mtr.r2_score(y_pred, y))
print('Error cuadrático medio', mtr.mean_squared_error(y_pred, y))
print('Error absoluto medio', mtr.mean_absolute_error(y_pred, y))
print('Mediana del error absoluto', mtr.median_absolute_error(y_pred, y))

predicted3 = model3.predict(testdata.reshape(-1, 1))
"""
Polinomio grado 5
"""
model5 = Pipeline([('poly', PolynomialFeatures(degree=5, include_bias= False)),
                  ('linear', LinearRegression(fit_intercept=False))])
model5 = model5.fit(X,y)

#Vamos a obtener los valores del modelo
print("Coeficientes con grado 5:", model5.named_steps['linear'].coef_)

y_pred=model5.predict(X)
print('Otra forma de calcular el R^2', mtr.r2_score(y_pred, y))
print('Error cuadrático medio', mtr.mean_squared_error(y_pred, y))
print('Error absoluto medio', mtr.mean_absolute_error(y_pred, y))
print('Mediana del error absoluto', mtr.median_absolute_error(y_pred, y))

predicted5 = model5.predict(testdata.reshape(-1, 1))
plt.title('Datos predichos con polinomios de diferentes grados')
plt.plot(testdata,predicted,c='b',label='Grado 2')
plt.plot(testdata,predicted3,c='r',label='Grado 3')
plt.plot(testdata,predicted5,c='g',label='Grado 5')
plt.scatter(X,y)
plt.xlabel('Horas')
plt.ylabel('Producción')
plt.legend(loc = 2)
plt.show()