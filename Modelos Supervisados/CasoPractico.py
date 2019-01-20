#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 19:26:58 2019

@author: Ruman
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn.metrics as mtr

#Acceso a datos
data = pd.read_csv('winequality-white.csv', sep=';')

#Dividamos target y variables independientes
X = data.loc[:,'fixed acidity':'alcohol']
y = data.loc[:, 'quality']

#Dividimos el conjunto de datos en dos para luego aprovecharlo para Test.
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state=4)

plt.title('Datos de entrenamiento')
plt.xlabel('Alcohol')
plt.ylabel('Sulfitos')
plt.scatter(X.iloc[:,10], X.iloc[:,9], c=y)
plt.show()

#Revisamos la guía de selección de algoritmos de scikit-learn -> https://scikit-learn.org/stable/_static/ml_map.png
#Nos decantamos por un modelo SGD 

sgd_model = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train, Y_train)

#Predecimos y evaluamos el modelo
y_train_pred        = sgd_model.predict(X_train)
print('\n\n SGD')
print('La precisión con datos de train es:',mtr.accuracy_score(Y_train, y_train_pred))
print('La exactitud con datos de train es:',mtr.precision_score(Y_train, y_train_pred, average='macro'))
print('La exhaustividad con datos de train es:',mtr.recall_score(Y_train, y_train_pred, average='macro'))


print('\n\n Regresión Lineal')
#Vamos a probar con regresión lineal
reg = LinearRegression().fit(X_train, Y_train)
y_train_pred        = reg.predict(X_train)
print('Error cuadrático medio', mtr.mean_squared_error(y_train_pred, Y_train))
print('Error absoluto medio', mtr.mean_absolute_error(y_train_pred, Y_train))
print('Mediana del error absoluto', mtr.median_absolute_error(y_train_pred, Y_train))
print('R2 en entrenamiento es: ', reg.score(X_train, Y_train))


#Vamos a probar con arboles
dt_classifier_improved = DecisionTreeClassifier(criterion = 'entropy',
                                       random_state = 1,
                                       max_depth = 7).fit(X_train, Y_train)
print('\n\n Árboles decisión')
y_train_pred        = dt_classifier_improved.predict(X_train)
print('La precisión con datos de train es:',mtr.accuracy_score(Y_train, y_train_pred))
print('La exactitud con datos de train es:',mtr.precision_score(Y_train, y_train_pred, average='macro'))
print('La exhaustividad con datos de train es:',mtr.recall_score(Y_train, y_train_pred, average='macro'))

y_train_pred        = dt_classifier_improved.predict(X_test)
print('La precisión con datos de test es:',mtr.accuracy_score(Y_test, y_train_pred))
print('La exactitud con datos de test es:',mtr.precision_score(Y_test, y_train_pred, average='macro'))
print('La exhaustividad con datos de test es:',mtr.recall_score(Y_test, y_train_pred, average='macro'))


#Ninguno de los modelos utilizados se aproxima bien. Parece bastante subjetiva la calidad.
#Para mejorarlo, "Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods"
#Esto se verá mas adelante en el curso.