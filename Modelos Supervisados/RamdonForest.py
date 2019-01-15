#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 21:53:22 2019

@author: Ruman
"""

#Explicación sencilla -> http://ligdigonzalez.com/aprendizaje-supervisado-random-forest-classification/

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


# importamos los datos
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

#Dividimos el conjunto de datos en dos para luego aprovecharlo para Test.
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state=4)

#Entrenar el modelo con Random forest. Importante indicar el número de estimadores = número de árboles.
model = RandomForestClassifier(n_estimators = 5,
                               criterion='entropy',
                               max_depth = 6,
                               random_state = 1)

model.fit(X_train, Y_train)

y_pred = model.predict(X_train)

print('Precisión:', accuracy_score(Y_train, y_pred))
print('Exactitud:', precision_score(Y_train, y_pred))
print('Exhaustividad:', recall_score(Y_train, y_pred))

#LTras realizar varias pruebas con los parámetros obtenemos este que rula guay.
#Probamos con los datos de test.
y_pred = model.predict(X_test)
print('Precisión_Test:', accuracy_score(Y_test, y_pred))
print('Exactitud_Test:', precision_score(Y_test, y_pred))
print('Exhaustividad_Test:', recall_score(Y_test, y_pred))

