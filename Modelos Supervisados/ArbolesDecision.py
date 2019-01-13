#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:35:26 2019

@author: Ruman

Vamos a tratar de hacerlo con datos de cáncer de mama.
"""

#Ejemplo visual muy sencillo: https://www.youtube.com/watch?v=269QJ5joMCc

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import numpy as np

# import some data to play with
cancer = datasets.load_breast_cancer()
X = cancer.data  # we only take the first two features.
y = cancer.target


#Comos estamos utilizando solo dos variables podemos pintar los datos en un scatterplot, diferenciando en función de la clase. 
plt.title('Datos de entrenamiento')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

#Dividimos el conjunto de datos en dos para luego aprovecharlo para Test.
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.33, 
                                                    random_state=4)

#Entrenamos el modelo.
dt_classifier = DecisionTreeClassifier(criterion = 'entropy',
                                       random_state = 1).fit(X_train, Y_train)

#Predecimos y evaluamos el modelo
y_train_pred        = dt_classifier.predict(X_train)
print('La precisión con datos de train es:',accuracy_score(Y_train, y_train_pred))
print('La exactitud con datos de train es:',precision_score(Y_train, y_train_pred))
print('La exhaustividad con datos de train es:',recall_score(Y_train, y_train_pred))

y_train_test       = dt_classifier.predict(X_test)
print('La precisión con datos de train es:',accuracy_score(Y_test, y_train_test))
print('La exactitud con datos de train es:',precision_score(Y_test, y_train_test))
print('La exhaustividad con datos de train es:',recall_score(Y_test, y_train_test))
#-> Vemos que no clasifica muy bien. Puede ser debido al parámetro max_depth que por defecto es ilimitado y tiende a sobreajustar.


for i in range(3):
    print('##################################################################')
#Entrenamos el modelo especificando un max_depth.
dt_classifier_improved = DecisionTreeClassifier(criterion = 'entropy',
                                       random_state = 1,
                                       max_depth = 5).fit(X_train, Y_train)

#Predecimos y evaluamos el modelo
y_train_pred        = dt_classifier_improved.predict(X_train)
print('La precisión con datos de train es:',accuracy_score(Y_train, y_train_pred))
print('La exactitud con datos de train es:',precision_score(Y_train, y_train_pred))
print('La exhaustividad con datos de train es:',recall_score(Y_train, y_train_pred))

y_train_test       = dt_classifier_improved.predict(X_test)
print('La precisión con datos de train es:',accuracy_score(Y_test, y_train_test))
print('La exactitud con datos de train es:',precision_score(Y_test, y_train_test))
print('La exhaustividad con datos de train es:',recall_score(Y_test, y_train_test))
#Vemos que ha mejorado con una profundidaz de 5, a partir de aquí sobreajusta.


false_positive_rate, recall, thresholds = roc_curve(Y_test, y_train_test)
roc_auc = auc(false_positive_rate, recall)

print('AUC:', auc(false_positive_rate, recall))

plt.plot(false_positive_rate, recall, 'b')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('AUC = %0.2f' % roc_auc)
plt.show()

#Es la importancia de cada variable independiente sobre la dependiente
print(dt_classifier_improved.feature_importances_)

for i in range(3):
    print('##################################################################')
#Vamos a tratar de acceder/almacenar la información del modelo que acabamos de construir y pintar el árbol.

export_graphviz(dt_classifier_improved, 
                out_file='arbol.dot', 
                class_names=cancer.target_names,
                feature_names=cancer.feature_names,
                impurity=False,
                filled=True)

