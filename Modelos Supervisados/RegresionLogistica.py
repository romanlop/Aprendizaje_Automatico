#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:35:57 2019

@author: Ruman
"""

from sklearn.datasets import make_classification #Crear conjuntos de datos aleatorios.
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

#vamos a crear un conjunto de datos aleatorios para casos de clasificación.
#-> https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
sample_data = make_classification(n_samples=2500,
                                  n_features=3,
                                  n_redundant=0,
                                  random_state=1)   

#Dividimos el conjunto de datos en dos para luego aprovecharlo para Test.
X_train, X_test, Y_train, Y_test = train_test_split(sample_data[0], 
                                                    sample_data[1], 
                                                    test_size=0.33, 
                                                    random_state=4)

#Vamos a crear el modelo de regresión logística
clf = LogisticRegression(random_state=0, solver='liblinear').fit(X_train, Y_train)
y_train_pred = clf.predict(X_train)
y_test_pred  = clf.predict(X_test)

#Construimos la matriz de confusión
cfm_train = confusion_matrix(Y_train, y_train_pred)
cfm_test = confusion_matrix(Y_test, y_test_pred)
print('La matriz de confusión para entrenamiento es')
print(cfm_train)
print('La matriz de confusión para test es')
print(cfm_test)
"""
Las matrices de confusión muestran que la predicción es buena, 
ya que hay pocos registros fuera de la diagonal principal, 
donde se sitúan los falsos positivos y los falsos negativos. 
Para comparar los resultados y verificar si existe o no sobreajuste, 
se pueden normalizar las matrices dividiéndolas por el número total de 
registros de cada una, obteniendo así los porcentajes de cada 
tipo de acierto y fallo.
"""
print('La matriz de confusión para entrenamiento normalizada es:')
print(cfm_train / cfm_train.sum())
print('La matriz de confusión para test normalizada es')
print(cfm_test / cfm_test.sum())

#Las matrices obtenidas son similares lo que indica que no parece haber sobreajuste.

"""
La existencia o no de sobreajuste se puede comprobar también utilizando las 
métricas que se han visto anteriormente, para lo cual se han de importar las 
funciones accuracy_score, precision_score y recall_score para obtener la 
precisión, exactitud y exhaustividad, respectivamente.
"""
print('La precisión con datos de train es:',accuracy_score(Y_train, y_train_pred))
print('La precisión con datos de test es:',accuracy_score(Y_test, y_test_pred))
print('La exactitud con datos de train es:',precision_score(Y_train, y_train_pred))
print('La exactitud con datos de test es:',precision_score(Y_test, y_test_pred))
print('La exhaustividad con datos de train es:',recall_score(Y_train, y_train_pred))
print('La exhaustividad con datos de test es:',recall_score(Y_test, y_test_pred))

#Con estos datos confirmamos que no parece haber sobreajuste.
false_positive_rate, recall, thresholds = roc_curve(Y_test, y_test_pred)
roc_auc = auc(false_positive_rate, recall)

print('AUC:', auc(false_positive_rate, recall))

plt.plot(false_positive_rate, recall, 'b')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('AUC = %0.2f' % roc_auc)
plt.show

