#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:42:02 2019

@author: Ruman
"""

#Se basan en el uso de hiperplanos -> https://es.wikipedia.org/wiki/Hiperplano

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


# importamos los datos
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

#Comos estamos utilizando solo dos variables podemos pintar los datos en un scatterplot, diferenciando en función de la clase. 
plt.title('Datos de entrenamiento')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()


#Con un gráfico 3D se podría ver mejor.
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


#Dividimos el conjunto de datos en dos para luego aprovecharlo para Test.
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=4)

svm_classifier = SVC().fit(X_train, Y_train)
y_pred         = svm_classifier.predict(X_train)
print('Precisión:', accuracy_score(Y_train, y_pred))
print('Exactitud:', precision_score(Y_train, y_pred,average='macro'))
print('Exhaustividad:', recall_score(Y_train, y_pred,average='macro'))


y_pred         = svm_classifier.predict(X_test)
print('Precisión Test:', accuracy_score(Y_test, y_pred))
print('Exactitud Test:', precision_score(Y_test, y_pred,average='macro'))
print('Exhaustividad Test:', recall_score(Y_test, y_pred,average='macro'))

#Ejemplo para dibujar el resultado en 3D -> https://stackoverflow.com/questions/36232334/plotting-3d-decision-boundary-from-linear-svm