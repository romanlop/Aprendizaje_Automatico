#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:24:56 2019

@author: Ruman
"""

import matplotlib.pyplot as plt
from sklearn import datasets
#Importamos los algoritmos de scikit learn que nos interesan -> https://scikit-learn.org/stable/modules/neighbors.html
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import numpy as np

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

#Comos estamos utilizando solo dos variables podemos pintar los datos en un scatterplot, diferenciando en función de la clase. 

plt.title('Datos de entrenamiento')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.scatter(x[:,0], x[:,1], c=y)
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


###############################################################################
#Entrenamos el modelo
model = KNeighborsClassifier(15)
# Entrenamos el modelo:
model.fit(x, y)

# Version clara y oscura de los coloes
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#Calculamos los márgenes para los ejes y sumamos 1 para que haya un margen
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                            
#A partir de estos valores se crea una malla con una separación de 0,05. 
#En todos los puntos de esta malla se calcula una predicción con el modelo y se representa en una gráfica.    
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))


#ravel devuelve una matriz aplanada contigua.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.c_.html
Z = Z.reshape(xx.shape)  

plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap = cmap_bold)
#¡plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())  
plt.show()
      
                                                    