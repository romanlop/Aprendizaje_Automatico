#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:05:34 2019

Objetivo:
Utilizar la clase RadiusNeighborsClassifier para crear un clasificador 
que permita identificar la especie de Iris del conjunto de datos Iris en 
base al largo y ancho de los sépalos.

Probar el funcionamiento con los radios de 0,3, 0,5, 0,7, 0,9 y 1,1.

@author: Ruman
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn import datasets
from matplotlib.colors import ListedColormap

# Importamos iris y nos quedamos con longitud de anchura de los sépalos.
iris = datasets.load_iris()
data = iris.data[:, :2]  
target = iris.target

#Comos estamos utilizando solo dos variables podemos pintar los datos en un scatterplot, diferenciando en función de la clase. 
plt.title('Datos de entrenamiento')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.scatter(data[:,0], data[:,1], c=target)
plt.show()

"""Entrenamos el modelo con 0.3: Devuelve un error pq hay valores anómalos/lejanos 
que no son recogidos por radio. Hay que eliminarlos o asignarles una predicción fija."""
model = RadiusNeighborsClassifier(radius = 0.3, outlier_label=4)
model.fit(data, target)

# Version clara y oscura de los coloes
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AAAAAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                            
#Calculamos los márgenes para los ejes y sumamos 1 para que haya un margen
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

#A partir de estos valores se crea una malla con una separación de 0,05. 
#En todos los puntos de esta malla se calcula una predicción con el modelo y se representa en una gráfica.    
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

#ravel devuelve una matriz aplanada contigua.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.c_.html
Z = Z.reshape(xx.shape)  

plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
plt.scatter(data[:, 0], data[:, 1], c=target, cmap = cmap_bold)
plt.ylim(yy.min(), yy.max())  
plt.title('Predicción con 0.3')
plt.show()


"""Entrenamos el modelo con 0.5"""
model = RadiusNeighborsClassifier(radius = 0.5, outlier_label=4)
model.fit(data, target)


#A partir de estos valores se crea una malla con una separación de 0,05. 
#En todos los puntos de esta malla se calcula una predicción con el modelo y se representa en una gráfica.    
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

#ravel devuelve una matriz aplanada contigua.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.c_.html
Z = Z.reshape(xx.shape)  

plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
plt.scatter(data[:, 0], data[:, 1], c=target, cmap = cmap_bold)
plt.ylim(yy.min(), yy.max())  
plt.title('Predicción con 0.5')
plt.show()


"""Entrenamos el modelo con 0.7"""
model = RadiusNeighborsClassifier(radius = 0.7, outlier_label=4)
model.fit(data, target)


#A partir de estos valores se crea una malla con una separación de 0,05. 
#En todos los puntos de esta malla se calcula una predicción con el modelo y se representa en una gráfica.    
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

#ravel devuelve una matriz aplanada contigua.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.c_.html
Z = Z.reshape(xx.shape)  

plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
plt.scatter(data[:, 0], data[:, 1], c=target, cmap = cmap_bold)
plt.ylim(yy.min(), yy.max())  
plt.title('Predicción con 0.7')
plt.show()

"""Entrenamos el modelo con 1.1"""
model = RadiusNeighborsClassifier(radius = 1.1, outlier_label=4)
model.fit(data, target)


#A partir de estos valores se crea una malla con una separación de 0,05. 
#En todos los puntos de esta malla se calcula una predicción con el modelo y se representa en una gráfica.    
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

#ravel devuelve una matriz aplanada contigua.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.c_.html
Z = Z.reshape(xx.shape)  

plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
#plt.scatter(data[:, 0], data[:, 1], c=target, cmap = cmap_bold)
plt.ylim(yy.min(), yy.max())  
plt.title('Predicción con 1.1')
plt.show()

"""Entrenamos el modelo con 2"""
model = RadiusNeighborsClassifier(radius = 2, outlier_label=4)
model.fit(data, target)


#A partir de estos valores se crea una malla con una separación de 0,05. 
#En todos los puntos de esta malla se calcula una predicción con el modelo y se representa en una gráfica.    
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

#ravel devuelve una matriz aplanada contigua.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.c_.html
Z = Z.reshape(xx.shape)  

plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
#plt.scatter(data[:, 0], data[:, 1], c=target, cmap = cmap_bold)
plt.ylim(yy.min(), yy.max())  
plt.title('Predicción con 2')
plt.show()