#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 19:10:30 2019

@author: Ruman

Los algoritmos K-Means y MiniBatchKMeans usan siempre distancia euclídea. 
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

"""
En esta ocasión se utilizará la función make_blobs para crear datos de prueba.
Vamos a usar make_blobs
Esta función genera aleatoriamente datos para un número de clases indicadas en 
torno a unos centroides por lo que son ideales para este caso de uso.

Podemos definir la desviación estándar de cada cluster.
"""

blobs_3, classes_3 = make_blobs(300,
                                centers      = 3,
                                cluster_std  = 0.5,
                                random_state = 1)
blobs_5, classes_5 = make_blobs(300,
                                centers      = 5,
                                cluster_std  = 0.5,
                                random_state = 1)


plt.title("Datos Training.")
plt.scatter(blobs_3[:,0],blobs_3[:,1])
plt.show()

plt.title("Datos Test.")
plt.scatter(blobs_5[:,0],blobs_5[:,1])
plt.show()

#Por defecto ejecuta el KMeans 10 veces
kmeans = KMeans(n_clusters = 3,
                random_state = 1).fit(blobs_3)


#La clase de cada uno de los registros se obtiene utilizando el método predict 
#del objeto entrenado en el conjunto de datos. Finalmente, se representa la 
#muestra de datos estableciendo un color en función del clúster asignado en la predicción.
classes = kmeans.predict(blobs_3)

color_map = np.array(['b','g','r','c','m','y','k'])

#Entrenamiento datos Test
plt.title("Datos Test.")
plt.scatter(blobs_3[:,0],blobs_3[:,1], color = color_map[classes])
plt.scatter(kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker = '*',
        s = 250,
        color = 'black')
plt.show()

#Vamos a ejecutarlo sobre el conjunto de 5 centros sin cambiar el número de clúster,
#veremos que el resultado no es correcto.

#Por defecto ejecuta el KMeans 10 veces
kmeans = KMeans(n_clusters = 3,
                random_state = 1).fit(blobs_5)

#La clase de cada uno de los registros se obtiene utilizando el método predict 
#del objeto entrenado en el conjunto de datos. Finalmente, se representa la 
#muestra de datos estableciendo un color en función del clúster asignado en la predicción.
classes = kmeans.predict(blobs_5)

#Entrenamiento datos Test
plt.title("Datos Test.")
plt.scatter(blobs_5[:,0],blobs_5[:,1], color = color_map[classes])
plt.scatter(kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker = '*',
        s = 250,
        color = 'black')
plt.show()

#Pasaría lo mismo si utilizamos demasiados clusters, por lo que es clave lo que se explicará en la siguiente lección.