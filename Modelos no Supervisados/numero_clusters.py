#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 19:11:06 2019

@author: Ruman

Uno de los métodos más fáciles de implementar y de interpretar para la 
selección del número de clústeres es el de la distorsión o método del codo.
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


"""
Vamos a crear una función para calcular el K-means con diferentes números de clústers.

Inertia:  
El algoritmo KMeans agrupa los datos al tratar de separar muestras en n grupos 
de igual varianza, minimizando un criterio conocido como la inercia o la suma 
de cuadrados dentro del clúster.

La inercia es una medida del grado de coherencia de los clusteres. La inercia es 
la suma de las distancias al cuadrado de cada objeto del Cluster a su centroide:

La inercia no es una métrica normalizada: solo sabemos que los valores más bajos 
son mejores y que el cero es óptimo.

Pero en espacios de muy alta dimensión, las distancias euclidianas tienden a inflarse 
(esto es un ejemplo de la llamada "maldición de la dimensionalidad"). 
La ejecución de un algoritmo de reducción de dimensionalidad como PCA antes del agrupamiento 
de k-means puede aliviar este problema y acelerar los cálculos.
"""

def clasificacion (nombre, n_clusters, data):
    inertia = []
    for i in range(1,n_clusters+1):
        kmeans = KMeans(n_clusters = i,
                        random_state = 1).fit(data)
        inertia.append(kmeans.inertia_)
        
    plt.plot(range(1,n_clusters+1), inertia, 'ro-')   
    plt.title("Dispersión: " + nombre)
    plt.xlabel("Número Clústers")
    plt.ylabel("Inertia / Dispersión")

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

#Resultados con 10 clústers y blobs_3 (vemos que según esto lo óptimo son 3 clústers)
plt.subplot(1, 2, 1)
clasificacion ("blobs3", 10, blobs_3)
plt.subplot(1, 2, 2)
clasificacion ("blobs5", 10, blobs_5)
plt.show()

