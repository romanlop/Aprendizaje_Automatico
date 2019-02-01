#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 19:48:54 2019

@author: Ruman

En algunas ocasiones, el método de la distorsión no ofrece resultados porque los 
clústeres no se encuentran tan claramente separados como en el ejemplo anterior. 
En estos casos se puede utilizar el método de la Silhouette.

El coeficiente de la Silhouette se define como la diferencia entre la distancia 
media a los elementos del clúster más cercano (b) y a distancia intra-clúster 
media de los elementos de un clúster (a) dividido por el máximo de los dos

El objetivo es maximizar la shiloutte_score
"""
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics


def clasificacion_silhouette (nombre, n_clusters, data):
    silhouette = []
    #Para silhouette hay que partir de 2 clusters. En caso contrario casca. No tiene sentido aplicarlo.
    for i in range(2,n_clusters+1):
        kmeans = KMeans(n_clusters = i,
                        random_state = 1).fit(data)
        labels = kmeans.labels_
        silhouette.append(metrics.silhouette_score(data, labels))
        
    plt.plot(range(2,n_clusters+1), silhouette, 'ro-')   
    plt.title("Silhouette Coeficcient: " + nombre)
    plt.xlabel("Número Clústers")
    plt.ylabel("silhouette")

blobs_3, classes_3 = make_blobs(300,
                                centers      = 3,
                                cluster_std  = 0.5,
                                random_state = 1)
blobs_5, classes_5 = make_blobs(300,
                                centers      = 5,
                                cluster_std  = 0.5,
                                random_state = 1)

plt.subplot(1, 2, 1)
clasificacion_silhouette ("blobs3", 10, blobs_3)
plt.subplot(1, 2, 2)
clasificacion_silhouette ("blobs5", 10, blobs_5)
plt.show()

print("Obtenemos 2 y 5 clústers como valores óptimos.")


