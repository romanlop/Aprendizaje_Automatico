#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 21:29:37 2019

@author: Ruman
Finalmente, se va a estudiar otro algoritmo existente dentro de la familia de análisis de clúster: 
    el agrupamiento espacial basado en densidad de aplicaciones con ruido 
    (DBSCAN, Density-Based Spatial Clustering of Applications with Noise). 
    
La densidad es el número de puntos que se encuentran en un radio X del punto

Importante:
    Una de las principales ventajas de DBSCAN es que no asume que los clústeres 
    tienen que ser necesariamente esféricos, como en el caso de k-means, 
    por lo que se puede utilizar para identificar objetos con formas irregulares.
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np

blobs_3, classes_3 = make_blobs(300,
                                centers      = 3,
                                cluster_std  = 0.5,
                                random_state = 1)
blobs_5, classes_5 = make_blobs(300,
                                centers      = 5,
                                cluster_std  = 0.5,
                                random_state = 1)

db = DBSCAN(eps = 0.5,
            min_samples = 5,
            metric = 'euclidean').fit_predict(blobs_3)

km = KMeans(n_clusters = 3,
            random_state = 1).fit_predict(blobs_3)

ac = AgglomerativeClustering(n_clusters = 3, 
                             affinity = 'euclidean',
                             linkage = 'complete').fit_predict(blobs_3)

color_map = np.array(['b','g','r','c','m','y','k'])

plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.title('k-means', size=18)
plt.scatter(blobs_3[:, 0], blobs_3[:, 1], s=10, color = color_map[km])
plt.subplot(1, 3, 2)
plt.title('AgglomerativeClustering', size=18)
plt.scatter(blobs_3[:, 0], blobs_3[:, 1], s=10, color = color_map[ac])
plt.subplot(1, 3, 3)
plt.title('DBSCAN', size=18)
plt.scatter(blobs_3[:, 0], blobs_3[:, 1], s=10, color = color_map[db])

#IMPORTANTE. DB SCAN deja algunos puntos sin clasificar!!!

