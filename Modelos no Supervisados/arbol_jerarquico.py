#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:51:56 2019

@author: Ruman
En esta sección únicamente se va a estudiar el enfoque aglomerativo de enlace 
complejo, siendo este un procedimiento iterativo que se puede resumir en los siguientes pasos:

1. Calcular la matriz de distancias entre todos los registros.
2. Representar cada punto como un clúster único.
3. Combinar los dos clústeres más cercanos en función de la distancia de los miembros más separados.
4. Actualizar la distancia entre todos los nuevos clústeres.
5. Repetir los pasos de 3 a 5 hasta que quede un único clúster.
"""
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist, squareform

blobs_3, classes_3 = make_blobs(300,
                                centers      = 3,
                                cluster_std  = 0.5,
                                random_state = 1)
blobs_5, classes_5 = make_blobs(300,
                                centers      = 5,
                                cluster_std  = 0.5,
                                random_state = 1)

"""
En la implementación del algoritmo, el primer paso es la construcción de la matriz 
de distancias. Para facilitar la comprensión de los resultados, se utilizarán únicamente 
los cinco primeros registros del conjunto de datos blobs_3 utilizado previamente. 
Para el cálculo se utiliza el código de la figura 3.24
"""

import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame(blobs_3[0:5, :])
row_dist = pd.DataFrame(squareform(pdist(df, metric = 'euclidean')))
print (row_dist)

plt.title("Datos Training.")
plt.scatter(blobs_3[0:5,0],blobs_3[0:5,1])
plt.show()

from scipy.cluster.hierarchy import linkage

row_clusters = linkage(pdist(df, metric = 'euclidean'),
                       method = 'complete')

result = pd.DataFrame(row_clusters,
             columns = ['row 1', 'row 2', 'distance', 'items in cluster'],
             index = ['cluster %d' %(i) for i in range(row_clusters.shape[0])])

from scipy.cluster.hierarchy import dendrogram
dendrogram(row_clusters)


#La forma mas sencilla de hacerlo es la siguiente y además reduce la complejidad de andar calculando las distancias. 
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters = 2,
                             affinity = 'euclidean',
                             linkage = 'complete').fit_predict(df)
