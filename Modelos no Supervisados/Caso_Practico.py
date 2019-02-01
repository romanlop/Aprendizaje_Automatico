#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 22:47:33 2019

@author: Ruman
En el archivo “mammals.csv” se encuentra una lista de mamíferos y los constituyentes 
de su leche. A partir de esta información, segmenta los mamíferos sobre la base de 
los constituyentes de la leche y obtén los valores promedios de la leche para cada 
grupo de animales.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Definimos dos funciones
def clasificacion_silhouette (n_clusters, data):
    silhouette = []
    for i in range(2,n_clusters+1):
        kmeans = KMeans(n_clusters = i,
                        random_state = 1).fit(data)
        labels = kmeans.labels_
        silhouette.append(metrics.silhouette_score(data, labels))
        
    plt.plot(range(2,n_clusters+1), silhouette, 'ro-')   
    plt.title("Silhouette Coeficcient.")
    plt.xlabel("Número Clústers")
    plt.ylabel("silhouette")   
        
    
def clasificacion_inertia (n_clusters, data):
    inertia = []
    for i in range(1,n_clusters+1):
        kmeans = KMeans(n_clusters = i,
                        random_state = 1).fit(data)
        inertia.append(kmeans.inertia_)
        
    plt.plot(range(1,n_clusters+1), inertia, 'ro-')   
    plt.title("Dispersión.")
    plt.xlabel("Número Clústers")
    plt.ylabel("Inertia / Dispersión")
    

#Accedemos al fichero y prepararmos los datos

with open('data/mammals.csv', 'rb') as csvfile:
     animales = pd.read_csv(csvfile)

data = animales.loc[ : , ['water','protein','fat','lactose','ash'] ]
target = animales.loc[ : , ['name'] ]

  
#Ejecutamos ambas funciones 
plt.subplot(1, 2, 1)
clasificacion_silhouette (10, data)
plt.subplot(1, 2, 2)
clasificacion_inertia (10, data)
plt.show()   

#Según silhouette podríamos crear 3 clústers. Según inertia podrían ser 3 o 4.

#Vamos a realizar una reducción de la dimensionalidad para representar los datos y ver si esto nos ayuda a decidir.
animales_mejorado = StandardScaler().fit_transform(data)
animales_pca = PCA(n_components=2)
animales_pca.fit(data)

#Transformamos los datos en base a los nuevos componentes
componentes_principales = animales_pca.fit_transform(data)

#Con esto tampoco lo veo nada claro. Podríamos crear 
plt.title("Datos sobre componentes principales.")
plt.scatter(componentes_principales[:,0],componentes_principales[:,1])

#Vamos a ejecutar el Kmeans con 3
kmeans = KMeans(n_clusters = 3,
                random_state = 1).fit(data)
classes = kmeans.predict(data)
target['KMeans 3'] = classes

#Vamos a ejecutar el Kmeans con 4
kmeans = KMeans(n_clusters = 4,
                random_state = 1).fit(data)
classes = kmeans.predict(data)
target['KMeans 4'] = classes

#DBSCAN
db = DBSCAN(eps = 3,
            min_samples = 3,
            metric = 'euclidean').fit_predict(data)

target['DBSCAN'] = db