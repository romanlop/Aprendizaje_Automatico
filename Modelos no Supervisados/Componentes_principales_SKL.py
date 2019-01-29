#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:07:40 2019

@author: Ruman
"""

from sklearn.decomposition import PCA
import numpy as np

##############################################################################
#Primer ejemplo sencillo.
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)

pca.fit(X)

print(pca.explained_variance_ratio_) 

#Ver documentación PCA y https://en.wiktionary.org/wiki/two-norm
print(pca.singular_values_)  

#No podemos poner mas de 2 componentes
print(pca.n_features_)


#############################################################################
"""
svd_solver : string {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
auto :
the solver is selected by a default policy based on X.shape and n_components: if the input data is larger than 500x500 and the number of components to extract is lower than 80% of the smallest dimension of the data, then the more efficient ‘randomized’ method is enabled. Otherwise the exact full SVD is computed and optionally truncated afterwards.

full :
run exact full SVD calling the standard LAPACK solver via scipy.linalg.svd and select the components by postprocessing

arpack :
run SVD truncated to n_components calling ARPACK solver via scipy.sparse.linalg.svds. It requires strictly 0 < n_components < min(X.shape)

randomized :
run randomized SVD by the method of Halko et al.
"""
pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)


pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

#############################################################################
"""
El PCA se puede utilizar para la representación de conjuntos de datos que 
tengan más de dos dimensiones.
Por ejemplo, el conjunto de datos Iris disponible entre los ejemplos 
de scikit-learn. Este conjunto de datos dispone de 50 muestras de tres 
especies de flor, en los que se han medido cuatro características de cada 
flor: el largo y el ancho de los sépalos y pétalos.

VER ESTO!!!
https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
"""

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

iris_names  = datasets.load_iris().target_names
iris_target = datasets.load_iris().target
iris_values = datasets.load_iris().data

#Pintamos los datos en función de las dos primeras variables
#Comos estamos utilizando solo dos variables podemos pintar los datos en un scatterplot, diferenciando en función de la clase. 
plt.title("Datos Originales.")
plt.scatter(iris_values[iris_target == 0, 0], iris_values[iris_target == 0, 1], c='r')
plt.scatter(iris_values[iris_target == 1, 0], iris_values[iris_target == 1, 1], c='g')
plt.scatter(iris_values[iris_target == 2, 0], iris_values[iris_target == 2, 1], c='b')
plt.legend(iris_names)
plt.show()

"""
IMPORTANTE!
PCA se realiza por escala, por lo que es bueno escalar/estandarizar los datos de cada 
característica antes de aplicar PCA.
Use StandardScaler para ayudarlo a estandarizar las características del 
conjunto de datos en la escala de la unidad (media = 0 y varianza = 1), 
que es un requisito para el rendimiento óptimo de muchos algoritmos de 
aprendizaje automático. 
"""

iris_values = StandardScaler().fit_transform(iris_values)

plt.title("Datos Estandarizados.")
plt.scatter(iris_values[iris_target == 0, 0], iris_values[iris_target == 0, 1], c='r')
plt.scatter(iris_values[iris_target == 1, 0], iris_values[iris_target == 1, 1], c='g')
plt.scatter(iris_values[iris_target == 2, 0], iris_values[iris_target == 2, 1], c='b')
plt.legend(iris_names)
plt.show()

#Hence, the None case results in:
#n_components == min(n_samples, n_features) - 1
iris_pca = PCA(n_components=None)
iris_pca.fit(iris_values)

print("Número de componentes", iris_pca.n_components_)
#print("Ratio de varianza explicada", iris_pca.explained_variance_ratio_) 
#Importante es que la suma es 1. Con 4 componentes explicas el 100% de la varianza. Con menos componentes explicas menos.
for i in range(np.shape(iris_pca.explained_variance_ratio_)[0]):  
    explained_var = iris_pca.explained_variance_ratio_[range(i + 1)].sum()
    print("Varianza explicada con", i + 1, "componentes:", explained_var)


#Para poder representarlos, vamos a utilizar 2 componentes
iris_pca = PCA(n_components=2)
iris_pca.fit(iris_values)
#Transformamos los datos en base a los nuevos componentes
componentes_principales = iris_pca.fit_transform(iris_values)

plt.title("Datos sobre componentes principales.")
plt.scatter(componentes_principales[iris_target == 0, 0], componentes_principales[iris_target == 0, 1], c='r')
plt.scatter(componentes_principales[iris_target == 1, 0], componentes_principales[iris_target == 1, 1], c='g')
plt.scatter(componentes_principales[iris_target == 2, 0], componentes_principales[iris_target == 2, 1], c='b')
plt.legend(iris_names)
plt.show()


#Podríamos utlizar 3 componmentes y dibujarlo en 3D.