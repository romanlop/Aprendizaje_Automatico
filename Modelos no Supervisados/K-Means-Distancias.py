#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 17:42:29 2019

@author: Ruman
"""

#Existen diferentes formas de medir la distancia entre dos puntos. Esto es fundamental para el análisis de clúster.

"""
Distancia euclídea
Explicación 
    https://www.youtube.com/watch?v=QVK1uY8lC2M
    
    
Para espacios no físicos, donde se utilizan diferentes unidades de medida, 
se utiliza mas las Distancia Euclidea normalizada
 El uso de esta distancia ofrece dos ventajas importantes frente a la euclídea. 
 La primera es la independencia de los resultados de las unidades utilizadas en cada una de las coordenadas; 
 la segunda es la de ajustar el peso de las coordenadas en función de su varianza, 
 lo que hace que no pese igual una separación de un euro o dólar en el sueldo de 
 un cliente que un hijo en la segmentación de clientes, como sucede en la euclídea
 
 Existen otras como la Manhatan...
"""

#Vamos a probar con caracteristicas 2D o 3D y con mas dimensiones para ver que pasa. 

from scipy.spatial.distance import cdist
from sklearn import datasets
import numpy as np


print("Distancias:")
print(cdist([[0, 1]], [[1, 2]], 'euclidean'))
print(cdist([[0, 1]], [[1, 2]], 'minkowski', p = 3))
print(cdist([[0, 1]], [[1, 2]], 'cityblock'))
print(cdist([[0, 1]], [[1, 2]], 'cosine'))
print(cdist([[0, 1]], [[1, 2]], 'correlation'))
print(cdist([[0, 1]], [[1, 2]], 'seuclidean', V =[0.25, 0.5]))
print("----------------------------")

iris_names  = datasets.load_iris().target_names
iris_target = datasets.load_iris().target
iris_values = datasets.load_iris().data

#vamos a tratar de coger una muestra de cada tipo de flor.
iris_data_setosa = iris_values[0:4]
iris_data_setosa2 = iris_values[5:9]
iris_data_versi = iris_values[51:55]
iris_data_virgi = iris_values[101:105]

print("Distancias Iris:")
#Lo que devuelve es una matriz con todas las distancias.
print("Distancias entre setosa y versicolor:\n", cdist(iris_data_setosa, iris_data_versi, 'euclidean'))
print("Distancias entre muestras de setosas:\n", cdist(iris_data_setosa, iris_data_setosa2, 'euclidean'))
print("Distancias entre setosa y virginica:\n", cdist(iris_data_setosa, iris_data_virgi, 'euclidean'))
print("Distancias entre versicolor y virginica:\n", cdist(iris_data_versi, iris_data_virgi, 'euclidean'))

#En este caso las unidades de las cuatro características son las mismas, por lo que quizás no sea necesario aplicar la normalizada.
#En todo caso, vamos a probar.
#Segun la documentación, por defecto se aplica la siguiente varianza
varianzas = np.var(np.vstack([iris_data_setosa, iris_data_virgi]), axis=0, ddof=1)

print("\nDistancias Iris Normalizadas:")
print("Distancias entre setosa y versicolor:\n", cdist(iris_data_setosa, iris_data_versi, 'seuclidean'))
print("Distancias entre muestras de setosas:\n", cdist(iris_data_setosa, iris_data_setosa2, 'seuclidean'))
print("Distancias entre setosa y virginica:\n", cdist(iris_data_setosa, iris_data_virgi, 'seuclidean'))
print("Distancias entre versicolor y virginica:\n", cdist(iris_data_versi, iris_data_virgi, 'seuclidean'))
#De hecho parece que el resultado no es muy bueno
