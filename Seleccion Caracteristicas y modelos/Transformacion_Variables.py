#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:23:08 2019

@author: Ruman
"""

#DISCRETIZACIÓN

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

print(pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]), 3,
             labels = ['bad', 'medium', 'good'],
             right = True,
             include_lowest = False,
             retbins = True))


raw_data = load_breast_cancer()
data = raw_data.data

print("\nEl valor máximo es", max(data[:,0])) #28.11
print("\nEl valor mínimo es", min(data[:,0])) #6.981

print(pd.cut(data[:,0],
             4,
             labels = ['bad', 'medium', 'good','excellent'],
             right = False, #con false no mete el máximo como valor superior del intervalo, si no que lo incluye en el intervalo mayor.
             include_lowest = False,
             retbins = True))

"""
Existen diferentes métodos para seleccionar la división más adecuada de una
característica continua. Por ejemplo, se pueden utilizar estadísticas de 
tendencia central, como la media o la mediana o la optimización de una función 
objetivo, como es el peso de la evidencia.

El peso de la evidencia (WoE, Weight of Evidence) es un valor que indica la 
capacidad predictiva de cada uno de los niveles de una característica categórica 
respecto a una característica binaria.

https://medium.com/@sundarstyles89/weight-of-evidence-and-information-value-using-python-6f05072e83eb
"""


#crosstab -> https://pbpython.com/pandas-crosstab.html
def get_WoE(data, var, target):
    crosstab = pd.crosstab(data[target], data[var])
    
    print("Obteniendo el Woe para la variable", var, ":")
    
    for col in crosstab.columns:
        if crosstab[col][1] == 0:
            print("  El WoE para", col, "[", sum(crosstab[col]), "] es infinito")
        else:
            WoE = np.log(float(crosstab[col][0]) / float(crosstab[col][1]))
            print("  El WoE para", col, "[", sum(crosstab[col]), "] es", WoE)
            

data = pd.DataFrame({'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'Target': [True, True, False, True, True, False, True, False, False, False]})

data['Cat 1'] = data['Value'] > 3
data['Cat 2'] = data['Value'] > 6

get_WoE(data, 'Cat 1', 'Target')
get_WoE(data, 'Cat 2', 'Target')

"""
Uno de los motivos por los que se emplea la discretización en conjuntos de características 
cuantitativos es para eliminar las dimensiones en las que se han medido, es decir, 
para evitar los sesgos que puede introducir utilizar unidades diferentes, como metros o 
pies en la medida. Pero, en muchas ocasiones puede que no sea aceptable la pérdida de información 
que conlleva la discretización de una característica. En estas ocasiones se puede utilizar la 
normalización para eliminar las dimensiones de las características y, al mismo tiempo, conservar 
toda la información disponible en las mismas.
"""

from sklearn.preprocessing import MinMaxScaler  #devuelve valores entre 0 y 1

data2 = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
print(scaler.fit(data2))
print(scaler.transform(data2))


from sklearn.preprocessing import StandardScaler  #basado en media y desviación estándar
scaler = StandardScaler().fit(data2)
print(scaler.transform(data2))


from sklearn.preprocessing import RobustScaler #rango intercuantílico
scaler = RobustScaler().fit(data2)
print(scaler.transform(data2))
