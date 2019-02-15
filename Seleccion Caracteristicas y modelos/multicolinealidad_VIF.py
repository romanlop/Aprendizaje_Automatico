#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:52:17 2019

@author: Ruman
"""

#Una manera de evitar la multicolinealidad es utilizar el Factor de inflación de la varianza VIF
#Permite cuantificar la intensidad de la multicolinealidad.

#https://etav.github.io/python/vif_factor_python.html

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

wine = pd.read_csv('data/winequality-white.csv', sep = ';')
target = 'quality'
features = list(wine.columns)
features.remove('quality')

caracteristicas = wine[features]
calidad = wine[target]

#Lo primiero que podemos hacer es calcular la matriz de correlacción
matriz_correlaccion = np.corrcoef(caracteristicas,rowvar = False)
#Simplemente revisando la matriz vemos un 0.8 entre la variable 3 y la 7.
#Vemos también un valor elevado entre la 10 y la 7.


def calculateVIF(data):
    features = list(data.columns)
    num_features = len(features)
    
    #vamos a buscar el R^2 para poder aplicar la formula de VIF
    model = LinearRegression()
    
    #Crea un dataframe con las columnas de los datos originales que pasamos como parámetro. Crea una fila con NAN
    result = pd.DataFrame(index = ['VIF'], columns = features)
    #Sustituimos los NAN por ceros.
    result = result.fillna(0)
    
    #Iteramos las características
    for ite in range(num_features):
        #esta variable almacena un listado con las caracteristicas
        x_features = features[:]
        #en esta almacenamos la caracteristica sobre la que estamos iterando. Una en cada iteración del bucle.
        y_feature = features[ite]
        #eliminamos dicha variable del conjunto de variables.
        x_features.remove(y_feature)
        
        x = data[x_features]
        #en y guardamos la carateristica que hemos seleccionado en este paso del bucle, la trataremos como target en la regresión.
        y = data[y_feature]
        
        model.fit(data[x_features], data[y_feature])
        
        #Aplicamos la fórmula de VIF
        result[y_feature] = 1/(1 - model.score(data[x_features], data[y_feature]))
    
    return result


def selectDataUsingVIF(data, max_VIF = 5):
    result = data.copy(deep = True)
    
    VIF = calculateVIF(result)
    
    while VIF.as_matrix().max() > max_VIF:
        col_max = np.where(VIF == VIF.as_matrix().max())[1][0]
        features = list(result.columns)
        features.remove(features[col_max])
        result = result[features]
        
        VIF = calculateVIF(result)
        
    return result
    
  
"""
When deep=True (default), a new object will be created with a copy of the 
calling object’s data and indices. Modifications to the data or indices of 
the copy will not be reflected in the original object (see notes below).
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html
"""
    
caracteristicas_finales = selectDataUsingVIF(caracteristicas.copy(deep = True))   
    