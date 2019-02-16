#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:00:57 2019

@author: Ruman
"""

"""
La regresión LASSO, al penalizar los parámetros que requieren mayor peso, 
permite seleccionar las características eliminado aquellas que necesitan valores 
de los parámetros más grandes para poder aportar información al modelo.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

wine = pd.read_csv('data/winequality-white.csv', sep = ';')
target = 'quality'
features = list(wine.columns)
features.remove('quality')

caracteristicas = wine[features]
calidad = wine[target]


#Dividimos el conjunto de datos en dos para luego aprovecharlo para Test.
X_train, X_test, Y_train, Y_test = train_test_split(caracteristicas, 
                                                    calidad, 
                                                    test_size=0.33, 
                                                    random_state=4)

"""
Un parñametro importante dentro de Lasso:
normalize: es un valor lógico que indica si las características se normalizan antes de la regresión. 
En caso de que se seleccione esta opción, los resultados suelen ser más robustos. Por defecto, este valor es falso.
"""

clf = Lasso(alpha=0.1, normalize=False)
clf.fit(X_train,Y_train)
print("Vemos que características ha utilizado para el modelo:", clf.coef_)
print("El intercept es:", clf.intercept_) 

"""
La correcta determinación del valor del parámetro de regularización es clave 
para la obtención de modelos con una buena capacidad predictiva y que, a su vez, 
no presenten sobreajuste.

En la siguiente sección se estudiarán algunas técnicas disponibles para determinar 
este valor.
"""