#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:17:52 2019

@author: Ruman
"""

#GridSearchCV, permite crear modelos para la selección de parámetros.

#La validación cruzada se puede utilizar para la selección de los parámetros 
#con la que se ejecutan los algoritmos de aprendizaje, por ejemplo, el valor del alpha en una regresión LASSO


import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

wine = pd.read_csv('data/winequality-white.csv', sep = ';')
target = 'quality'
features = list(wine.columns)
features.remove('quality')

caracteristicas = wine[features]
calidad = wine[target]

# Listado de alphas para ser evaluados
alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001])

model = Lasso()

# Selección del modelo
grid = GridSearchCV(estimator = model,
                    param_grid = dict(alpha = alphas),
                    cv = 10)

grid.fit(caracteristicas, calidad)

# Los mejores parametros para el modelo
print('El mejor parametro es', grid.best_params_)
print('El mejor score es', grid.best_score_)


#probamos con 0.001
clf = Lasso(alpha=0.001, normalize=False)
clf.fit(caracteristicas,calidad)
print("Vemos que características ha utilizado para el modelo:", clf.coef_)
print("El intercept es:", clf.intercept_) 



#Una de las ventajas de GridSearchCV es que ofrece la posibilidad de probar múltiples 
#parámetros de los modelos a la vez. Por ejemplo, se puede probar si es mejor un modelo con término independiente o sin él.

# Listado de valores para intercept
intercepts = np.array([True, False])
diccionario= dict(alpha = intercepts)
#diccionario con los 2 parámetros que vamos a probar.
parameters = {'fit_intercept':('True', 'False'), 'alpha':[1, 0.1, 0.01, 0.001, 0.0001]}

#Creación del modelo
model2 = Lasso()

# Selección del modelo
grid2 = GridSearchCV(estimator = model2,
                    param_grid = parameters,
                    cv = 10)

grid2.fit(caracteristicas, calidad)

# Los mejores parametros para el modelo
print('El mejor parametro es', grid2.best_params_)
print('El mejor score es', grid2.best_score_)


"""
Hasta ahora ha sido necesario indicar al algoritmo los valores a probar de los parámetros, 
lo que no se conoce en muchas ocasiones. Una alternativa es utilizar el constructor RandomizedSearchCV 
junto a una distribución que genere números aleatorios, como puede ser sp_rand, para seleccionar 
aleatoriamente el valor de los parámetros
"""
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV

param_grid = dict(alpha = sp_rand())
model2 = Lasso()
rsearch = RandomizedSearchCV(estimator = model2,
                             param_distributions = param_grid,
                             n_iter = 100,
                             cv = 10,
                             random_state = 1)

rsearch.fit(caracteristicas, calidad)

# Los mejores parametros para el modelo
print('El mejor parametro es', rsearch.best_params_)
print('El mejor score es', rsearch.best_score_)

#CREAMOS EL MODELO
use_features = features[:]
use_features.remove('SPHEQ_grp')

data_model = pd.concat([myopia[use_features], pd.get_dummies(myopia['SPHEQ_grp'], prefix = 'pclass')], axis = 1)
calculateVIF(data_model)