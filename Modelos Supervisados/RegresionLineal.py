#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:39:36 2019

@author: Ruman
"""
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.metrics as mtr

# Conjunto de datos
x = [[80], [79], [83], [84], [78], [60], [82], [85], [79], [84], [80], [62]]
y = [[300], [302], [315], [330], [300], [250], [300], [340], [315], [330], [310], [240]]


plt.title('Datos de entrenamiento')
plt.scatter(x,y)
plt.xlabel('Horas')
plt.ylabel('Producción')
plt.show()

#Entrenamiento del modelo
model = LinearRegression().fit(x, y)

#Obtención de los parámetros de ajuste
print('Coeficiente calculado:',model.coef_)
print('Intercept calculado:',model.intercept_)

#Podemos calcular el coeficiente R^2 desde el modelo, o bien como veremos a continuación.
print('R^2', model.score(x, y))

#Para calcular el error cuadrático medio, el absoluto y la mediana del error absoluto, tenemos que entrenarlo contra el dataset inicial
y_pred=model.predict(x)
print('Otra forma de calcular el R^2', mtr.r2_score(y_pred, y))
print('Error cuadrático medio', mtr.mean_squared_error(y_pred, y))
print('Error absoluto medio', mtr.mean_absolute_error(y_pred, y))
print('Mediana del error absoluto', mtr.median_absolute_error(y_pred, y))

#vamos a predecir un valor.
print('Predicción para 75 horas:',model.predict(75))

#Vamos a representar el modelo.
new_data = [[50], [120]]
training=model.predict(new_data)

plt.title('Modelo entrenado')
plt.plot(new_data,training,'r',label='Modelo')
plt.plot(x,y,'b.',label='Datos')
plt.xlabel('Horas')
plt.ylabel('Producción')
plt.axis([50,90,200,350])
plt.legend(loc = 2)
plt.show()

print('\n##################################################################')
print('##################################################################\n')
"""
Hay un problema con este modelo. Vemos que el intercept es mayor que cero.
¿El modelo está diciendo que cuando no se trabaja se producen 31 unidades? -> Intercept
Sí, el modelo así construido indica que existirá una producción fija 
sin trabajar. 
Para solucionar este problema se ha de crear un modelo en el que el término 
independiente sea 0. Eso se puede hacer añadiendo la opción 
fit_intercept = False cuando se construye el objeto
"""

model_improved = LinearRegression(fit_intercept=False).fit(x, y)

#Pintamos las métricas.
print('R^2', model_improved.score(x, y))

#vamos a predecir un valor.
print("Nuevo modelo:")
print('Predicción para 75 horas:',model_improved.predict(75))
print('Predicción para 50 horas:',model_improved.predict(50))
print('Predicción para 20 horas:',model_improved.predict(20))
print('Predicción para 5 horas:',model_improved.predict(5))
print('Predicción para 0 horas:',model_improved.predict(0))

print("Modelo Anterior:")
print('Predicción para 75 horas:',model.predict(75))
print('Predicción para 50 horas:',model.predict(50))
print('Predicción para 20 horas:',model.predict(20))
print('Predicción para 5 horas:',model.predict(5))
print('Predicción para 0 horas:',model.predict(0))

#Una forma de comparar la validez de los dos modelos es mediante la representación grafica 
#de las predicciones de ambos y los datos en una misma figura.

new_data2 = [[0], [120]]
training2=model_improved.predict(new_data2)
plt.title('Comparativa Modelos')
plt.plot(new_data,training,'r',label='Modelo')
plt.plot(new_data2,training2,'g',label='Modelo Sin Itrcp')
plt.plot(x,y,'b.',label='Datos')
plt.xlabel('Horas')
plt.ylabel('Producción')
plt.axis([50,90,200,350])
plt.legend(loc = 2)
plt.show()