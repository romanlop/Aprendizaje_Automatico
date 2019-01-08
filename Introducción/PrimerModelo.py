# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
#Importamos los algoritmos de scikit learn que nos interesan -> https://scikit-learn.org/stable/modules/neighbors.html
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#Definimos el conjunto de datos de entrenamiento. Se trata de coordenadas X,Y
data = np.array([[1, 1], [1.5, 0.95], [1.5, 1.75], [1.9, 2], [2.2, 1.8], [2.5, 2.2]])
#Definimos el conjunto de resultados asociados (colores)
target =  np.array(['r', 'r', 'r', 'g', 'g', 'g'])
#Conjunto que datos que utilizaremos para testear/buscar una predicción.
newData = np.array([[0.8, 1.5], [2.3, 2.8], [2, 2]])


#############################################################################
#El problema consiste en encontrar las categorías a las que pertenecen los nuevos puntos, en la gráfica los de color negro.
#vstack -> Stack arrays in sequence vertically (row wise).
X = np.vstack([data,newData])
col = np.concatenate([target, np.array(['k', 'k', 'k'])])
plt.title('Datos Iniciales y Datos a predecir')
plt.scatter(X[:,0], X[:,1], c=col)
plt.show()

#RADIUS

# Creamos el modelo sin entrenar:
model = RadiusNeighborsClassifier(radius = 1)

# Entrenamos el modelo:
model.fit(data, target)

# Predecimos la clase para tres puntos diferentes:
prediction = model.predict(newData)

#Representamos la predicción
new_col = np.concatenate([target, prediction])
plt.title('Predicción Radius')
plt.scatter(X[:,0], X[:,1], c=new_col)
plt.show()

#############################################################################
#En este caso, si se revisa la documentación de la librería se verá que se aproxima mejor el radius. Pero vamos a probar tb el # KNeighborsClassifier
# Creamos el modelo sin entrenar:
model = KNeighborsClassifier(n_neighbors = 5)

# Entrenamos el modelo:
model.fit(data, target)

# Predecimos la clase para tres puntos diferentes:
prediction = model.predict(newData)

#Representamos la predicción
new_col = np.concatenate([target, prediction])
plt.title('Predicción KNeighbors con n<=5')
plt.scatter(X[:,0], X[:,1], c=new_col)
plt.show()

#############################################################################
#Con n=6 para el primer punto tiene 3 rojos y 3 verdes.
model = KNeighborsClassifier(n_neighbors = 6) 

# Entrenamos el modelo:
model.fit(data, target)

# Predecimos la clase para tres puntos diferentes:
prediction = model.predict(newData)

#Representamos la predicción
new_col = np.concatenate([target, prediction])
plt.title('Predicción KNeighbors con n=6')
plt.scatter(X[:,0], X[:,1], c=new_col)
plt.show()

#############################################################################
#Con n=6 para el primer punto tiene 3 rojos y 3 verdes. Pero si les damos un peso en función de la distancia arreglamos.
model = KNeighborsClassifier(n_neighbors = 6, weights = 'distance') 

# Entrenamos el modelo:
model.fit(data, target)

# Predecimos la clase para tres puntos diferentes:
prediction = model.predict(newData)

#Representamos la predicción
new_col = np.concatenate([target, prediction])
plt.title('Predicción KNeighbors con n=6 pero especificando un peso')
plt.scatter(X[:,0], X[:,1], c=new_col)
plt.show()