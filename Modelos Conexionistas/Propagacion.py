# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:46:43 2019

@author: rlopseo
"""

import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
import matplotlib.pyplot as plt

#Propagación hacia atrás
#Definición de variables simbólicas
x = T.matrix('x') #usamos matriz en lugar de vector como en perceptron para mostrar todos los resultados de forma conjunta.
w = theano.shared(np.array([1,1], dtype = theano.config.floatX))
b = theano.shared(float(1.0)) #este parámetro en este caso es desconocido pero marcamos uno inicial.
learning_rate = 0.01

#definición de la neurona. Vamos a usar una versión logarítmica para la función de activación. 
#entropía cruzada
z = T.dot(x,w) + b
a = 1 / (1 +np.exp(-z))


#función de coste que trataremos de minimizar
a_hat = T.vector('a_hat') #esto representa los resultados conocidos para la salida.
cost = -(a_hat * T.log(a) + (1 - a_hat) * T.log(1 - a)).sum()

#gradiente de la función de costes
dw, db = T.grad(cost, [w,b]) 

train = theano.function(inputs=[x, a_hat],
                        outputs=[a,cost],
                        updates=[[w,w-learning_rate * dw],[b, b - learning_rate * db]])#esta línea es la que permite la propagaión hacia atrás.


#Entrenamiento de la red
# Conjunto de datos de entrenamiento
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0,0,0,1]

# Iteramos sobre el conjunto de entrenamiento 10000 veces 
cost = []
for iteration in range(10000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter) 
    
# Se imprimen los resultados por pantalla
print('Los resultados de la red son:')
for i in range(len(inputs)):
    print('El resultado para [%d, %d] es %.2f' % (inputs[i][0], inputs[i][1], pred[i]))
    
# Resultados
print
print('El vector w es [%.2f, %.2f]' % (w.get_value()[0], w.get_value()[0]))
print('El valor del bias es %.2f' % b.get_value())

# Función de esfuerzo en función del número de iteraciones
plt.plot(cost)
