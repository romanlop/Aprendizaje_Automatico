# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:54:33 2019

@author: rlopseo
"""

#Red con dos capas para problema XNOR.


import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
import matplotlib.pyplot as plt


x = T.matrix('x') #usamos matriz en lugar de vector como en perceptron para mostrar todos los resultados de forma conjunta.
w1 = theano.shared(np.array([.1,.2], dtype = theano.config.floatX)) #pesos para la primera neurona
w2 = theano.shared(np.array([.3,.4], dtype = theano.config.floatX))
w3 = theano.shared(np.array([.5,.6], dtype = theano.config.floatX))
b1 = theano.shared(1.) #vamos a usar el mismo bias para las dos neuronas de la capa de entrada
b2 = theano.shared(1.)
learning_rate = 0.01

#Importante, los valores inciales podrían ser aleatorios. Nosotros los estamos fijando para reproducir resultados.

a1 = 1 / (1 + T.exp(-T.dot(x, w1) - b1))
a2 = 1 / (1 + T.exp(-T.dot(x, w2) - b1))
x2 = T.stack([a1, a2], axis = 1) #agrupa las salidas de las dos primeras neuronas como entrada de la siguiente.
a3 = 1 / (1 + T.exp(-T.dot(x2, w3) - b2))

#Función de coste, vemos que el coste se calcula en la neurona de salida.
a_hat = T.vector('a_hat')
cost = -(a_hat * T.log(a3) + (1 - a_hat) * T.log(1 - a3)).sum()
dw1, dw2, dw3, db1, db2 = T.grad(cost, [w1, w2, w3, b1, b2])

train = theano.function(
    inputs = [x,a_hat],
    outputs = [a3,cost],
    updates = [
        [w1, w1 - learning_rate * dw1],
        [w2, w2 - learning_rate * dw2],
        [w3, w3 - learning_rate * dw3],
        [b1, b1 - learning_rate * db1],
        [b2, b2 - learning_rate * db2]
    ]
)

#Entrenamiento de la red
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [1, 0, 0, 1]

# Iteramos sobre el conjunto de entrenamiento 
cost = []
for iteration in range(50000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)
    
# Se imprimen los resultados por pantalla
print('Los resultados de la red son:')
for i in range(len(inputs)):
    print('El resultado para [%d, %d] es %.2f' % (inputs[i][0], inputs[i][1], pred[i]))

# Resultados
print
print('El vector w1 es [%.2f, %.2f]' % (w1.get_value()[0], w1.get_value()[0]))
print('El vector w2 es [%.2f, %.2f]' % (w2.get_value()[0], w2.get_value()[0]))
print('El vector w3 es [%.2f, %.2f]' % (w3.get_value()[0], w3.get_value()[0]))
print('El valor del bias 1 es %.2f' % b1.get_value())
print('El valor del bias 2 es %.2f' % b2.get_value())

# Función de esfuerzo en función del número de iteraciones
plt.plot(cost)
