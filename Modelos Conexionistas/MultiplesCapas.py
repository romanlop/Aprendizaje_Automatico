# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:36:45 2019

@author: rlopseo
"""
import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
import matplotlib.pyplot as plt

#Creación simplificada de redes neuronales.

def floatX(X):
    return np.asarray(X, dtype = theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

x = T.matrix('x')
a_hat = T.vector('a_hat')
learning_rate = 0.01

# Semilla
rng = np.random.RandomState(1)

# Bias
b1 = theano.shared(1.)
b2 = theano.shared(1.)

# Pesos iniciales aleatorios
w1 = init_weights((2, 3))
w2 = init_weights((3, 1))

"""
Importante, dos matrices son multiplicables si el número de comlumas de A es = al número
de filas de B. En este caso X (que es el imput) tiene 1 fila y dos columnas (una lista de
4 elementos con esta forma) es decir 4 filas 2 columnas (4x2). El número de filas de w1 es 2, 
por tanto son multiplicables. 

En caso contrario, a1 fallaría (ver código mas abajo). El resultado de multiplicar estas
dos matrices es una matriz de 4 x 3. Es por esto que W2 tiene 3 columnas y 1 fila.

"""

# Definición de la red. Una neurona unida a otra + salida.
a1 = T.nnet.sigmoid(T.dot(x, w1) + b1)
a2 = T.nnet.sigmoid(T.dot(a1, w2) + b2)
a3 = T.flatten(a2) #tenemos que hacer esto para que tenga la misma forma que el vector de entrenamiento.
#en caso contrario no funcionaría la retroalimentación. vector outputs.

# Función de esfuerzo
cost = T.nnet.binary_crossentropy(a3, a_hat).mean()

# Función de entrenamiento
train = theano.function(inputs = [x, a_hat],
                        outputs = [a3, cost],
                        updates = [
                            (w1, w1 - learning_rate * T.grad(cost, w1)),
                            (w2, w2 - learning_rate * T.grad(cost, w2)),
                            (b1, b1 - learning_rate * T.grad(cost, b1)),
                            (b2, b2 - learning_rate * T.grad(cost, b2))
                        ])
    
    
learning_rate = 0.1

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [1, 0, 0, 1]

# Iteramos sobre el conjunto de entrenamiento 
cost = []
for iteration in range(3000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)
    print(pred)
    
# Se imprimen los resultados por pantalla
print('Los resultados de la red son:')
for i in range(len(inputs)):
    print('El resultado para [%d, %d] es %.2f' % (inputs[i][0], inputs[i][1], pred[i]))
    

# Función de esfuerzo en función del número de iteraciones
plt.plot(cost)