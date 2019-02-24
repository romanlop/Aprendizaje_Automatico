# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:06:26 2019

@author: rlopseo
"""
#Se pretende construir un preceptron con el que se obtenga un resultado similar
#al presentado en la figura 5.1. AND Lógico. 

import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse

#Definicion de las variables de entrada de la red. Variables simbólicas.
x = T.vector('x')

#Variables shared que tendrán valores fijos
w = theano.shared(np.array([1,1], dtype = theano.config.floatX))
b = theano.shared(float(-1.5))

#definición de la neurona
z = T.dot(x, w) + b
formula_and = ifelse(T.lt(z,0),0, 1)

#vector de entrada
inputs = [[0,0],[0,1],[1,0],[1,1]]

#función and
f_and = theano.function(inputs=[x], outputs=formula_and)

#iteramos el vector de entrada
for i in range(len(inputs)):
    t = inputs[i]
    out = f_and(t)
    print('AND: El resultado para [%d,%d] es %d' % (t[0],t[1],out))
    
##############################################################################
#Vamos a hacer los mismo pero para la función or.  
b2 = theano.shared(float(-0.5))   
#definición de la neurona
z2 = T.dot(x, w) + b2
formula_or = ifelse(T.lt(z2,0),0, 1)

#función or
f_or = theano.function(inputs=[x], outputs=formula_or)
 
for i in range(len(inputs)):
    t = inputs[i]
    out = f_or(t)
    print('OR: El resultado para [%d,%d] es %d' % (t[0],t[1],out))