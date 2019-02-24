# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:33:08 2019

@author: rlopseo
"""

import os
import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
import matplotlib.pyplot as plt
import matplotlib.colors as cm

def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h


#En la web se explica el formato de los ficheros. Por ejemplo el que contiene las imágenes de train
#tiene 60000 imágenes de 28x28 píxeles.

def mnist(ntrain=60000,ntest=10000,onehot=True):
    fd = open('C:/Users/rlopseo/Documents/Python_Scripts/data/train-images.idx3-ubyte','r')
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28)).astype(float)#en la web indica que hay que pillar a partir del 16

    fd = open('C:/Users/rlopseo/Documents/Python_Scripts/data/train-labels.idx1-ubyte','r')
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))#en la web indica que hay que pillar a partir del 8

    fd = open('C:/Users/rlopseo/Documents/Python_Scripts/data/t10k-images.idx3-ubyte','r')
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28)).astype(float)#en la web indica que hay que pillar a partir del 16

    fd = open('C:/Users/rlopseo/Documents/Python_Scripts/data/t10k-labels.idx1-ubyte','r')
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))#en la web indica que hay que pillar a partir del 8

    trX = trX/255. #los pixeles están representados entre 0 y 255, al dividir por 255 mantenemos la escala pero entre 0 y 1
    teX = teX/255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

#en este punto Y contiene el valor para cada imágen, 0, 1 ,2 ,3 ...
#Lo que queremos es un 1 en la columna correspondiente. Esto nos lo da la función one_hot
    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX,teX,trY,teY


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def floatX(X):
    return np.asarray(X, dtype = theano.config.floatX)

trX, teX, trY, teY = mnist()

#vamos a pintar uno de los números/imágenes
print(trY[1,:])
plt.imshow(trX[0,:].reshape(28, 28), cmap = 'gray')
plt.show()


#Cada imagen tiene 28x28 píxeles = 784. Por tanto la cada de entrada debe tern 784 neuronas o pesos.
#La cada de salida tiene que ser capaz de etiquetar 10 resultados, por necesitamos 10 neuronas.

#Podemos crear una red sencilla con 10 neuronas cada una de ellas con 784 pesos.
#Vamos a utilziar sigmoid.
num_iter = 25

X = T.fmatrix()
Y = T.fmatrix()

w = init_weights((784, 10)) #784 pesos para cada neurona.

#podemos utilziar diferentes funciones de activación.
#py_x = T.nnet.sigmoid(T.dot(X, w))
py_x = T.nnet.softmax(T.dot(X, w))

y_pred = T.argmax(py_x, axis=1) #devuelve la neurona con el mayor valor de activación.

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost, w)
update = [[w, w - gradient * 0.1]]

train = theano.function(inputs = [X, Y],
                        outputs = cost,
                        updates = update,
                        allow_input_downcast=True) #Esto permite que se realice conversión de tipos de datos para las variables de entrada.

predict = theano.function(inputs = [X],
                          outputs = y_pred,
                          allow_input_downcast = True)

#vamos a entrenar cogiendo un subconjunto, para ello ponemos un step de 128.
for i in range(num_iter):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print(i, np.mean(np.argmax(teY, axis=1) == predict(teX)), cost)
    #np armax devuelve el índice del valor máximo. En este caso con axis = 1 lo que nos devuelve es el valor del número, ya que es donde tenemos un uno.
    #Por ejemplo el primer número es un 7
    #Por tanto en el print, en el segundo parámetro, se está pintando la precisión.

#######################################################################################
#Podemos aumentar el número de capas para aumentar la capacidad predictiva.
#Vamos a crear 625 neuronas en la primera capa y 10 en la segunda. Vamos a crear dos capas de dos tipos diferentes.
"""
w_h = init_weights((784, 625))
w_o = init_weights((625, 10))

h = T.nnet.sigmoid(T.dot(X, w_h))
py_x = T.nnet.softmax(T.dot(h, w_o))
y_x = T.argmax(py_x, axis = 1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
updates = [[w_h, w_h - T.grad(cost, w_h) * 0.1],
          [w_o, w_o - T.grad(cost, w_o) * 0.1] ]

train = theano.function(inputs = [X, Y], 
                        outputs = cost,
                        updates = updates,
                        allow_input_downcast = True)
predict = theano.function(inputs = [X],
                          outputs = y_x,
                          allow_input_downcast = True)

for i in range(num_iter):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print(i, np.mean(np.argmax(teY, axis=1) == predict(teX)), cost)
"""


###DROPOUT#####################################################################
from theano.sandbox.rng_mrg import MRG_RandomStreams

srng = MRG_RandomStreams()

def dropout(X, p):
    if p > 0:
        X *= srng.binomial(X.shape, p = 1 - p, dtype = theano.config.floatX)
        X /= 1 - p
    return X

def model(X, w_h, w_o, p_drop):
    X = dropout(X, p_drop)
    h = T.nnet.sigmoid(T.dot(X, w_h))
    
    h = dropout(h, p_drop)
    py_x = T.nnet.softmax(T.dot(h, w_o))

    return h, py_x


w_h = init_weights((784, 40))
w_o = init_weights((40, 10))

# Modelo de entrenamiento
h, py_x = model(X, w_h, w_o, 0.20)
y_x = T.argmax(py_x, axis = 1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
updates = [[w_h, w_h - T.grad(cost, w_h) * 0.1],
          [w_o, w_o - T.grad(cost, w_o) * 0.1] ]

train = theano.function(inputs = [X, Y], 
                        outputs = cost,
                        updates = updates,
                        allow_input_downcast = True)

# Modelo para evaluación, con p == 0
h_predict, py_predict = model(X, w_h, w_o, 0.0)
y_predict = T.argmax(py_predict, axis = 1)

predict = theano.function(inputs = [X],
                          outputs = y_predict,
                          allow_input_downcast = True)

# Evaluación del modelo
for i in range(num_iter):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print(i, np.mean(np.argmax(teY, axis=1) == predict(teX)), cost)