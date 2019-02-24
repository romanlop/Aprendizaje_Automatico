# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import theano
import theano.tensor as T
import numpy as np
import time
from theano.ifelse import ifelse

#Theano es un lenguaje simbólico por lo que se trata de representar expresiones, 
#mas que dar valor a las variables.s
x = T.scalar('x')
y = x ** 3 #Elevado a.

print(y.eval({x : 2}))

#Podemos realizar operaciones más complejas.
x = T.scalar('x')
y = T.scalar('y')
z = 2 * x + 3 * y
print(z.eval({x:1, y:10}))

#normalmente no se utiliza eval, sino que se crean funciones como veremos a continuación.
f = theano.function(inputs = [x,y], outputs = z)
print(f(1,10))

#para definir las funciones a realizar sobre las variables, se puede utilizar código python
def sum_vars(x,y):
    return x + y

z = sum_vars(x,y)
f = theano.function(inputs = [x,y], outputs = z)
print(f(1,10))

#se puende utilziar también las funciones ya incluidas en python
x = T.scalar('x')
y = np.cos(x)
f = theano.function(inputs = [x], outputs = y)
print("Coseno:",f(0))

print("\n##################CONTROL DE FLUJO##################")
#En las funciones creadas no pueden utilizarse operadores lógicos con las variables.
#Por tanto no se pueden crear controles de flujo python.
#Se han de utilizar los que proporciona Theano.
x = T.scalar('x')
y = T.switch(T.gt(x,0),x, -x)
f = theano.function(inputs = [x], outputs = y)
print(f(-1))
print(f(20))

#Se puede utilizar también ifelse. La diferencia de ifelse es que solo se ejecuta si es true.
#Por tanto este es mas eficiente.
a, b = T.scalars('a', 'b')
x, y = T.matrices('x', 'y')

z_switch = T.switch(T.lt(a,b), T.mean(x),T.mean(y))
z_ifelse = ifelse(T.lt(a,b), T.mean(x),T.mean(y))

f_switch = theano.function(inputs = [a,b,x,y], outputs = z_switch)
f_ifelse = theano.function(inputs = [a,b,x,y], outputs = z_ifelse)


val1 = 0.
val2 = 1.
big_matrix = np.ones((15000,15000))

tic = time.clock()
f_switch(val1,val2,big_matrix,big_matrix)
print('El tiempo utilizando switch es %f' % (time.clock() - tic))

tic = time.clock()
f_ifelse(val1,val2,big_matrix,big_matrix)
print('El tiempo utilizando ifelse es %f' % (time.clock() - tic))

print("\n##################VALORES POR DEFECTO##################")
#Se utiliza theano.in
x, y = T.scalars('x','y')
z = x * y
f = theano.function(inputs=[x, theano.In(y, value = 3)], outputs = z)
print("El resultado es:", f(3))
print("El resultado es:", f(3,5))

print("\n##################VARIABLES COMPARTIDAS##################")
"""
A ninguna de las variables definidas en Theano se le ha asignado un valor explícito, 
solamente se han utilizado para indicar las operaciones a realizar. 
Los únicos valores que se han usado son los de las entradas de las funciones. 
Poder indicar valores explícitos para algunas variables es necesario en muchas ocasiones,
 de hecho, los parámetros de un modelo se han de indicar de esta manera. 
 En Theano, para que una variable pueda tener un valor explícito esta ha de ser de 
 tipo shared (compartido).      
"""
#es mejor indicar el tipo de dato, no como en python
x = theano.shared(np.array(1,dtype = theano.config.floatX))
A = T.scalar()
f = theano.function(inputs = [A], outputs = x, updates = {x: x - A})
print("Salida de la función:",f(5))
print(x.get_value())
#Vemos que la función devuelve el valor de entrada de X y luego se actualiza su valor.

f = theano.function(inputs = [A], updates = {x: x * A})
#Si la definimos así no devuelve el valor original de X, sino que devuelve vacio.
print("Salida de la función:",f(5))
print(x.get_value())

#otra forma de definir los shared, con otro tipo de dato
x = theano.shared(5)
A = T.iscalar() #tiene que ser del mismo tipo que la definida anteriormente
f = theano.function(inputs = [A], outputs = x, updates = {x: x - A})
print("Salida de la función:",f(10))
print(x.get_value())

#otra forma de definir los shared, con otro tipo de dato
x = theano.shared(float(5))
A = T.fscalar() #tiene que ser del mismo tipo que la definida anteriormente
f = theano.function(inputs = [A], outputs = x, updates = {x: x - A})
print("Salida de la función:",f(10))
print(x.get_value())

#Las variables compartidas pueden ser también matrices o tensores
x = theano.shared(np.array([[1,2],[3,4]],dtype = theano.config.floatX))
A = T.matrix()
f = theano.function(inputs = [A], outputs = x, updates = {x: x - A})
print(f(np.array([[1,1],[1,1]])))
print(x.get_value())

print("\n##################OPERACIONES MATRICIALES##################")
W = T.matrix('W')
v = T.vector('v')
b = T.vector('biases')

x = T.dot(v, W) + b
f = theano.function(inputs = [v, W, b], outputs=x)
print (f([1,1],[[2,4],[3,5]],[2,3]))  

print("\n##################GRADIENTES##################")
x = T.scalar()
y = x**2

#y_grad = dy/dx 
y_grad = T.grad(y,x)     

#La derivada de X^2 es 2X.
print(y_grad.eval({x: 10}))

#La función grad permite derivar cualquier tipo de función. Por ejemplo funciones trignometricas
x = T.scalar()
y = np.cos(x)

y_grad = T.grad(y,x) 
#La derivada de cos(x) es -x*sen(x).
print(y_grad.eval({x: 10}))


x = T.scalar()
y = x**3 + x**2

#y_grad = dy/dx 
y_grad = T.grad(y,x)     

#La derivada de X^2 es 3x^2*2X.= 300 + 20
print(y_grad.eval({x: 10}))


print("\n##################GRADIENTE DESCENDETE##################")
trX = np.linspace(-1, 1, 101) #Devuelve 101 valores entre -1 y 1
trY = 2 * trX + np.random.randn(*trX.shape) * 0.50 + 10 
#Simulamos una función y = 2X + 10 con ruido blanco

X = T.scalar()
Y = T.scalar()

def model(X, w, c):
    return X * w + c

w = theano.shared(np.asarray(0., dtype = theano.config.floatX))
c = theano.shared(np.asarray(0., dtype = theano.config.floatX))
y = model(X, w, c)

cost     = T.mean(T.sqr(y - Y))
gradient_w = T.grad(cost = cost, wrt = w)
gradient_c = T.grad(cost = cost, wrt = c)
updates  = [[w, w - gradient_w * 0.01], [c, c - gradient_c * 0.01]]

train = theano.function(inputs = [X, Y], outputs = cost, updates = updates)

for i in range(15):
    for x, y in zip(trX, trY):
        cost_i = train(x, y)
    print('En el paso', i, 'el valor de w es', w.get_value(),
           'y c es', c.get_value(), 'con un coste', cost_i)