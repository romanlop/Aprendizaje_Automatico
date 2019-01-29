#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:48:56 2019

@author: Ruman
"""
#Explicación intuitiva del Análisis de Componentes Principales: https://www.youtube.com/watch?v=LI_iQAUa228 

import numpy as np

x = np.array([[0.9, 1],
              [2.4, 2.6],
              [1.2, 1.7],
              [0.5, 0.7],
              [0.3, 0.7],
              [1.8, 1.4],
              [0.5, 0.6],
              [0.3, 0.6],
              [2.5, 2.6],
              [1.3, 1.1]])

y = np.array([x.T[0] - np.mean(x.T[0]),
              x.T[1] - np.mean(x.T[1])])
c = np.cov(y)

l, v = np.linalg.eig(c)

print("Los vectores propios son: ", v[0], "y", v[1])
print("Los valores propios son: ", l)

print("Primer componente: ", np.dot(y.T, v.T[0]))
print("Segundo componente: ", -np.dot(y.T, v.T[1]))

