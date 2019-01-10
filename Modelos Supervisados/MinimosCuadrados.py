#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:01:02 2019

@author: Ruman
"""

import numpy as np


def stepGradient(par, x, y, learningRate):
    b_0_gradient = 0
    b_1_gradient = 0
    N = float(len(x))
    
    for i in range(0, len(x)):
        b_0_gradient += (2/N) *        (y[i] - (par[0] + par[1] * x[i]))
        b_1_gradient += (2/N) * x[i] * (y[i] - (par[0] + par[1] * x[i]))
        
    new_b_0 = par[0] + (learningRate * b_0_gradient)
    new_b_1 = par[1] + (learningRate * b_1_gradient)
    
    return [new_b_0, new_b_1]

def fitGradient(par, x, y, learningRate, maxDifference = 1e-6, maxIter = 30):
    prev_step = par[:]
    num_iter = 0;
    
    num_iter += 1
    results = stepGradient(prev_step, trX, trY, learningRate)   
    difference = abs(prev_step[0] - results[0]) + abs(prev_step[1] - results[1])

    while ((difference > maxDifference) & (num_iter < maxIter)):
        num_iter += 1
        prev_step = results
        results = stepGradient(prev_step, trX, trY, learningRate)    
        difference = abs(prev_step[0] - results[0]) + abs(prev_step[1] - results[1])

    return results

trX = np.linspace(-2, 2, 101)
trY = 3 + 2 * trX + np.random.randn(*trX.shape) * 0.33

print(fitGradient([1,1], trX, trY, 0.05))

