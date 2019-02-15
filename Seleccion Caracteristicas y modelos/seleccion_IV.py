#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:40:38 2019

@author: Ruman
"""

import pandas as pd
import numpy as np

#Lectura del fichero
credit_approval = pd.read_csv('data/crx_data.txt', sep = ',')
print(credit_approval.head())

#filtro para eliminar registros con campos vacíos.
mask = credit_approval.applymap(lambda x: x in ['?'])
credit_not_null = credit_approval[-mask.any(axis=1)]

#Los detalles de este conjunto de datos se pueden encontrar en el repositorio 
#de datos para Aprendizaje Automático existente en la página web de la Universidad 
#de California (https://archive.ics.uci.edu/ml/datasets/Credit+Approval)

# Separación de las variables
var_categoricas = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A11', 'A12', 'A13']
var_numericas   = ['A2', 'A3', 'A8', 'A14', 'A15']
var_target      = credit_not_null['A16'] == '+'


result_IV = []

for v_cat in var_categoricas:
    var_target = np.array(var_target)
    var_values = np.array(credit_not_null[v_cat])
    var_levels = np.unique(var_values)

    mat_values = np.zeros(shape=(len(var_levels),2))

    for i in range(len(var_target)):
        # Obtención de la posición en los niveles del valor
        for j in range(len(var_levels)):
            if var_levels[j] == var_values[i]:
                pos = j
                break

        # Estimación del número valores en cada nivel
        if var_target[i]:
            mat_values[pos][0] += 1
        else:
            mat_values[pos][1] += 1

        # Obtención del IV
        IV = 0
        for j in range(len(var_levels)):
            if mat_values[j][0] > 0 and mat_values[j][1] > 0:
                rt = mat_values[j][0] / (mat_values[j][0] + mat_values[j][1])
                rf = mat_values[j][1] / (mat_values[j][0] + mat_values[j][1])
                IV += (rt - rf) * np.log(rt / rf)
        
    # Se agrega el IV al listado
    result_IV.append(IV)

for i in range(len(var_categoricas)):
    print("La característica", var_categoricas[i], "el IV es", result_IV[i])