#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 10:47:23 2019

@author: Ruman
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import GridSearchCV
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt



"""
FUNCIONES
"""

def calculateIV(data, features, target):
    result = pd.DataFrame(index = ['IV'], columns = features)
    result = result.fillna(0)
    var_target = np.array(data[target])
    
    for cat in features:
        var_values = np.array(data[cat])
        var_levels = np.unique(var_values)

        mat_values = np.zeros(shape=(len(var_levels),2))
        
        for i in range(len(var_target)):
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
        result[cat] = IV
        
    return result

def calculateVIF(data):
    features = list(data.columns)
    num_features = len(features)
    
    #vamos a buscar el R^2 para poder aplicar la formula de VIF
    model = LinearRegression()
    
    #Crea un dataframe con las columnas de los datos originales que pasamos como parámetro. Crea una fila con NAN
    result = pd.DataFrame(index = ['VIF'], columns = features)
    #Sustituimos los NAN por ceros.
    result = result.fillna(0)
    
    #Iteramos las características
    for ite in range(num_features):
        #esta variable almacena un listado con las caracteristicas
        x_features = features[:]
        #en esta almacenamos la caracteristica sobre la que estamos iterando. Una en cada iteración del bucle.
        y_feature = features[ite]
        #eliminamos dicha variable del conjunto de variables.
        x_features.remove(y_feature)
        
        x = data[x_features]
        #en y guardamos la carateristica que hemos seleccionado en este paso del bucle, la trataremos como target en la regresión.
        y = data[y_feature]
        
        model.fit(data[x_features], data[y_feature])
        
        #Aplicamos la fórmula de VIF
        result[y_feature] = 1/(1 - model.score(data[x_features], data[y_feature]))
    
    return result


def selectDataUsingVIF(data, max_VIF = 5):
    result = data.copy(deep = True)
    
    VIF = calculateVIF(result)
    
    while VIF.as_matrix().max() > max_VIF:
        col_max = np.where(VIF == VIF.as_matrix().max())[1][0]
        features = list(result.columns)
        features.remove(features[col_max])
        result = result[features]
        
        VIF = calculateVIF(result)
        
    return result


def get_WoE(data, var, target):
    crosstab = pd.crosstab(data[target], data[var])
    
    print("Obteniendo el Woe para la variable", var, ":")
    
    for col in crosstab.columns:
        if crosstab[col][1] == 0:
            print("  El WoE para", col, "[", sum(crosstab[col]), "] es infinito")
        else:
            print("  El WoE para", col, "[", sum(crosstab[col]), "] es", np.log(float(crosstab[col][0]) / float(crosstab[col][1])))
    
"""
CArGA DATOS
"""
myopia = pd.read_csv('data/myopia.csv', sep = ';')

# Separación de la variable objetivo y las explicativas
target = 'MYOPIC'
features = list(myopia.columns)
features.remove('MYOPIC')

# Listado de variables disponibles para hacer un modelo. Pintamos el número de valores diferentes para cada una de ellas
for var in features:
    print(var , ':' , len(set(myopia[var])))
    

#Pintamos una matriz de scatter plots
#scatter_matrix(myopia, figsize = (12, 12), diagonal = 'kde');


#ELIMINACIÓN DE VARIABLES
features.remove('ID')
features.remove('STUDYYEAR')

"""
Análisis de variables discretas.
En las variables discretas se ha de realizar un análisis para seleccionar las que tienen mayor capacidad predictiva.
"""
print(myopia[features].head())
#Las discretas que nos quedan son AGE, GENDER, MOMMY, DADMY

#Para estas variables se puede obtener la tabla de frecuencias de cada una de ella para ver la posibilidad de que sean incluidas en el modelo y si pueden presentar problemas.
categorical = ['AGE', 'GENDER', 'MOMMY', 'DADMY']
continuous = ['SPHEQ', 'AL', 'ACD', 'LT', 'VCD', 'SPORTHR' ,'READHR', 'COMPHR', 'TVHR', 'DIOPTERHR']

for var in categorical:
    print("Tabla de frecuencias para:", var)
    print(pd.crosstab(myopia[target], myopia[var]))
    print

#Solamente es necesario analizar la variable AGE ya que el resto solamente tienen dos niveles. Tenemos que ver si es necesario discretizar.
get_WoE(myopia, 'AGE', target)   
   
myopia.loc[:, 'AGE_grp'] = None

for row in myopia.index:
    if myopia.loc[row, 'AGE'] <= 7: #7 obtiene el mayor valor para WoE
        myopia.loc[row, 'AGE_grp'] = True
    else:
        myopia.loc[row, 'AGE_grp'] = False

get_WoE(myopia, 'AGE_grp', target) 

features.remove('AGE')
features.append('AGE_grp')

categorical.remove('AGE')
categorical.append('AGE_grp')

"""
Evaluación del IV
"""
print(calculateIV(myopia, categorical, target))
#En este caso todas las variables muestran una relación fuerte con la variable objetivo (IV > 0.5). Ya que son binarias no es necesario crear variables dummies.


#Para las continuas.
print(calculateIV(myopia, continuous, target))
#Para las continuas la relación también es muy fuerte.
for var in continuous:
    f, axarr = plt.subplots(2, sharex = True)
    
    axarr[0].hist(myopia[var][myopia[target]==0])
    axarr[1].hist(myopia[var][myopia[target]==1])
    
    axarr[0].set_ylabel('False')
    axarr[1].set_ylabel('True')
    axarr[0].set_title(var)
    
plt.show()
    
#Se puede ver que puede ser interesante realizar una categorización de las variables: SPHEQ
    
myopia.loc[:, 'SPHEQ_grp'] = myopia['SPHEQ'].map(lambda x: 'n0' if x < 0.05 else 'n1' if x < 0.6 else 'n2')
get_WoE(myopia, 'SPHEQ_grp', target)
print(calculateIV(myopia, ['SPHEQ_grp'], target))

#Eliminamos las características sobrantes
features.remove('SPHEQ')
features.append('SPHEQ_grp')

continuous.remove('SPHEQ')
categorical.append('SPHEQ_grp')

###############################################################################
#CREAMOS EL MODELO

use_features = features[:]
use_features.remove('SPHEQ_grp')

data_model = pd.concat([myopia[use_features], pd.get_dummies(myopia['SPHEQ_grp'], prefix = 'pclass')], axis = 1)

model_vars= selectDataUsingVIF(data_model)
print(calculateVIF(model_vars))


#Separamos los datos
x_train, x_test, y_train, y_test = train_test_split(model_vars, myopia[target])

#Creamos y validamos el modelos
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve

def metricas_modelos(y_true, y_pred):
    from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve

    # Obtención de matriz de confusión
    confusion_matrix = confusion_matrix(y_true, y_pred)

    print("La matriz de confusión es ")
    print(confusion_matrix)

    print('Precisión:', accuracy_score(y_true, y_pred))
    print('Exactitud:', precision_score(y_true, y_pred))
    print('Exhaustividad:', recall_score(y_true, y_pred))
    print('F1:', f1_score(y_true, y_pred))

    false_positive_rate, recall, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(false_positive_rate, recall)

    print('AUC:', auc(false_positive_rate, recall))

    plt.plot(false_positive_rate, recall, 'b')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('AUC = %0.2f' % roc_auc)
    
from sklearn.linear_model.logistic import LogisticRegression

model = LogisticRegression().fit(x_train, y_train)
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

metricas_modelos(y_train, y_pred_train)
metricas_modelos(y_test, y_pred_test)