#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:53:39 2019

Una de las principales utilidades de PCA es mejorar el rendimiento de los algotimos de machine learning.
The MNIST database of handwritten digits is more suitable as it has 784 
feature columns (784 dimensions), a training set of 60,000 examples, 
and a test set of 10,000 examples.

https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

Se trata de imágenes. 

@author: Ruman
"""

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

"""
Código para descargar de internet del fichero de matlab:
    
from six.moves import urllib
from sklearn.datasets import fetch_mldata
mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
mnist_path = "./mnist-original.mat"
response = urllib.request.urlopen(mnist_alternative_url)
with open(mnist_path, "wb") as f:
    content = response.read()
    f.write(content)
"""
    
#Loadmat carga ficheros de Matlab.
mnist_raw = loadmat("Data/mnist-original.mat")
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}

"""
The images that you downloaded are contained in mnist.data and has a shape of 
(70000, 784) meaning there are 70,000 images with 784 dimensions (784 features).

The labels (the integers 0–9) are contained in mnist.target. The features are 
784 dimensional (28 x 28 images) and the labels are simply numbers from 0–9.
"""

"""
Typically the train test split is 80% training and 20% test. In this case, 
I chose 6/7th of the data to be training and 1/7th of the data to be in the 
test set.
"""

train_img, test_img, train_lbl, test_lbl = train_test_split( mnist["data"], 
                                                            mnist["target"], 
                                                            test_size=1/7.0, 
                                                            random_state=0)

"""
PCA is effected by scale so you need to scale the features in the data before 
applying PCA. You can transform the data onto unit scale (mean = 0 and variance = 1) 
which is a requirement for the optimal performance of many machine learning algorithms. 
StandardScaler helps standardize the dataset’s features. Note you fit on the training 
set and transform on the training and test set. If you want to see the negative 
effect not scaling your data can have, scikit-learn has a section on the effects 
of not standardizing your data.
"""

scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

"""
Aplicamos PCA
Notice the code below has .95 for the number of components parameter. 
It means that scikit-learn choose the minimum number of principal components 
such that 95% of the variance is retained
"""
pca = PCA(.95)
pca.fit(train_img)

"""
Trasformamos los datos de test y de training
"""

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

"""
Vamos a aplicar regresión logistica a los datos transformados
"""
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
#entremamos el modelo
logisticRegr.fit(train_img, train_lbl)

"""
Predicción con datos de Test
"""
logisticRegr.predict(test_img[0:10000])

"""
Vamos a medir el desempeño del modelo
"""
score = logisticRegr.score(test_img, test_lbl)

print("Finalizado!!!")

#PARA REPRESENTAR LAS IMÁGENES VER EL SIGUIENTE CÓDIGO -> https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Image_Reconstruction_and_such.ipynb

