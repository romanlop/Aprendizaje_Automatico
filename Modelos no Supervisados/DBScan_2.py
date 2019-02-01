#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 21:51:31 2019

@author: Ruman
"""

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np

moons, moons_classes = make_moons(n_samples = 200,
                                  noise = 0.05,
                                  random_state = 1)

plt.scatter(moons[:, 0],moons[:, 1])
plt.title("Raw data")
plt.show()

color_map = np.array(['b','g','r','c','m','y','k'])


km = KMeans(n_clusters = 3,
            random_state = 1).fit_predict(moons)


ac = AgglomerativeClustering(n_clusters = 3, 
                             affinity = 'euclidean',
                             linkage = 'complete').fit_predict(moons)

db = DBSCAN(eps = 0.2,
            min_samples = 5,
            metric = 'euclidean').fit_predict(moons)



plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.title('k-means', size=18)
plt.scatter(moons[:, 0], moons[:, 1], s=10, color = color_map[km])


plt.subplot(1, 3, 2)
plt.title('AgglomerativeClustering', size=18)
plt.scatter(moons[:, 0], moons[:, 1], s=10, color = color_map[ac])


plt.subplot(1, 3, 3)
plt.title('DBSCAN', size=18)
plt.scatter(moons[:, 0], moons[:, 1], s=10, color = color_map[db])

