# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
Y = iris.target
ds = np.column_stack((iris.data, Y))

markers = [ 'o', 'v', '*' ]
color = [ 'blue', 'green', 'red' ]
scatters = []
for i in range( 3 ):
    clase = np.array( [e[0:4] for e in ds if e[4] == i] )
    print clase
    scatters += [ plt.scatter(clase[:,0], clase[:,3], marker=markers[i], color=color[i], s=50, facecolors='none', edgecolors=color[i]) ]

plt.legend((scatters[0], scatters[1], scatters[2]),
           ('Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica'),
           scatterpoints=1,
           loc='lower left',
           ncol=3)

plt.xlabel('Rasgo 1 (Largo del Sépalo)')
plt.ylabel('Rasgo 4 (Ancho del Pétalo)')
plt.grid()
plt.show()
