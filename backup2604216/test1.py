import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from sklearn import datasets

class prettyfloat(float):
    def __repr__(self):
        return "% 9.4f" % self

iris = datasets.load_iris()
Y = iris.target

ds = iris.data

ds = np.column_stack((ds, Y)) # Dataset completo

c1 = np.array( [e for e in ds if e[4] == 0] ) # vectores de la clase 1
c2 = np.array( [e for e in ds if e[4] == 1] ) # vectores de la clase 2
c3 = np.array( [e for e in ds if e[4] == 2] ) # vectores de la clase 3

x = ds[:, 0] # componente 1
y = ds[:, 1] # componente 2

c1_a = rnd.choice( c1 )[0:2]  # vector aleatorio de la primera clase (tomando solo 2 componentes
c1_b = rnd.choice( c1 )[0:2]  # vector aleatorio de la primera clase (tomando solo 2 componentes

c2_a = rnd.choice( c2 )[0:2]  # vector aleatorio de la segunda clase (tomando solo 2 componentes
c2_b = rnd.choice( c2 )[0:2]  # vector aleatorio de la segunda clase (tomando solo 2 componentes

c3_a = rnd.choice( c3 )[0:2]  # vector aleatorio de la tercera clase (tomando solo 2 componentes
c3_b = rnd.choice( c3 )[0:2]  # vector aleatorio de la tercera clase (tomando solo 2 componentes

N = 150 # numero de patrones en total

memoria_clase_1 = np.array( [c1_a, c1_b] )
memoria_clase_2 = np.array( [c2_a, c2_b] )
memoria_clase_3 = np.array( [c3_a, c3_b] )

coeficientes_c1 = []
coeficientes_c2 = []
coeficientes_c3 = []
for i in range(N):
    test = np.array( [ x[i], y[i] ] )

    coeficientes_c1 += map( prettyfloat, np.linalg.solve( np.transpose( memoria_clase_1 ), test ) )
    coeficientes_c2 += map( prettyfloat, np.linalg.solve( np.transpose( memoria_clase_2 ), test ) )
    coeficientes_c3 += map( prettyfloat, np.linalg.solve( np.transpose( memoria_clase_3 ), test ) )
    # aqui es donde se va a trabajar con las pruebas
    # fin de las pruebas
    #varianza = np.var( coeficientes )
mean_c1 = np.mean( coeficientes_c1 )
mean_c2 = np.mean( coeficientes_c2 )
mean_c3 = np.mean( coeficientes_c3 )

print mean_c1, mean_c2, mean_c3
