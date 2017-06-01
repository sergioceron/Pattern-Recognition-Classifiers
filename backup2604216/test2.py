import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut

class prettyfloat(float):
    def __repr__(self):
        return "% 9.4f" % float(self)

def printc(coef1, coef2, coef3):
    for i in range( len( coef1 ) ):
        print "% 4d % 15s, % 15s, % 15s" % (i+1, coef1[i], coef2[i], coef3[i])

iris = datasets.load_iris()
Y = iris.target

ds = iris.data

#loo = LeaveOneOut(ds)
#for train, test in loo:
#    print ("%s %s" % (train, test))


ds = np.column_stack((ds, Y)) # Dataset completo

c1 = np.array( [e[0:4] for e in ds if e[4] == 0] ) # vectores de la clase 1
c2 = np.array( [e[0:4] for e in ds if e[4] == 1] ) # vectores de la clase 2
c3 = np.array( [e[0:4] for e in ds if e[4] == 2] ) # vectores de la clase 3

# Como seleccionar estos 4 vectores
# Se toman de manera aletaoria
rnd.shuffle( c1 )
rnd.shuffle( c2 )
rnd.shuffle( c3 )

# n vectores donde n = tamanioo del vecto
c1_v = c1[0:4]  # 3 vectores aleatorios de la primera clase
c2_v = c2[0:4]  # 3 vectores aleatorios de la segunda clase
c3_v = c3[0:4]  # 3 vectores aleatorios de la tercera clase

N = len( ds ) # numero de patrones en total

# TRAINING PHASE

memoria_clase_1 = np.array( c1_v )
memoria_clase_2 = np.array( c2_v )
memoria_clase_3 = np.array( c3_v )

coeficientes_c1 = []
coeficientes_c2 = []
coeficientes_c3 = []
for i in range(N):
    test = np.array( ds[i, 0:4] )

    cc1 = map( prettyfloat, np.linalg.solve( np.transpose( memoria_clase_1 ), test ) )
    cc2 = map( prettyfloat, np.linalg.solve( np.transpose( memoria_clase_2 ), test ) )
    cc3 = map( prettyfloat, np.linalg.solve( np.transpose( memoria_clase_3 ), test ) )

    print cc1, cc2, cc3, ds[i, 4]

    if ds[i, 4] == 0:
        coeficientes_c1 += [cc1]
    if ds[i, 4] == 1:
        coeficientes_c2 += [cc2]
    if ds[i, 4] == 2:
        coeficientes_c3 += [cc3]

    # aqui es donde se va a trabajar con las pruebas
    # fin de las pruebas
    #varianza = np.var( coeficientes )
#mean_c1 = np.mean( coeficientes_c1 )
#mean_c2 = np.mean( coeficientes_c2 )
#mean_c3 = np.mean( coeficientes_c3 )

print '=============================================='
printc( coeficientes_c1, coeficientes_c2, coeficientes_c3 )
print '=============================================='
# Se calcula el vector medio ( centroide ) de los coeficientes correspondientes
# a la combinacion lineal entre el vector y los vectores de apoyo
cc1_mean = stats.mode( coeficientes_c1, axis = 0 )
cc2_mean = stats.mode( coeficientes_c2, axis = 0 )
cc3_mean = stats.mode( coeficientes_c3, axis = 0 )
print cc1_mean, cc2_mean, cc3_mean

# TESTING PHASE
print '=============================================='

correct = 0
for i in range( N ):
    test = np.array( ds[i, 0:4] )

    # coeficientes calculados hacia cada memoria
    c1test = np.linalg.solve( np.transpose( memoria_clase_1 ), test )
    c2test = np.linalg.solve( np.transpose( memoria_clase_2 ), test )
    c3test = np.linalg.solve( np.transpose( memoria_clase_3 ), test )

    # distancias entre los coeficientes
    dist_c1 = np.linalg.norm( c1test - cc1_mean )
    dist_c2 = np.linalg.norm( c2test - cc2_mean )
    dist_c3 = np.linalg.norm( c3test - cc3_mean )

    dist = [ dist_c1, dist_c2, dist_c3 ]
    c = np.argmin( dist )
    r = ds[i, 4]

    if c == r:
        print test, '->', r, dist, c, 'correct'
        correct += 1
    else :
        print test, '->', r, dist, c, 'incorrect'
print 'Performance: %3.4f' %  float((correct/150.0*100))

print '=============================================='

print c1_v, c2_v, c3_v
