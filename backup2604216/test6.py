import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy
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

m = len(ds[0])

#loo = LeaveOneOut(ds)
#for train, test in loo:
#    print ("%s %s" % (train, test))


ds = np.column_stack((ds, Y)) # Dataset completo

c1 = np.array( [e[0:m] for e in ds if e[m] == 0] ) # vectores de la clase 1
c2 = np.array( [e[0:m] for e in ds if e[m] == 1] ) # vectores de la clase 2
c3 = np.array( [e[0:m] for e in ds if e[m] == 2] ) # vectores de la clase 3

# Como seleccionar estos 4 vectores
# Se toman de manera aletaoria
mean_c1 = scipy.stats.hmean( c1, axis = 0 )
mean_c2 = scipy.stats.hmean( c2, axis = 0 )
mean_c3 = scipy.stats.hmean( c3, axis = 0 )

searchTree_c1 = scipy.spatial.cKDTree( np.copy( c1 ), leafsize = 100 )
searchTree_c2 = scipy.spatial.cKDTree( np.copy( c2 ), leafsize = 100 )
searchTree_c3 = scipy.spatial.cKDTree( np.copy( c3 ), leafsize = 100 )

dummy1, nearest_c1 = searchTree_c1.query( mean_c1, k = m, p = m )
dummy2, nearest_c2 = searchTree_c2.query( mean_c2, k = m, p = m )
dummy3, nearest_c3 = searchTree_c3.query( mean_c3, k = m, p = m )

N = len( ds ) # numero de patrones en total

# TRAINING PHASE

memoria_clase_1 = np.array( c1[nearest_c1] )
memoria_clase_2 = np.array( c2[nearest_c2] )
memoria_clase_3 = np.array( c3[nearest_c3] )

print 'c1', memoria_clase_1
print 'c2', memoria_clase_2
print 'c3', memoria_clase_3

coeficientes_c1 = []
coeficientes_c2 = []
coeficientes_c3 = []
for i in range(N):
    test = np.array( ds[i, 0:m] )

    cc1 = map( prettyfloat, np.linalg.solve( np.transpose( memoria_clase_1 ), test ) )
    cc2 = map( prettyfloat, np.linalg.solve( np.transpose( memoria_clase_2 ), test ) )
    cc3 = map( prettyfloat, np.linalg.solve( np.transpose( memoria_clase_3 ), test ) )

    print cc1, cc2, cc3, ds[i, m]

    if ds[i, m] == 0:
        coeficientes_c1 += [cc1] # Aqui puede estar un problema
    if ds[i, m] == 1:
        coeficientes_c2 += [cc2]
    if ds[i, m] == 2:
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
cc1_mean = np.mean( coeficientes_c1, axis = 0 )
cc2_mean = np.mean( coeficientes_c2, axis = 0 )
cc3_mean = np.mean( coeficientes_c3, axis = 0 )
print cc1_mean, cc2_mean, cc3_mean

# TESTING PHASE
print '=============================================='

correct = 0
for i in range( N ):
    test = np.array( ds[i, 0:m] )

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
    r = ds[i, m]

    if c == r:
        print test, '->', r, dist, c, 'correct'
        correct += 1
    else :
        print test, '->', r, dist, c, 'incorrect'
print 'Performance: %3.4f' %  float((correct/float(N)*100))

print '=============================================='

print #c1_v, c2_v, c3_v
