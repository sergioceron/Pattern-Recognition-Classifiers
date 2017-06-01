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

m = len( ds[0] ) # size of dataset (iris=150)
c = 3 # number of classes

#loo = LeaveOneOut(ds)
#for train, test in loo:
#    print ("%s %s" % (train, test))

ds = np.column_stack((ds, Y)) # Dataset completo

classes = []
means = []
nearests = []
memorias = []
for i in range( c ):
    classes += [np.array( [e[0:m] for e in ds if e[m] == i] )] # vectores de la clase i
    means += [scipy.stats.hmean( classes[i], axis = 0 )]
    searchTree = scipy.spatial.cKDTree( np.copy( classes[i] ), leafsize = 100 )
    dummy, nearest = searchTree.query( means[i], k = m, p = m )
    nearests +=  [nearest]
    memorias += [np.array( classes[i][ nearests[i] ]) ]


N = len( ds ) # numero de patrones en total

# TRAINING PHASE

for i in range( c ):
    print 'c', i, memorias[i]

coeficientes = []
for i in range( N ):
    test = np.array( ds[i, 0:m] )

    cc = []
    for j in range( c ):
        cc += [map( prettyfloat, np.linalg.solve( np.transpose( memorias[j] ), test ) )]
        print cc[j]

    print ds[i, m]

    for j in range( c ):
        if ds[i, m] == j:
            coeficientes += [cc[j]]


print '=============================================='
#printc( coeficientes[0], coeficientes[1], coeficientes[2] )
print '=============================================='
# Se calcula el vector medio ( centroide ) de los coeficientes correspondientes
# a la combinacion lineal entre el vector y los vectores de apoyo
cc_means = []
for i in range( c ):
    cc_means += [np.mean( coeficientes[i], axis = 0 )]

#print cc_means[0], cc_means[1], cc_means[2]

# TESTING PHASE
print '=============================================='

correct = 0
for i in range( N ):
    test = np.array( ds[i, 0:m] )

    # coeficientes calculados hacia cada memoria
    ctest = []
    for j in range( c ):
        ctest += [np.linalg.solve( np.transpose( memorias[j] ), test )]

    # distancias entre los coeficientes
    dists = []
    for j in range( c ):
        dists += [np.linalg.norm( ctest[j] - cc_means[j] )]

    cr = np.argmin( dists )
    r = ds[i, m]

    if cr == r:
        print test, '->', r, dists, cr, 'correct'
        correct += 1
    else :
        print test, '->', r, dists, cr, 'incorrect'
print 'Performance: %3.4f' %  float((correct/float(N)*100))

print '=============================================='

print #c1_v, c2_v, c3_v
