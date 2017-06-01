import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy
import urllib
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
    
#raw_data = urllib.urlopen( "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data" )
raw_data = urllib.urlopen( "http://goo.gl/j0Rvxq" )
# load the CSV file as a numpy matrix
#ds = np.column_stack((iris.data, Y)) 
ds = np.loadtxt(raw_data, delimiter=",")

m = len( ds[0] ) - 1 # size of dataset (iris=150)
c = 2 # number of classes

classes = []
memorias = []
for i in range( c ):
    classes += [np.array( [e[0:m] for e in ds if e[m] == i] )] # vectores de la clase i
    #mean = scipy.stats.hmean( classes[i], axis = 0 )
    if len( classes[i] ) > 0 :
        mean = scipy.stats.gmean( classes[i], axis = 0 )
        searchTree = scipy.spatial.cKDTree( np.copy( classes[i] ), leafsize = 100 )
        dummy, nearest = searchTree.query( mean, k = m, p = m )
        memorias += [np.array( classes[i][ nearest ]) ]
    else:
        memorias += [[]]


N = len( ds ) # numero de patrones en total

# TRAINING PHASE

for i in range( c ):
    print 'memoria', i, memorias[i]

coeficientes = []
for i in range( N ):
    test = np.array( ds[i, 0:m] )

    for j in range( c ):
        cc = map( prettyfloat, np.linalg.solve( np.transpose( memorias[j] ), test ) )
        if ds[i, m] == j:
            coeficientes += [cc]

# Se calcula el vector medio ( centroide ) de los coeficientes correspondientes
# a la combinacion lineal entre el vector y los vectores de apoyo
cc_means = []
for i in range( c ):
    cc_means += [np.mean( coeficientes[i], axis = 0 )]


# TESTING PHASE
print '====================== TESTING ======================'

correct = 0
for i in range( N ):
    test = np.array( ds[i, 0:m] )

    # coeficientes calculados hacia cada memoria
    dists = []
    for j in range( c ):
        ctest = np.linalg.solve( np.transpose( memorias[j] ), test )
        dists += [np.linalg.norm( ctest - cc_means[j] )]

    cr = np.argmin( dists )
    r = ds[i, m]

    if cr == r:
        print test, '->', r, dists, cr, 'correct'
        correct += 1
    else :
        print test, '->', r, dists, cr, 'incorrect'

print '=============================================='
print 'Performance: %3.4f' %  float((correct/float(N)*100))