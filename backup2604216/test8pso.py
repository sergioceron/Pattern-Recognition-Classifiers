import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy
import urllib
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut

from pyswarm import pso

class prettyfloat(float):
    def __repr__(self):
        return "% 9.4f" % float(self)

def printc(coef1, coef2, coef3):
    for i in range( len( coef1 ) ):
        print "% 4d % 15s, % 15s, % 15s" % (i+1, coef1[i], coef2[i], coef3[i])

# TESTING PHASE ( FOR PSO WILL BE USED AS COST FUNCTION )

def J(p, *args):
    ds, c, m, N = args

    #print '====================== INDICES ======================'
    #for i in range( len( p ) ):
    #    print int( p[i] )

    memorias = []
    for i in range( c ):
        memoria = []
        for j in range( m ):
            index = int( p [ i*m + j ] ) # round lower
            memoria += [ ds[ index ][0:m] ]
        memorias += [ np.array( memoria ) ]

    #print '====================== MEMORIES ======================'
    #for i in range( c ):
    #    print i, memorias[i]

    # TRAINING PHASE SECOND STAGE (COEFFICIENT GENERATION)
    #print '====================== COEFFICIENTS ======================'
    coeficientes = {}
    for i in range( N ):
        test = np.array( ds[i, 0:m] )

        if not coeficientes.has_key( str(int(ds[i, m])) ):
            coeficientes[ str(int(ds[i, m])) ] = []

        try :
            for j in range( c ):
                cc = map( prettyfloat, np.linalg.solve( np.transpose( memorias[j] ), test ) )
                if ds[i, m] == j: # si es una clase nominal va a fallar
                    coeficientes[ str(j) ] += [cc] # AQUI ESTA EL PUTO ERROR, LOS CC MEANS DEBEN SER UN VECTOR NO UN VALOR, Y ESO ES POR QUE COEFICIENTES ES UN VECTOR 1D Y DEBE SER 2D
        except:
            return 1

    cc_means = []
    for i in range( c ):
        cc_means += [np.mean( coeficientes[str(i)], axis = 0 )]
        #print np.mean( coeficientes[str(i)], axis = 0 )

    #print '====================== TESTING ======================'
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
            correct += 1

    #print '=============================================='
    performance = float((correct/float(N)))
    print 'Performance: %3.4f' %  performance
    return 1-performance


iris = datasets.load_iris()
Y = iris.target

#raw_data = urllib.urlopen( "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data" )
raw_data = urllib.urlopen( "http://goo.gl/j0Rvxq" )
# load the CSV file as a numpy matrix
#ds = np.column_stack((iris.data, Y))
ds = np.loadtxt(raw_data, delimiter=",")

m = len( ds[0] ) - 1 # size of dataset (iris=150)
c = 2 # number of classes
N = len( ds ) # numero de patrones en total

# TRAINING PHASE FIRST STAGE (SUPPORT PATTERNS)

# This method uses the c-nearest to mean of every class
'''
memorias = []
for i in range( c ):
    classe = np.array( [e[0:m] for e in ds if e[m] == i] ) # vectores de la clase actual
    #mean = scipy.stats.hmean( classes[i], axis = 0 )
    if len( classe ) > 0 :
        mean = scipy.stats.gmean( classe, axis = 0 )
        searchTree = scipy.spatial.cKDTree( np.copy( classe ), leafsize = 100 )
        dummy, nearest = searchTree.query( mean, k = m, p = m )
        memorias += [np.array( classe[ nearest ]) ]
    else:
        memorias += [[]]
'''
# This method uses a pso metaheuristic
args = (ds, c, m, N)
#lb = [ 0,  0,  0,  0, 50, 50, 50, 50, 100, 100, 100, 100]
#ub = [49, 49, 49, 50, 99, 99, 99, 99, 149, 149, 149, 149]
lb = [  0,   0,   0,   0,   0,   0,   0,   0, 499, 499, 499, 499, 499, 499, 499, 499]
ub = [498, 498, 498, 498, 498, 498, 498, 498, 767, 767, 767, 767, 767, 767, 767, 767 ]

xopt, fopt = pso(J, lb, ub, args=args, debug=False, maxiter=1, swarmsize=1000, omega=1, phip=1, phig=15, minstep=1) # it=10, par=30, phig=5 good choice

print '====================== FINAL MEMORIES ======================'
memorias = []
for i in range( c ):
    memoria = []
    for j in range( m ):
        index = int( xopt [ i*m + j ] ) # round lower
        memoria += [ ds[ index ][0:m] ]
    memorias += [ np.array( memoria ) ]

for i in range( c ):
    print 'Memoria ', i
    print memorias[i]

J( xopt, *args )
