# EN ESTA VERSION LO UNICO QUE SE HACE ES APLICAR EL METODO LEAVE ONE OUT PARA
# CALCULAR EL VERDADERO PERFORMANCE
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

    # ====================== MEMORIES ======================
    memorias = []
    for i in range( c ):
        memoria = []
        for j in range( m ):
            index = int( p [ i*m + j ] ) # round lower
            memoria += [ ds[ index ][0:m] ]
        memorias += [ np.array( memoria ) ]

    # ====================== COEFFICIENTS ======================
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
    #print 'Performance: %3.4f' %  performance
    return 1-performance

iris = datasets.load_iris()
Y = iris.target

#raw_data = urllib.urlopen( "file:///Users/sergio/Downloads/glass.csv" )
#raw_data = urllib.urlopen( "http://goo.gl/j0Rvxq" )
# load the CSV file as a numpy matrix
#ds = np.column_stack((iris.data, Y))
ds = np.loadtxt("/Users/sergio/Downloads/mammographic_filled.csv", delimiter=",")
#ds = np.loadtxt("/Users/sergio/Downloads/wine.csv", delimiter=",")

m = len( ds[0] ) - 1                # size of dataset (ej. iris = 150)
c = len( np.unique( ds[ :, m ] ) )  # number of classes
N = len( ds )                       # numero de patrones en total

# TRAINING PHASE FIRST STAGE (SUPPORT PATTERNS)
correct = 0
loo = LeaveOneOut( N )
for train_index, test_index in loo:
    train_ds = ds[ train_index ]
    test_ds = ds[ test_index ]

    args = (train_ds, c, m, N - 1) # -1 for LeaveOneOut

    clases = []
    for i in range( c ):
        clase = np.array( [e[0:m] for e in train_ds if e[m] == i] )
        clases += [clase]

    lb = []
    ub = []
    offset = 0
    for i in range( c ):
        for j in range( m ):
            lb += [ offset ]
            ub += [ offset + len( clases[i] ) - 1 ]
        offset += len( clases[i] )

    print lb
    print ub

    xopt, fopt = pso(J, lb, ub, args=args, debug=False, maxiter=10, swarmsize=10, omega=1, phip=1, phig=15, minstep=1) # it=10, par=30, phig=5 good choice

    # ====================== FINAL MEMORIES ======================
    memorias = []
    for i in range( c ):
        memoria = []
        for j in range( m ):
            index = int( xopt [ i*m + j ] ) # round lower
            memoria += [ train_ds[ index ][0:m] ]
        memorias += [ np.array( memoria ) ]

    # TEST PATTERN
    coeficientes = {}
    for i in range( N - 1 ):
        test = test_ds[0, 0:m]

        if not coeficientes.has_key( str(int(train_ds[i, m])) ):
            coeficientes[ str(int(train_ds[i, m])) ] = []

        for j in range( c ):
            cc = map( prettyfloat, np.linalg.solve( np.transpose( memorias[j] ), test ) )
            if train_ds[i, m] == j: # si es una clase nominal va a fallar
                coeficientes[ str(j) ] += [cc]

    cc_means = []
    for i in range( c ):
        cc_means += [np.mean( coeficientes[str(i)], axis = 0 )]

    dists = []
    for j in range( c ):
        ctest = np.linalg.solve( np.transpose( memorias[j] ), test )
        dists += [np.linalg.norm( ctest - cc_means[j] )]

    cr = np.argmin( dists )
    r = test_ds[0, m]
    if cr == r:
        correct += 1

print 'Final Performance: %3.4f' %  float((correct/float(N)))
    #for i in range( c ):
    #    print 'Memoria ', i
    #    print memorias[i]

    #J( xopt, *args )
