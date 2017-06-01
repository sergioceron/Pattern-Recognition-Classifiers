# EN ESTA VERSION LO UNICO QUE SE HACE ES APLICAR EL METODO LEAVE ONE OUT PARA
# CALCULAR EL VERDADERO PERFORMANCE

import numpy as np
import numpy.random as rnd
from random import randrange
import matplotlib.pyplot as plt
import scipy
import urllib
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut
from pyswarm import pso

class prettyfloat(float):
    def __repr__(self):
        return "% 9.4f" % float(self)

# COST FUNCTION
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

    #print '====================== TESTING ======================'
    correct = 0

    for _i in range( c ):
        i = randrange(0, len(ds))
        test = np.array( ds[i, 0:m] )

        # coeficientes calculados hacia cada memoria
        varianzas = []
        for j in range( c ):
            try :
                ctest = np.linalg.solve( np.transpose( memorias[j] ), test )
                varianzas += [ np.var( ctest ) ]
            except:
                varianzas += [ float('inf') ]


        cr = np.argmin( varianzas )
        r = ds[i, m]
        if cr == r:
            correct += 1

    #print '=============================================='
    performance = float((correct/float(c)))
    #print 'Performance: %3.4f' %  performance
    return 1-performance

# MAIN CODE PROGRAM
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

    # ====================== SE CALCULAN LAS MEMORIAS ======================
    args = (train_ds, c, m, N - 1) # -1 for LeaveOneOut

    # CACULAMOS EL UB Y LB
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

    # EN XOPT ESTAN LOS INDICES DE LOS PATRONES QUE FORMARAN A LAS MEMORIAS
    xopt, fopt = pso(J, lb, ub, args=args, debug=False, maxiter=10, swarmsize=10, omega=1, phip=1, phig=15, minstep=1) # it=10, par=30, phig=5 good choice

    # ====================== FINAL MEMORIES ======================
    memorias = []
    for i in range( c ):
        memoria = []
        for j in range( m ):
            index = int( xopt [ i*m + j ] ) # round lower
            memoria += [ train_ds[ index ][0:m] ]
        memorias += [ np.array( memoria ) ]

    # ====================== SE PRUEBA EL PATRON DEL LOO ======================
    test = test_ds[0, 0:m]

    varianzas = []
    for j in range( c ):
        try:
            ctest = np.linalg.solve( np.transpose( memorias[j] ), test )
            varianzas += [ np.var( ctest ) ]
        except:
            varianzas += [ float('inf') ]

    cr = np.argmin( varianzas )
    r = test_ds[0, m]
    if cr == r:
        correct += 1

print 'Final Performance: %3.4f' %  float((correct/float(N)))
