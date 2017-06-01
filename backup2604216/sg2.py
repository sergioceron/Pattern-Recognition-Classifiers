import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy import stats
import scipy
import urllib
import sys
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut, train_test_split
from pyswarm import pso

def get_memories_from_farest( ds, num_clases, pattern_size ):
    classes  = []
    means    = []
    nearests = []
    memorias = []
    for i in range( num_clases ):
        classes += [np.array( [e[0:pattern_size] for e in ds if e[pattern_size] == i] )]
        means += [scipy.stats.gmean( classes[i], axis = 0 )]
        searchTree = scipy.spatial.cKDTree( np.copy( classes[i] ), leafsize = 100 )
        dummy, nearest = searchTree.query( means[i], k = len( classes[i] ) ) # todos los patrones de la clase
        nearests += [ nearest[ len( classes[i] )-pattern_size:len( classes[i] ) ]]
        memorias += [ np.array( classes[i][ nearests[i] ]) ]

    return memorias

def get_memories_from_centroids( ds, num_clases, pattern_size ):
    classes = []
    means = []
    nearests = []
    memorias = []
    for i in range( num_clases ):
        classes += [np.array( [e[0:pattern_size] for e in ds if e[pattern_size] == i] )]
        means += [scipy.stats.gmean( classes[i], axis = 0 )]
        searchTree = scipy.spatial.cKDTree( np.copy( classes[i] ), leafsize = 100 )
        dummy, nearest = searchTree.query( means[i], k = pattern_size, p = pattern_size )
        nearests +=  [nearest]
        memorias += [np.array( classes[i][ nearests[i] ]) ]

    return memorias

def get_memories_from_pso( ds, indexes, num_clases, pattern_size ):
    memories = []

    for i in range( num_clases ):
        memory = []
        for j in range( pattern_size ):
            index = int( indexes [ i*pattern_size + j ] ) # round lower
            memory += [ ds[ index ][0:pattern_size] ]
        memories += [ np.array( memory ) ]

    return memories

def get_coeficientes( ds, memories ):
    c = len( memories )
    coeficientes = {}

    for i in range( c ):
        coeficientes[ str( i ) ] = []

    for i in range( len( ds ) ):
        test = np.array( ds[i, 0:m] )
        tclass = ds[i, m]
        try:
            cc = np.linalg.solve( np.transpose( memories[ int( tclass ) ] ), test )
            coeficientes[ str( int( tclass ) ) ] += [ cc ]
        except:
            pass

    return coeficientes

def get_distances( coeficientes, memories, test ):
    c = len( memories )
    distances = []
    for clase in range( c ):
        coeficiente = coeficientes[ str( clase ) ]
        try :
            ctest = np.linalg.solve( np.transpose( memories[ int( clase ) ] ), test )
            dists = []
            for coef in coeficiente:
                #dists += [ np.linalg.norm( ctest - coef ) ]
                dists += [ scipy.spatial.distance.cdist([ctest], [coef], 'chebyshev')[0][0] ]
            #distances += [ stats.mode( dists )[0][0] ]
            #distances += [ np.mean( dists ) ]
            distances += [ min( dists ) ]
        except:
            distances += [ 9e20 ]

    return distances

# COST FUNCTION
def J(p, *args):
    ds, c, m, N = args
    # ====================== MEMORIES ======================
    memorias = get_memories_from_pso( ds, p, c, m )

    # ====================== COEFFICIENTS ======================
    train, testds = train_test_split(ds, test_size=0.3, random_state=12)

    coeficientes = get_coeficientes( train, memorias )

    #print '====================== TESTING ======================'
    correct = 0
    for i in range( len( testds ) ):
        test = np.array( testds[i, 0:m] )
        tclass = testds[i, m]

        distances = get_distances( coeficientes, memorias, test )

        minclass = np.argmin( distances )

        if minclass == tclass:
            correct += 1

    performance = float((correct/float(len(testds))))
    #print 'Performance: %3.4f' %  performance
    return 1.0-performance

# MAIN CODE PROGRAM
iris = datasets.load_iris()
Y = iris.target
#raw_data = urllib.urlopen( "file:///Users/sergio/Downloads/glass.csv" )
#raw_data = urllib.urlopen( "http://goo.gl/j0Rvxq" )
# load the CSV file as a numpy matrix
#ds = np.column_stack((iris.data, Y))
#ds = np.loadtxt("/Users/sergio/Downloads/mammographic_filled.csv", delimiter=",")
ds = np.loadtxt("/Users/sergio/Downloads/wine.csv", delimiter=",")

m = len( ds[0] ) - 1                # size of pattern (ej. iris = 4)
c = len( np.unique( ds[ :, m ] ) )  # number of classes
N = len( ds )                       # size of dataset

# TRAINING PHASE FIRST STAGE (SUPPORT PATTERNS)
correct = 0
loo = LeaveOneOut( N )
for train_index, test_index in loo:
    train_ds = ds[ train_index ]
    test_ds = ds[ test_index ]

    test = np.array( test_ds[0, 0:m] )
    tclass = test_ds[0, m]

    # ====================== SE CALCULAN LAS MEMORIAS ======================
    args = (train_ds, c, m, N - 1) # -1 because LeaveOneOut

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
    #xopt, fopt = pso(J, lb, ub, args=args, debug=False, maxiter=10, swarmsize=10, omega=1, phip=1, phig=15, minstep=1) # it=10, par=30, phig=5 good choice
    #xopt, fopt = pso(J, lb, ub, args=args, debug=False, minfunc=9e-3, maxiter=1, swarmsize=1)

    # ====================== FINAL MEMORIES ======================
    #memorias = get_memories_from_pso( train_ds, xopt, c, m )
    memorias = get_memories_from_centroids( train_ds, c, m )
    #print memorias
    # ====================== SE PRUEBA EL PATRON DEL LOO ======================
    coeficientes = get_coeficientes( train_ds, memorias )

    distances = get_distances( coeficientes, memorias, test )

    minclass = np.argmin( distances )

    print tclass, '<->', distances

    if minclass == tclass:
        correct += 1

print 'Final Performance: %3.4f' %  float((correct/float(N)))
