# +CON TRASLACION, PERO NO MEJORA MUCHO EL PERFORMANCE
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

def get_memories_from_centroids( ds, num_clases, pattern_size ):
    classes = []
    means = []
    nearests = []
    memorias = []
    for i in range( num_clases ):
        classes += [np.array( [e[0:pattern_size] for e in ds if e[pattern_size] == i] )]
        means += [ scipy.stats.variation( classes[i], axis = 0 ) ]
        searchTree = scipy.spatial.cKDTree( np.copy( classes[i] ), leafsize = 100 )
        dummy, nearest = searchTree.query( means[i], k = pattern_size, p = pattern_size )
        nearests +=  [nearest]
        print means[i], nearest
        memorias += [np.array( classes[i][ nearests[i] ]) ]

    return memorias

def translate( ds, centroide ):
    ds_copy = []
    for i in range( len( ds ) ):
        ds_copy += [ np.subtract( ds[i], centroide ) ]
        #print 'T:', ds[i], '+', centroide, '=', ds_copy[i]
    return np.array( ds_copy )

def get_coeficientes( ds, memories ):
    c = len( memories )
    coeficientes = {}

    for i in range( c ):
        coeficientes[ str( i ) ] = []

    for i in range( len( ds ) ):
        test = np.array( ds[i, 0:m] )
        tclass = ds[i, m]
        cc = []
        for j in range( c ):
            try:
                cc = np.concatenate((cc, np.linalg.solve( np.transpose( memories[ int( j ) ] ), test ) ))
            except:
                cc = np.concatenate(( cc, np.zeros( m ) ))
                pass
        #print 'cc:', len(cc)
        coeficientes[ str( int( tclass ) ) ] += [ cc ]

    return coeficientes

# GENERAR EL VECTOR CTEST COMPLETO PARA CALCULAR LA DISTANCIA Y LUEGO
# CREAR UN ONE VECTOR CON ZERO EN LA COMPONENTE QUE REPRESENTA LA CLASE (ZERO POR EL MIN)
def get_distances( coeficientes, memories, test ):
    c = len( memories )

    ctest = []
    for j in range( c ):
        try :
            ctest = np.concatenate(( ctest, np.linalg.solve( np.transpose( memories[ j ] ), test ) ))
        except:
            ctest = np.concatenate(( ctest, np.zeros( len( test ) ) )) #m = len( test )

    distances = []
    for clase in range( c ):
        coeficiente = coeficientes[ str( clase ) ]

        dists = []
        for coef in coeficiente:
            dists += [ scipy.spatial.distance.cdist([ctest], [coef], 'cityblock')[0][0] ]
        distances += [ min( dists ) ]

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
ds = np.column_stack((iris.data, Y))
#ds = np.loadtxt("/Users/sergio/Downloads/mammographic_filled.csv", delimiter=",") # ok
#ds = np.loadtxt("/Users/sergio/Downloads/wine.csv", delimiter=",") # ok
#ds = np.loadtxt("/Users/sergio/Downloads/parkinsons.csv", delimiter=",") # ok
#ds = np.loadtxt("/Users/sergio/Downloads/w-breast-cancer.csv", delimiter=",")
#ds = np.loadtxt("/Users/sergio/Downloads/tae.csv", delimiter=",")
#ds = np.loadtxt("/Users/sergio/Downloads/ecoli.csv", delimiter=",")
#ds = np.loadtxt("/Users/sergio/Downloads/habberman.csv", delimiter=",") # ok
#ds = np.loadtxt("/Users/sergio/Downloads/heart-stat.csv", delimiter=",")

m = len( ds[0] ) - 1                # size of pattern (ej. iris = 4)
c = len( np.unique( ds[ :, m ] ) )  # number of classes
N = len( ds )                       # size of dataset

# TRAINING PHASE FIRST STAGE (SUPPORT PATTERNS)
correct = 0
loo = LeaveOneOut( N )
for train_index, test_index in loo:
    train_ds = ds[ train_index ]
    test_ds = ds[ test_index ]

    centroide = np.concatenate( [ np.mean( train_ds[:, 0:m], axis=0 ) , [0] ])
    train_ds_t = translate( train_ds, centroide )

    memorias = get_memories_from_centroids( train_ds_t, c, m )
    coeficientes = get_coeficientes( train_ds_t, memorias )

    for i in range( len( test_ds ) ):
        test_t = np.subtract( test_ds[i], centroide )

        test = np.array( test_t[0:m] )
        tclass = test_t[m]

        distances = get_distances( coeficientes, memorias, test )

        minclass = np.argmin( distances )

        print tclass, '<->', distances, ' :: ', (minclass==tclass)

        if minclass == tclass:
            correct += 1

print 'Final Performance: %3.4f' %  float((correct/float(N)))
