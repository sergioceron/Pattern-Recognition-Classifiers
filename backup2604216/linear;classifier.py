import os
import time
import numpy as np
import numpy.random as rnd
from scipy import stats
import scipy
import urllib
import sys
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut, train_test_split

from ui import DownloadProgressSpinner, DownloadProgressSpinnerMoon, DownloadProgressSpinnerPie, DownloadProgressSpinnerLine, DownloadProgressBarShady, DownloadProgressBarCharging, DownloadProgressBarFillingSquares, DownloadProgressBarFillingCircles

def get_memories_from_centroids( ds, num_clases, pattern_size ):
    memorias = []
    for i in range( num_clases ):
        clase = np.array( [e[0:pattern_size] for e in ds if e[pattern_size] == i] )
        mean  = scipy.stats.variation( clase, axis = 0 )
        searchTree = scipy.spatial.cKDTree( np.copy( clase ), leafsize = 100 )
        dummy, nearest = searchTree.query( mean, k = pattern_size, p = pattern_size )
        #print mean, nearest
        memorias += [ np.array( clase[ nearest ]) ]

    return memorias

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
            dists += [ scipy.spatial.distance.cdist( [ctest], [coef], 'cityblock' )[0][0] ]
        distances += [ min( dists ) ]

    return distances

# MAIN CODE PROGRAM
iris = datasets.load_iris()
Y = iris.target
ds = np.column_stack((iris.data, Y))

#ds = np.loadtxt("/Users/sergio/Downloads/mammographic_filled.csv", delimiter=",") # ok
#ds = np.loadtxt("wine.csv", delimiter=",") # ok
#ds = np.loadtxt("parkinsons.csv", delimiter=",") # ok
#ds = np.loadtxt("w-breast-cancer.csv", delimiter=",")
#ds = np.loadtxt("tae.csv", delimiter=",")
#ds = np.loadtxt("ecoli.csv", delimiter=",")
#ds = np.loadtxt("habberman.csv", delimiter=",") # ok
#ds = np.loadtxt("heart-stat.csv", delimiter=",")
os.system('cls' if os.name == 'nt' else 'clear')
print '    __    _                         ________                _ _____          '
print '   / /   (_)___  ___  ____ _______ / ____/ /___ ___________(_) __(_)__  _____'
print '  / /   / / __ \/ _ \/ __ `/ ___(_) /   / / __ `/ ___/ ___/ / /_/ / _ \/ ___/'
print ' / /___/ / / / /  __/ /_/ / /  _ / /___/ / /_/ (__  |__  ) / __/ /  __/ /    '
print '/_____/_/_/ /_/\___/\__,_/_/  ( )\____/_/\__,_/____/____/_/_/ /_/\___/_/     '
print '                              |/                                             '
print ''
print '============================================================================='
print ''

_datasets_ = []
print 'asdasdasd',len(sys.argv)
if len(sys.argv) == 1:
    iris = datasets.load_iris()
    Y = iris.target
    ds = np.column_stack((iris.data, Y))
    _datasets_ += [ { 'name': 'Iris Plant', 'data': ds } ]
else :
    print 'asdasdasdass'
    if sys.argv[1] == 'all':
        for file in os.listdir('./datasets/'):
            if file.endswith('.csv'):
                ds = np.loadtxt( file, delimiter=",")
                _datasets_ += [ { 'name': file , 'data': ds } ]
    else:
        ds = np.loadtxt( sys.argv[1], delimiter=",")
        _datasets_ += [ { 'name': sys.argv[1] , 'data': ds } ]

for _ds_ in _datasets_:
    print _ds_[ 'name' ]
    ds = _ds_[ 'data' ]
    m = len( ds[0] ) - 1                # size of pattern (ej. iris = 4)
    c = len( np.unique( ds[ :, m ] ) )  # number of classes
    N = len( ds )                       # size of dataset

    bar = DownloadProgressBarShady( max = N )
    bar.suffix = ''
    #for i in bar.iter( xrange( N ) ):
		# some progress
	#	time.sleep(.1)

    correct = 0
    loo = LeaveOneOut( N )
    for train_index, test_index in loo:

        train_ds = ds[ train_index ]
        test_ds = ds[ test_index ]

        test = np.array( test_ds[0, 0:m] )
        tclass = test_ds[0, m]

        # TRAIN STAGE
        memorias = get_memories_from_centroids( train_ds, c, m )
        coeficientes = get_coeficientes( train_ds, memorias )

        # TEST STAGE
        distances = get_distances( coeficientes, memorias, test )
        minclass = np.argmin( distances )

        #print tclass, '<->', distances

        if minclass == tclass:
            correct += 1

        bar.next(1)

    print '\tPerformance: %3.4f' %  float((correct/float(N)))

print ''
