# YA NADAMAS TRATAR DE REDUCIR EL NUMERO DE COEFICIENTES PARA OPTIMIZAR EL ALGORITMO,
# EN VEZ DEL CENTROIDE DEBE SER UNA BASE DE LOS N MAS CERCANOS AL CENTROIDE
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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_memories_from_centroids( clases, pattern_size ):
    memorias = []
    for i in range( len( clases ) ):
        clase = clases[i]
        mean = scipy.stats.variation( clase, axis = 0 )
        searchTree = scipy.spatial.cKDTree( np.copy( clase ), leafsize = 100 )
        dummy, nearest = searchTree.query( mean, k = pattern_size, p = pattern_size )
        memoria = clase[ nearest ]
        #print 'Clase ', i,' det A: ', np.linalg.det( memoria ), ', det B: ', np.linalg.det( memoria.T )
        if np.linalg.det( memoria ) == 0: # NO SON L.I. => NO SON UNA BASE
            pca = PCA( n_components = pattern_size )
            pca.fit( clase )
            memoria = pca.components_.T
            print 'Clase ', i,' det A: ', np.linalg.det( memoria ), ', det B: ', np.linalg.det( memoria.T )

        memorias += [ memoria ]

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
                cc += [ np.linalg.solve( np.transpose( memories[ int( j ) ] ), test ) ]
            except:
                cc += [ np.zeros( m ) ]
                pass

        coeficientes[ str( int( tclass ) ) ] += [ cc ]

    return coeficientes


def get_distances( coeficientes, memories, test ):
    c = len( memories )

    ctest = []
    for j in range( c ):
        try :
            ctest += [ np.linalg.solve( np.transpose( memories[ j ] ), test ) ]
        except:
            ctest += [ np.zeros( len( test ) ) ]

    distances = []
    for clase in range( c ):
        coeficiente = coeficientes[ str( clase ) ]

        dists = []
        for coef in coeficiente:
            dd = 0
            for w in range( c ):
                if w == clase:
                    dd += scipy.spatial.distance.cdist( [ctest[w]], [coef[w]], 'cityblock')[0][0]
            dists += [ dd ]

        distances += [ min( dists ) ]

    return distances

def colors(z):
    mapa = ['red', 'green', 'blue']
    colors = []
    for _z in z:
        colors += [ mapa[ int(_z) ] ]
    return colors

# MAIN CODE PROGRAM

iris = datasets.load_iris()
Y = iris.target
#raw_data = urllib.urlopen( "file:///Users/sergio/Downloads/glass.csv" )
#raw_data = urllib.urlopen( "http://goo.gl/j0Rvxq" )
# load the CSV file as a numpy matrix
#ds = np.column_stack((iris.data, Y))
ds = np.loadtxt("iris2d.csv", delimiter=",") # ok
#ds = np.loadtxt("mammographic_filled.csv", delimiter=",") # ok
#ds = np.loadtxt("wine.csv", delimiter=",") # ok
#ds = np.loadtxt("parkinsons.csv", delimiter=",") # ok
#ds = np.loadtxt("w-breast-cancer.csv", delimiter=",")
#ds = np.loadtxt("tae.csv", delimiter=",")
#ds = np.loadtxt("ecoli.csv", delimiter=",")
#ds = np.loadtxt("habberman.csv", delimiter=",") # ok
#ds = np.loadtxt("heart-stat.csv", delimiter=",")

m = len( ds[0] ) - 1                # size of pattern (ej. iris = 4)
c = len( np.unique( ds[ :, m ] ) )  # number of classes
N = len( ds )                       # size of dataset

# TRAINING PHASE FIRST STAGE (SUPPORT PATTERNS)
correct = 0
loo = LeaveOneOut( N )
itera = 0
for train_index, test_index in loo:
    train_ds = ds[ train_index ]
    test_ds = ds[ test_index ]

    test = np.array( test_ds[0, 0:m] )
    tclass = test_ds[0, m]

    # ====================== SE CALCULAN LAS MEMORIAS ======================
    clases = []
    means = []
    for i in range( c ):
        clase = np.array( [e[0:m] for e in ds if e[m] == i] )
        clases += [ clase ]
        means += [ clase.mean( axis = 0 ) ]

    memorias = get_memories_from_centroids( clases, m )

    coeficientes = get_coeficientes( train_ds, memorias )
    print coeficientes
    plt.figure(0)
    plt.scatter( train_ds[:, 0], train_ds[:, 1], c = colors(train_ds[:, 2]) )

    markers = [ 'o', 'v', 'x' ]
    for mem in range( c ):
        x = []
        y = []
        z = []
        mark = []
        for k in range( c ): # este for solo esta por que separo los coeficientes por clase
            coef = coeficientes[str(k)]
            for _c in coef:
                #_c[k][0] += means[k][0]
                #_c[k][1] += means[k][1]
                x += [ _c[mem][0] ]
                y += [ _c[mem][1] ]
                z += [ k ]
                mark += [ markers[k] ]
        plt.figure(mem+1)
        plt.scatter( x, y, c = colors(z) )


    plt.show()

    distances = get_distances( coeficientes, memorias, test )

    minclass = np.argmin( distances )

    print 'i: ', itera, ', res: ', tclass, '<->', distances, '::', (minclass == tclass)
    itera += 1
    if minclass == tclass:
        correct += 1

    break

print 'Final Performance: %3.4f' %  float((correct/float(N)))
