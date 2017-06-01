# EN ESTE ARCHIVO SE TRATO DE OBTENER EL MINIMAL SPANING SET USANDO LA FORMA ECHELON DE LA MATRIZ
# ESTA FORMA ECHELON ES LA MATRIZ ESCALONADA Y DONDE ESTA EL PIVOTE, AHI ESTAN LOS VECTORES QUE FORMAN EL MSS
# NO SE OBTUVIERON LOS RESULTADOS ESPERADOS, DE HECHO NI SIQUIERA SE CALCULO EL PERFORMANCE
# YA QUE SE VIO QUE EN LOS BANCOS DE DATOS, EL PRIMERO Y SEGUNDO PATRON SON LOS PIVOTES Y POR TANTO, SON EL MSS
# ESTO ULTIMO INDICA QUE NO HAY MEJORA
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy import stats
import scipy
import sympy
import urllib
import sys
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut, train_test_split
from pyswarm import pso
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_memory( ds, pattern_size, multiplicador ):
    _k=pattern_size*multiplicador
    mean = scipy.stats.variation( ds, axis = 0 )
    searchTree = scipy.spatial.cKDTree( np.copy( ds ), leafsize = 100 )
    dummy, nearest = searchTree.query( mean, k = _k, p = _k )
    memoria = ds[ nearest[-pattern_size:] ]
    return [ memoria ]


def get_coeficientes( ds, memories, c ):
    coeficientes = {}

    for i in range( c ):
        coeficientes[ str( i ) ] = []

    for i in range( len( ds ) ):
        test = np.array( ds[i, 0:m] )
        tclass = ds[i, m]
        cc = []
        for j in range( len( memories ) ):
            try:
                cc += [ np.linalg.solve( np.transpose( memories[ int( j ) ] ), test ) ]
            except:
                cc += [ np.zeros( m ) ]
                pass

        #print tclass
        coeficientes[ str( int( tclass ) ) ] += [ cc ]

    return coeficientes

def translate( ds, centroide ):
    ds_copy = []
    for i in range( len( ds ) ):
        ds_copy += [ np.subtract( ds[i], centroide ) ]
    return np.array( ds_copy )


def get_distances( coeficientes, memories, test, c ):

    ctest = []
    for j in range( len(memories) ):
        try :
            ctest += [ np.linalg.solve( np.transpose( memories[ j ] ), test ) ]
        except:
            ctest += [ np.zeros( len( test ) ) ]

    distances = []
    for clase in range( c ):
        coeficiente = coeficientes[ str( clase ) ]

        dists = []
        for coef in coeficiente:
            distances += [ (scipy.spatial.distance.cdist( [ctest[0]], [coef[0]], 'cityblock' )[0][0], clase ) ]
        #distances += [ min( dists ) ]

	dtype = [('dist', float), ('clase', int)]
	_dist = np.array( distances, dtype = dtype )
	_dist = np.sort( _dist, order = 'dist' )

	dist = np.zeros( c )
	for _d in _dist[0:1]:
		dist[ _d['clase'] ] += 1

    return dist

def colors(z):
    mapa = ['red', 'green', 'blue']
    colors = []
    for _z in z:
        colors += [ mapa[ int(_z) ] ]
    return colors

def J(p, *args):
    ds, c, m, N = args

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

    for i in range( 3 ):
        clase = np.array( [e[0:m] for e in train_ds if e[m] == i] )
        print 'RRef ', i, ': '
        print sympy.Matrix(clase.T).rref()

    centroide = np.concatenate( [ stats.variation( train_ds[:, 0:m], axis=0 ) , [0] ])
    #train_ds = translate( train_ds, centroide )

    test = np.array( test_ds[0, 0:m] )
    #test = np.subtract( test, centroide[0:m] )

    tclass = test_ds[0, m]

    # ====================== SE CALCULAN LAS MEMORIAS ======================
    plt.figure(0)
    plt.scatter( train_ds[:, 0], train_ds[:, 1], c = colors(train_ds[:, -1]) )

    for mult in range(1,2):
        memorias = get_memory( train_ds[:, 0:m], m, mult )
        coeficientes = get_coeficientes( train_ds, memorias, c )

        markers = [ 'o', 'v', 'x' ]
        plt.figure(mult)
        for k in range( c ):
            x = []
            y = []
            z = []
            coef = coeficientes[str(k)]
            for _c in coef:
                x += [ _c[0][0] ]
                y += [ _c[0][1] ]
                z += [ k ]
            plt.scatter( x, y, c = colors(z), marker=markers[k] )

    plt.show()
    break

    distances = get_distances( coeficientes, memorias, test, c )

    minclass = np.argmax( distances )

    print 'i: ', itera, ', res: ', tclass, '<->', minclass, '::', distances, '::', (minclass == tclass)
    itera += 1
    if minclass == tclass:
        correct += 1

    #break

print 'Final Performance: %3.4f' %  float((correct/float(N)))
