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

def lda(X, y, num_components=0):
	y = np.asarray(y)
	[n,d] = X.shape
	c = np.unique(y)
	if (num_components <= 0) or (num_component>(len(c)-1)):
		num_components = (len(c)-1)
	meanTotal = X.mean(axis=0)
	Sw = np.zeros((d, d), dtype=np.float32)
	Sb = np.zeros((d, d), dtype=np.float32)
	for i in c:
		Xi = X[np.where(y==i)[0],:]
		meanClass = Xi.mean(axis=0)
		Sw = Sw + np.dot((Xi-meanClass).T, (Xi-meanClass))
		Sb = Sb + n * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal))
	eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
	idx = np.argsort(-eigenvalues.real)
	eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]
	eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
	eigenvectors = np.array(eigenvectors[0:,0:num_components].real, dtype=np.float32, copy=True)
	return [eigenvalues, eigenvectors]


def pca(X) :
    [n, d] = X.shape
    num_components = d
    mu = X.mean(axis=0)
    X = X - mu # LOS DATOS TIENEN QUE ESTAR CENTRADOS
    if n>d: # SI EL NUMERO DE PATRONES ES MAYOR AL NUMERO DE RASGOS (LO MAS COMUN)
        C = np.dot(X.T, X) # PRODUCTO CRUZ
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T,eigenvectors)
        for i in xrange(n):
            eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])

	idx = np.argsort(-eigenvalues) # SE ORDENAN DE MAYOR A MENOR EIGENVALOR
	eigenvalues = eigenvalues[idx] # SE OBTIENEN LOS EIGENVALORES EN EL ORDEN PREVIAMENTE SELECCIONADO
	eigenvectors = eigenvectors[:,idx] # SE OBTIENEN LOS EIGENVECTORES EN EL ORDEN PREVIAMENTE SELECCIONADO

    # select only num_components
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:,0:num_components].copy()
    return [eigenvalues, eigenvectors]

def get_memories_from_eigenvectors( ds, num_clases, pattern_size ):
    memorias = []
    for i in range( num_clases ):
        classe = np.array( np.array( [e[0:pattern_size] for e in ds if e[pattern_size] == i] ) )
        memorias  += [ pca( classe )[1] ]
    return memorias

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
            dists += [ scipy.spatial.distance.cdist([ctest], [coef], 'euclidean')[0][0] ]
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
#ds = np.loadtxt("mammographic_filled.csv", delimiter=",") # ok
#ds = np.loadtxt("wine.csv", delimiter=",") # ok
#ds = np.loadtxt("parkinsons.csv", delimiter=",") # ok
#ds = np.loadtxt("w-breast-cancer.csv", delimiter=",") # .9599 with pca
#ds = np.loadtxt("tae.csv", delimiter=",")
#ds = np.loadtxt("ecoli.csv", delimiter=",") # .7916 not suitable for centroids
#ds = np.loadtxt("habberman.csv", delimiter=",") # ok
#ds = np.loadtxt("heart-stat.csv", delimiter=",")

m = len( ds[0] ) - 1                # size of pattern (ej. iris = 4)
c = len( np.unique( ds[ :, m ] ) )  # number of classes
N = len( ds )                       # size of dataset


# TRAINING PHASE FIRST STAGE (SUPPORT PATTERNS)
correct = 0
loo = LeaveOneOut( N )
for train_index, test_index in loo:
    train_ds = ds[ train_index ]
    test_ds = ds[ test_index ]

    memorias = get_memories_from_eigenvectors( train_ds, c, m )
    #memorias = get_memories_from_centroids( train_ds, c, m )
    centroid = train_ds.mean(axis=0)
    centroid[m] = 0

    coeficientes = get_coeficientes( train_ds, memorias )

    for i in range( len( test_ds ) ):
        #test_t = np.subtract( test_ds[i], centroid )
        #test = np.array( test_t[0:m] )
        #tclass = test_t[m]

        test = np.array( test_ds[i, 0:m] )
        tclass = test_ds[i, m]

        distances = get_distances( coeficientes, memorias, test )

        minclass = np.argmin( distances )

        print tclass, '<->', distances, ' :: ', (minclass==tclass)

        if minclass == tclass:
            correct += 1

print 'Final Performance: %3.4f' %  float((correct/float(N)))
