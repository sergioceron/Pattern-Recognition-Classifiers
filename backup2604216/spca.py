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

def pca(X) :
    [n, d] = X.shape
    num_components = d-1
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

    for i in range( len( test_ds ) ):
        test = np.array( test_ds[i, 0:m] )
        tclass = test_ds[i, m]

        distances = []
        for j in range( c ):
            x = test.T
            A = memorias[ j ]
			#print A
			#print A.shape, np.linalg.inv( np.dot( A.T, A ) ).shape
            P = np.dot( np.dot( A, np.linalg.inv( np.dot( A.T, A ) ) ), A.T )
            x_ = np.dot( A.T, x )
            #x__ = np.subtract( x, x_ )
            #dist = np.linalg.norm( x__ )**2 / np.linalg.norm( x )**2
            #dist = np.dot( x.T, x_ )
            distances += [ np.linalg.norm( x_ )**2 ]

        minclass = np.argmin( distances )

        print tclass, '<->', distances, ' :: ', (minclass==tclass)

        if minclass == tclass:
            correct += 1

print 'Final Performance: %3.4f' %  float((correct/float(N)))
