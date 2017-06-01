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
from sklearn.decomposition import PCA, FastICA

def asRowMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((0, X[0].size), dtype=X[0].dtype)
	for row in X:
		mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
	return mat

def project(W, X, mu=None):
	if mu is None:
		return np.dot(X,W)
	return np.dot(X - mu, W)

def reconstruct(W, Y, mu=None):
	if mu is None:
		return np.dot(Y,W.T)
	return np.dot(Y, W.T) + mu

def pca(X, y, num_components=0):
	[n,d] = X.shape
	if (num_components <= 0) or (num_components>n):
		num_components = n
	mu = X.mean(axis=0)
	X = X - mu
	if n>d:
		C = np.dot(X.T,X)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
	else:
		C = np.dot(X,X.T)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
		eigenvectors = np.dot(X.T,eigenvectors)
		for i in xrange(n):
			eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])

	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]
	# select only num_components
	eigenvalues = eigenvalues[0:num_components].copy()
	eigenvectors = eigenvectors[:,0:num_components].copy()
	return [eigenvalues, eigenvectors, mu]

def lda(X, y, num_components=0):
	y = np.asarray(y)
	[n,d] = X.shape
	c = np.unique(y)
	if (num_components <= 0) or (num_components>(len(c)-1)):
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

def fisherfaces(X,y,num_components=0):
    y = np.asarray(y)
    [n,d] = X.shape
    c = len(np.unique(y))
    [eigenvalues_pca, eigenvectors_pca, mu_pca] = pca(X, y, c)
    #print 'PCA: ', eigenvectors_pca
    [eigenvalues_lda, eigenvectors_lda] = lda(project(eigenvectors_pca, X, mu_pca), y, num_components)
    #print 'LDA: ', eigenvectors_lda
    eigenvectors = np.dot(eigenvectors_pca, eigenvectors_lda)
    #print 'Eigenvectors: ', eigenvectors
    return [eigenvalues_lda, eigenvectors, mu_pca]

def get_memories_from_centroids( clases, pattern_size ):
    memorias = []
    for i in range( len( clases ) ):
        clase = clases[ i ]
        #mean = scipy.stats.variation( clase, axis = 0 )
        #searchTree = scipy.spatial.cKDTree( np.copy( clase ), leafsize = 100 )
        #dummy, nearest = searchTree.query( mean, k = pattern_size, p = pattern_size )
        #memoria = clase[ nearest ]

        pca = PCA( n_components = pattern_size )
        pca.fit( clase )
        memoria = pca.components_.T

        memorias += [ memoria ]

    return memorias


def get_variances( clases, memories, test ):
    c = len( memories )

    variances = []
    for clase in range( c ):
        mean_v = clases[i].mean( axis = 0 )
        test_t = test.copy()
        coeficientes_m = np.linalg.solve( memories[ clase ], mean_v )
        coeficientes_t = np.linalg.solve( memories[ clase ], test_t )
        print coeficientes_m, ' <-> ', coeficientes_t
        variances += [ scipy.spatial.distance.cdist([coeficientes_m], [coeficientes_t], 'euclidean')[0][0] ]

    return variances

# MAIN CODE PROGRAM

iris = datasets.load_iris()
Y = iris.target
#raw_data = urllib.urlopen( "file:///Users/sergio/Downloads/glass.csv" )
#raw_data = urllib.urlopen( "http://goo.gl/j0Rvxq" )
# load the CSV file as a numpy matrix
#ds = np.column_stack((iris.data, Y))
#ds = np.loadtxt("mammographic_filled.csv", delimiter=",") # ok
ds = np.loadtxt("wine.csv", delimiter=",") # ok
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
for train_index, test_index in loo:
    train_ds = ds[ train_index ]
    test_ds = ds[ test_index ]

    test = np.array( test_ds[0, 0:m] )
    tclass = test_ds[0, m]

    # ====================== SE CALCULAN LAS MEMORIAS ======================
    #clases = []
    #for i in range( c ):
    #    clases += [ np.array( [e[0:m] for e in ds if e[m] == i] ) ]
    #memorias = get_memories_from_centroids( clases, m )
    #print memorias
    #variances = get_variances( clases, memorias, test )
    #minclass = np.argmin( variances )

    X = train_ds[:, 0:m]
    Y = train_ds[:, m]
    [D, W, mu] = fisherfaces( X, Y )
    projections = []
    for xi in X:
        projections += [ project( W, xi, mu ) ]

    print W

    minDist = np.finfo('float').max
    minclass = -1
    Q = project( W, test, mu ) # aqui estaba el pedo
    #print Q
    for i in xrange( len ( projections ) ):
        #print 'pi: ', projections[i], ', Q: ', Q
        dist = scipy.spatial.distance.cdist([projections[i]], [Q], 'euclidean')[0][0]
        if dist < minDist:
            minDist = dist
            minclass = Y[i]

    print tclass, '<->', minclass, '::', (minclass == tclass)

    if minclass == tclass:
        correct += 1

print 'Final Performance: %3.4f' %  float((correct/float(N)))
