import sys
import numpy as np
import scipy
from scipy import stats
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut, train_test_split

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

# MAIN CODE PROGRAM

if len(sys.argv) == 1:
	iris = datasets.load_iris()
	Y = iris.target
	ds = np.column_stack((iris.data, Y))
	print 'iris: ',
else :
	ds = np.loadtxt( sys.argv[1], delimiter=",")
	print sys.argv[1], ': ',

m = len( ds[0] ) - 1                # size of pattern (ej. iris = 4)
c = len( np.unique( ds[ :, m ] ) )  # number of classes
N = len( ds )                       # size of dataset
correct = 0
loo = LeaveOneOut( N )
for train_index, test_index in loo:
	train_ds = ds[ train_index ]
	test_ds = ds[ test_index ]

	test = np.array( test_ds[0, 0:m] )
	tclass = test_ds[0, m]

	X = train_ds[:, 0:m]
	Y = train_ds[:, m]
	[D, W, mu] = pca( X, Y, m )
	projections = []
	for xi in X:
		projections += [ project( W, xi, mu ) ]

	minDist = np.finfo('float').max
	minclass = -1
	Q = project( W, test, mu ) # aqui estaba el pedo
	for i in xrange( len ( projections ) ):
		dist = scipy.spatial.distance.cdist([projections[i]], [Q], 'euclidean')[0][0]
		if dist < minDist:
			minDist = dist
			minclass = Y[i]

	print tclass, '<->', minclass, '::', (minclass == tclass)

	if minclass == tclass:
		correct += 1

print '%3.4f' %  float((correct/float(N)))
