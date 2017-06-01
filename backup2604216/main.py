import numpy as np
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut

from DIC import DIC


def expandir( datos, size=4 ):
	expandidos = []
	for patron in datos:
		expanded = patron.tolist()
		for w in range( size ):
			for z in range( size ):
				if w == 1 and w != z:
					expanded += [np.power( patron[w], 1 / (patron[z] + 1) )]
		expandidos += [expanded]
	return np.array( expandidos )


iris = datasets.load_iris()
Y = iris.target
ds = np.column_stack( (iris.data, Y) )
ds = np.genfromtxt( '../datasets/tae.csv', delimiter=",", filling_values=0 )

def exapand( ds, size=4 ):
    expandidos = []
    for patron in datos:
        expanded = patron.tolist()
        for w in range( size ):
            for z in range( size ):
                if w==1 and w != z:
                    expanded += [ np.power( patron[w], 1/(patron[z]+1) ) ]
        expandidos += [ expanded ]
    return np.array( expandidos )


def multiclassify( num_classes, train_ds, pattern ):
	for i in range( num_classes - 1 ):
		classifier = DIC( pivot=i, _lambda=.25, plot=True )
		classifier.train( train_ds )
		c = classifier.classify( pattern )
		if c == 0:
			return i
		elif c == 1 and i == num_classes - 2:
			return i + 1
	return -1


def re( debug=False ):
	classifier = DIC( pivot=0 )
	classifier.train( ds )
	test = np.array( [e for e in ds if e[-1] == 0 or e[-1] == 1] )
	correct = 0
	for p in test:
		c = classifier.classify( p )
		if debug: print p, ' -> ', c
		if c == p[-1]:
			correct += 1
	return float( correct ) / float( len( test ) )


def loo( num_classes, debug=False ):
	N = np.size( ds, axis=0 )
	loo = LeaveOneOut( N )
	correct = 0
	for train_index, test_index in loo:
		train_ds = ds[train_index]
		test = ds[test_index][0]
		c = multiclassify( num_classes, train_ds, test )
		if debug: print test, ' -> ', c
		if c == test[-1]:
			correct += 1
		break

	return float( correct ) / float( N )


print 'LOO:', loo( 3, debug=True )
