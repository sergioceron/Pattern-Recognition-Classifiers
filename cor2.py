import os
from multiprocessing import Process
import time
import numpy as np
import numpy.random as rnd
from scipy import stats
import scipy
import urllib
import sys
import sklearn
import math
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut, KFold
import matlab_wrapper
from scipy.fftpack import dct

from ui import DownloadProgressSpinner, DownloadProgressSpinnerMoon, DownloadProgressSpinnerPie, DownloadProgressSpinnerLine, DownloadProgressBarShady, DownloadProgressBarCharging, DownloadProgressBarFillingSquares, DownloadProgressBarFillingCircles

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

def get_clase(clases, patron ):

	maximos = []
	for clase in clases:
		valores = []
		for p in clase:
			valores += [ corr2(p, patron) ]

		maximos += [max(valores)]

	#print maximos
	return maximos



def linear_classifier(_ds_, mean_function, show_progress=False ):
	#print '\r\x1b[',str(x),';10H'
	if show_progress:
		print _ds_[ 'name' ]
	ds = _ds_[ 'data' ]
	m = len( ds[0] ) - 1                # size of pattern (ej. iris = 4)
	c = len( np.unique( ds[ :, m ] ) )  # number of classes
	N = len( ds )                       # size of dataset

	correct = 0

	loo = LeaveOneOut( N )
	total = N

	if show_progress:
		bar = DownloadProgressBarShady( max = total )
		bar.suffix = ''

	index = 0
	for train_index, test_index in loo:
		train_ds = ds[ train_index ]
		test_ds = ds[ test_index ]

		clases = []
		for i in range( c ):
			clase = np.array( [ dct(e[0:m]) for e in train_ds if e[m] == i] )
			clases += [ clase ]


		for _test_ in test_ds:
			test = np.array( _test_[0:m] )
			tclass = _test_[m]
			# TEST STAGE
			distances = get_clase( clases, test )
			print index, distances
			minclass = np.argmax( distances )

			if minclass == tclass:
				correct += 1

			if show_progress:
				bar.next(1)

		index += 1
	performance = float((correct/float(total)))
	if show_progress:
		print '\tPerformance: %3.4f' % performance

	return performance

if __name__ == '__main__':
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

    if len(sys.argv) == 1:
        iris = datasets.load_iris()
        Y = iris.target
        ds = np.column_stack((iris.data, Y))
        _datasets_ += [ { 'name': 'Iris Plant', 'data': ds } ]
    else :
        if sys.argv[1] == 'all':
            for file in os.listdir('./datasets/'):
                if file.endswith('.csv'):
                    ds = np.genfromtxt( './datasets/'+file, delimiter=",", filling_values=0 )
                    ds[ np.isnan( ds ) ] = 0
                    _datasets_ += [ { 'name': file , 'data': ds } ]
        else:
            ds = np.genfromtxt( sys.argv[1], delimiter=",", filling_values=0 )
            ds[ np.isnan( ds ) ] = 0
            _datasets_ += [ { 'name': sys.argv[1] , 'data': ds } ]

    for _ds_ in _datasets_:
		p1 = Process( target = linear_classifier, args = ( _ds_, stats.variation, True ) )
		p1.start()
		p1.join()

    print ''
