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
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cross_validation import LeaveOneOut, train_test_split
from HoldOut import HoldOut

from ui import DownloadProgressSpinner, DownloadProgressSpinnerMoon, DownloadProgressSpinnerPie, \
    DownloadProgressSpinnerLine, DownloadProgressBarShady, DownloadProgressBarCharging, \
    DownloadProgressBarFillingSquares, DownloadProgressBarFillingCircles


def pca(X, num_components=0):
    [n, d] = X.shape
    if (num_components <= 0) or (num_components > n):
        num_components = n
    mu = X.mean(axis=0)
    X = X - mu
    if n > d:
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in xrange(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # select only num_components
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:, 0:num_components].copy()
    return [eigenvalues, eigenvectors, mu]


def get_memories_from_centroids(ds, num_clases, pattern_size, mean_function, test):
    memorias = []
    for i in range(num_clases):
        print 'Clase: ', i
        clase = np.array([e[0:pattern_size] for e in ds if e[pattern_size] == i])
        mean = mean_function(clase, axis=0)
        print 'Variation: ', mean
        mean[np.isnan(mean)] = 0
        searchTree = scipy.spatial.cKDTree(np.copy(clase), leafsize=100)
        dummy, nearest = searchTree.query(test[0:pattern_size], k=pattern_size, p=pattern_size)
        if len(clase) < pattern_size:
            _pca = pca(clase[:, 0:pattern_size], num_components=pattern_size)
            memoria = _pca[1]
        elif len(clase) == len(nearest):
            memoria = clase
        else:
            memoria = np.array(clase[nearest])
        print memoria
        # memoria = memoria / np.linalg.norm(memoria)
        # memoria = scipy.linalg.tan(memoria)
        #
        memorias += [memoria]

    return memorias


def get_coeficientes(ds, memories):
    m = len(ds[0]) - 1
    c = len(memories)
    coeficientes = {}

    for i in range(c):
        coeficientes[str(i)] = []

    for i in range(len(ds)):
        test = np.array(ds[i, 0:m])
        tclass = ds[i, m]
        cc = []
        for j in range(c):
            try:
                print 'solving'
                print test
                print np.linalg.solve(np.transpose(memories[int(j)]), test)
                print '-' * 30
                cc = np.concatenate((cc, np.linalg.solve(np.transpose(memories[int(j)]), test)))
            except:
                cc = np.concatenate((cc, np.zeros(m)))
                pass
            # print 'cc:', len(cc)
        coeficientes[str(int(tclass))] += [cc]

    return coeficientes


def get_distances(coeficientes, memories, test):
    c = len(memories)
    ctest = []
    for j in range(c):
        try:
            ctest = np.concatenate((ctest, np.linalg.solve(np.transpose(memories[j]), test)))
        except:
            ctest = np.concatenate((ctest, np.zeros(len(test))))  # m = len( test )

    distances = []
    for clase in range(c):
        coeficiente = coeficientes[str(clase)]
        dists = []
        for coef in coeficiente:
            distances += [(scipy.spatial.distance.cdist([ctest], [coef], 'cityblock')[0][0], clase)]

    dtype = [('dist', float), ('clase', int)]
    _dist = np.array(distances, dtype=dtype)
    _dist = np.sort(_dist, order='dist')

    dist = np.zeros(c)
    for _d in _dist[0:1]:
        dist[_d['clase']] += 1

    print 'Test: ', test
    print 'CTest: ', ctest
    print 'dist: ', _dist
    return dist


def linear_classifier(_ds_, mean_function, show_progress=False):
    # print '\r\x1b[',str(x),';10H'
    if show_progress:
        print _ds_['name']
    ds = _ds_['data']
    m = len(ds[0]) - 1  # size of pattern (ej. iris = 4)
    c = len(np.unique(ds[:, m]))  # number of classes
    N = len(ds)  # size of dataset

    correct = 0

    loo = LeaveOneOut(N)
    total = N

    if show_progress:
        bar = DownloadProgressBarShady(max=total)
        bar.suffix = ''

    _mean_functions_ = [stats.variation, stats.gmean, np.mean]
    print loo
    it = 0
    # for train_index, test_index in loo:
    for ko in range(1):
        train_ds = ds  # todo: ds[train_index]
        test_ds = ds  # [ test_index ]
        print 'LOO, it: ', it
        memorias = get_memories_from_centroids(train_ds, c, m, mean_function, test_ds[0])
        coeficientes = get_coeficientes(train_ds, memorias)
        print 'Coeficientes', coeficientes

        for _test_ in test_ds:
            test = np.array(_test_[0:m])
            tclass = _test_[m]
            # TEST STAGE
            distances = get_distances(coeficientes, memorias, test)
            print 'Distances: ', distances
            minclass = np.argmax(distances)

            if minclass == tclass:
                correct += 1

            if show_progress:
                bar.next(1)
        it += 1

    performance = float((correct / float(total)))
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
        _datasets_ += [{'name': 'Iris Plant', 'data': ds}]
    else:
        if sys.argv[1] == 'all':
            for file in os.listdir('./datasets/'):
                if file.endswith('.csv'):
                    ds = np.genfromtxt('./datasets/' + file, delimiter=",", filling_values=0)
                    ds[np.isnan(ds)] = 0
                    _datasets_ += [{'name': file, 'data': ds}]
        else:
            ds = np.genfromtxt(sys.argv[1], delimiter=",", filling_values=0)
            ds[np.isnan(ds)] = 0
            _datasets_ += [{'name': sys.argv[1], 'data': ds}]

    for _ds_ in _datasets_:
        p1 = Process(target=linear_classifier, args=(_ds_, stats.variation, True))
        p1.start()
        p1.join()

    print ''
