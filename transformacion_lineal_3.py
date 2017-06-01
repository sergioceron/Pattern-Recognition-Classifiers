import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.spatial

ds = np.genfromtxt('./datasets/iris2d.csv', delimiter=",", filling_values=0)

def scatter( data, indx ):
    plt.figure(indx)
    markers = [ 'o', 'v', 'x' ]
    for k in range( 0, 3 ): 
        x = []
        y = []
        clase = np.array([e for e in data if e[2] == k])
        for patron in clase:
            x += [ patron[0] ]
            y += [ patron[1] ]
        plt.scatter( x, y, marker=markers[k] )

def getA( v1, v2, Tv1, Tv2 ):
    # Paso 1
    B = [ v1, v2 ]
    # Paso 2
    d1 = np.linalg.solve( np.transpose( B ), Tv1 )
    d2 = np.linalg.solve( np.transpose( B ), Tv2 )
    # Paso 3
    D = np.transpose( [ d1, d2 ] )
    # Paso 4
    C = np.transpose( B )
    # Paso 5
    C_inv = np.linalg.inv( C )
    # Paso 6
    CD = np.dot( C, D )
    A = np.dot( CD, C_inv )
    return A

def collinear(a, b):
    for i in range(len(a)):
        if a[i] == b[i]:
            return True
    return False

for it in range( 20 ):
    scatter( ds, it )

    means = []
    clases = []
    for k in range( 0, 3 ):
        clase = np.array([e[0:2] for e in ds if e[2] == k])
        means += [ clase.mean( axis = 0 ) ]
        clases += [ clase ]

    clase_pivote = it%3
    searchTree = scipy.spatial.cKDTree( np.copy(clases[clase_pivote]), leafsize=100 )
    dummy, nearest = searchTree.query( means[clase_pivote], k = len( clases[clase_pivote] ), p = len( clases[clase_pivote] ) )
    pivote = clases[clase_pivote][nearest[-1]]
    clase_estatico = [x for x in xrange(3) if x != clase_pivote][0]
    estatico = means[clase_estatico]

    print pivote,  clase_pivote
    print estatico,  clase_estatico
    print '-'*30

    v1 = pivote
    v2 = estatico

    Tv1 = means[clase_pivote][0:2]
    Tv2 = estatico
    A = getA( v1, v2, Tv1, Tv2 )

    #plt.scatter( [v1[0]], [v1[1]], s=[100] )
    #plt.scatter( [v2[0]], [v2[1]], s=[100] )

    # transformamos todos los vectores
    ds_transformado = []
    for patron in ds:
        vector = patron[0:2]
        vector_t = np.dot( A, vector )
        patron_t = np.concatenate((vector_t, [patron[2]]))
        ds_transformado += [ patron_t ]

    ds = ds_transformado

plt.show()
