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

def transformar( A, data ):
    ds_transformado = []
    for patron in data:
        vector = patron[0:2]
        vector_t = np.dot( A, vector )
        patron_t = np.concatenate((vector_t, [patron[2]]))
        ds_transformado += [ patron_t ]

    return ds_transformado

def collinear(a, b):
    for i in range(len(a)):
        if a[i] == b[i]:
            return True
    return False

memorias = []
scatter( ds, 0 )

new_ds = []

for k in range( 0, 3 ):
    memoria_clase = []
    clase_full = np.array([e for e in ds if e[-1] == k])
    for it in range(100):
        clase = clase_full[:,:-1]
        mean = clase.mean( axis = 0 )
        searchTree = scipy.spatial.cKDTree( np.copy(clase), leafsize=100 )
        dummy, nearest = searchTree.query( mean, k = len( clase ), p = len( clase ) )
        pivote = clase[nearest[0]]
        pivote2 = clase[nearest[-1]]

        v1 = pivote
        v2 = mean

        Tv1 = mean
        Tv2 = mean

        try :
            A = getA( v1, v2, Tv1, Tv2 )
            clase_full = np.array( transformar( A, clase_full ) )
            memoria_clase += [ A ]
        except:
            pass
            #break

    new_ds += clase_full.tolist()
    memorias += [ memoria_clase ]

print new_ds
ds = new_ds
scatter( ds, 1 )

print 'CLASIFICATION'
#plt.figure( 10 )
markers = [ 'o', 'v', 'x' ]
w = 0
plt.scatter( [4.9], [1.8], marker=markers[2], s=[100] ) # 6.7, 2
for m_clase in memorias :
    print 'Probando con:', w, len(m_clase)
    prueba=[[4.9,1.8,2]]
    for M in m_clase :
        print [prueba[0][0]], [prueba[0][1]]
        print M
        prueba = transformar( M, prueba )
    print [prueba[0][0]], [prueba[0][1]]
    plt.scatter( [prueba[0][0]], [prueba[0][1]], marker=markers[w], s=[100] )
    w += 1
    print '-'*30

plt.show()
