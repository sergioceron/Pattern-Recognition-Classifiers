import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.spatial
import scipy.stats
from sklearn.cross_validation import LeaveOneOut

o_ds = np.genfromtxt('./datasets/iris2d.csv', delimiter=",", filling_values=0)

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

def scatter2( data ):
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

scatter( o_ds, 0 )

loo = LeaveOneOut( len(o_ds) )
for train_index, test_index in loo:
    ds = o_ds[train_index]
    test = o_ds[test_index][0]

    memorias = []

    new_ds = []
    si = 1
    for k in range( 0, 3 ):
        memoria_clase = []
        clase_full = np.array([e for e in ds if e[-1] == k])
        for it in range( len( clase_full ) ):
            clase = clase_full[:,:-1]
            mean = clase.mean( axis = 0 )
            pivote = clase[it]

            v1 = pivote
            v2 = mean

            Tv1 = mean
            Tv2 = mean

            try :
                A = getA( v1, v2, Tv1, Tv2 )
                clase_tmp = np.array( transformar( A, clase_full ) )
                var_before = scipy.stats.nanmedian( scipy.stats.variation( clase_full, axis=0 ) )
                var_after = scipy.stats.nanmedian( scipy.stats.variation( clase_tmp, axis=0 ) )
                #print var_before, var_after, abs(var_before - var_after)
                if abs(var_before - var_after) < .1 :
                    memoria_clase += [ A ]
                    clase_full = clase_tmp
                    '''
                    scatter( clase_full, si )
                    for h in range(3):
                        if h != k:
                            other = np.array([e for e in ds if e[-1] == h])
                            scatter2( other )
                    si += 1
                    '''
            except:
                pass

        new_ds += clase_full.tolist()
        memorias += [ memoria_clase ]

    ds = new_ds
    #scatter( ds, si )
    print 'Testing: ', test[0:2], '->', test[2]
    markers = [ 'o', 'v', 'x' ]
    w = 0
    #plt.scatter( test[0], test[1], marker=markers[int(test[2])], s=[100] ) # 4.6, 1.4, 1
    res = []
    for m_clase in memorias :
        print 'Clase:', w, ', Transformaciones: ', len(m_clase),
        prueba=[test]
        for M in m_clase :
            #print [prueba[0][0]], [prueba[0][1]]
            prueba = transformar( M, prueba )

        clase_actual = np.array([e for e in ds if e[-1] == w])
        a = prueba[0][0:2]
        b = np.mean(clase_actual, axis=0)[0:2]
        searchTree = scipy.spatial.cKDTree(np.copy(clase_actual[:,0:2]), leafsize=100)
        dummy, nearest = searchTree.query(a, k=2, p=2)
        d = clase_actual[nearest[0]][0:2]
        x = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        y = scipy.spatial.distance.cdist([a], [d], 'cityblock')[0][0]
        z = ((1-x) - y)
        print ', Angulo: ', x, ', Distancia: ', y, ', Resultado: ', z
        res += [ abs(z) ]
        #plt.scatter( [prueba[0][0]], [prueba[0][1]], marker=markers[w], s=[100] )
        w += 1
    print 'Clase: ', np.argmin( res )
    print '-'*30

#plt.show()
