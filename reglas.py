import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.spatial
import scipy.stats
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut

iris = datasets.load_iris()
Y = iris.target
ds = np.column_stack((iris.data, Y))

o_clase2 = np.array([e[0:4] for e in ds if e[4] == 1])
o_clase3 = np.array([e[0:4] for e in ds if e[4] == 2])

def expandir( datos, size=4 ):
    expandidos = []
    for patron in datos:
        expanded = []
        for w in range( size ):
            for z in range( w+1, size ):
                expanded += [ patron[w] + patron[z] ]
        expandidos += [ expanded ]
    return np.array( expandidos )

def iteracion_general( clase2, clase3, size=4 ):
    max2 = []
    min3 = []
    for i in range( size ) :
        max2 += [ max( clase2[:, i] ) ]
        min3 += [ min( clase3[:, i] ) ]

    C2 = 0
    C3 = 0
    NOclase2 = []
    NOclase3 = []
    for i in range( len( clase2 ) ):
        signs = []
        for j in range( size ):
            signs += [ clase2[i, j] - min3[j] ]

        if min( np.sign( signs ) ) > -1 :
            C2 += 1
            NOclase2 += [ i ]

    for i in range( len( clase3 ) ):
        signs = []
        for j in range( size ):
            signs += [ clase3[i, j] - max2[j] ]

        if max( np.sign( signs ) ) < 1 :
            C3 += 1
            NOclase3 += [ i ]

    print 'Clase 2 (count): ', C2
    print 'Clase 2 (indices): ', NOclase2
    print 'Clase 2 (patrones): ', clase2[ NOclase2 ]

    print 'Clase 3 (count): ', C3
    print 'Clase 3 (indices): ', NOclase3
    print 'Clase 3 (patrones): ', clase3[ NOclase3 ]
    return NOclase2, NOclase3, max2, min3, clase2[ NOclase2 ], clase3[ NOclase3 ]

def iteracion( clase2, clase3 ):
    max2 = []
    min3 = []
    for i in range(4) :
        max2 += [ max( clase2[:, i] ) ]
        min3 += [ min( clase3[:, i] ) ]

    C2 = 0
    C3 = 0
    NOclase2 = []
    NOclase3 = []
    for i in range( len( clase2 ) ):
        if min( np.sign( [ clase2[i, 0]-min3[0], clase2[i, 1]-min3[1], clase2[i, 2]-min3[2], clase2[i, 3]-min3[3] ] ) ) > -1 :
            C2 += 1
            NOclase2 += [ i ]

    for i in range( len( clase3 ) ):
        if max( np.sign( [ clase3[i, 0]-max2[0], clase3[i, 1]-max2[1], clase3[i, 2]-max2[2], clase3[i, 3]-max2[3] ] ) ) < 1 :
            C3 += 1
            NOclase3 += [ i ]

    print 'Clase 2 (count): ', C2
    print 'Clase 2 (indices): ', NOclase2
    print 'Clase 2 (patrones): ', clase2[ NOclase2 ]

    print 'Clase 3 (count): ', C3
    print 'Clase 3 (indices): ', NOclase3
    print 'Clase 3 (patrones): ', clase3[ NOclase3 ]
    return max2, min3, clase2[ NOclase2 ], clase3[ NOclase3 ]

print ' --------------- ITERACION 1 ------------------ '
a,b,max2i1, min3i1, _clase2, _clase3 = iteracion_general( o_clase2, o_clase3 )
print ' --------------- ITERACION 2 ------------------ '
a,b,max2i2, min3i2, _clase2, _clase3 = iteracion_general( _clase2, _clase3 )
print ' --------------- EXPANDIR ------------------ '
print 'Clase 2 original'
print _clase2
print 'Clase 2 expandida'
ec2 = expandir( _clase2 )
print np.row_stack( ( ec2, np.min( ec2, axis=0 ), np.max( ec2, axis=0 ) ) )
print 'Clase 3 original'
print _clase3
print 'Clase 3 expandida'
ec3 = expandir( _clase3 )
print np.row_stack( ( ec3, np.min( ec3, axis=0 ), np.max( ec3, axis=0 ) ) )
print ' --------------- ITERACION 3 ------------------ '
NOclase2, NOclase3, max2i3, min3i3, _clase2a, _clase3a = iteracion_general( ec2, ec3, 6 )
print _clase2[NOclase2]
print _clase3[NOclase3]
print ' --------------- ITERACION 4 ------------------ '
a,b,max2i2, min3i2, _clase2, _clase3 = iteracion_general( _clase2[NOclase2], _clase3[NOclase3] )

'''
print ' --------------- PRUEBA ------------------ '
for i in range( len( o_clase2 ) ):
    prueba = o_clase2[i]
    if min( np.sign( [ prueba[0]-min3i1[0], prueba[1]-min3i1[1], prueba[2]-min3i1[2], prueba[3]-min3i1[3] ] ) ) == -1 and max( np.sign( [ prueba[0]-max2i1[0], prueba[1]-max2i1[1], prueba[2]-max2i1[2], prueba[3]-max2i1[3] ] ) ) < 1 :
        print 'El vector: ', i, ' (', o_clase2[i], ') pertenece a la clase 2'

for i in range( len( o_clase3 ) ):
    prueba = o_clase3[i]
    if min( np.sign( [ prueba[0]-min3i1[0], prueba[1]-min3i1[1], prueba[2]-min3i1[2], prueba[3]-min3i1[3] ] ) ) > -1 and max( np.sign( [ prueba[0]-max2i1[0], prueba[1]-max2i1[1], prueba[2]-max2i1[2], prueba[3]-max2i1[3] ] ) ) == 1 :
        print 'El vector: ', i, ' (', o_clase2[i], ') pertenece a la clase 3'
'''
