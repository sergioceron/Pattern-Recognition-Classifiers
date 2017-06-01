import numpy as np
import matplotlib.pyplot as plt

ds = np.genfromtxt('./datasets/ejemplo2.csv', delimiter=",", filling_values=0)

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

plt.figure(0)
markers = [ 'o', 'v', 'x' ]
means = []
for k in range( 0, 3 ): # este for solo esta por que separo los coeficientes por clase
    x = []
    y = []
    clase = np.array([e for e in ds if e[2] == k])
    means += [ clase.mean( axis = 0 ) ]
    for patron in clase:
        x += [ patron[0] ]
        y += [ patron[1] ]
    plt.scatter( x, y, marker=markers[k] )

mean_c1 = means[1][0:2]
mean_c2 = means[2][0:2]

v1 = [ 2, 2 ]
v2 = [ 5, 5 ]
plt.scatter( [v1[0]], [v1[1]], marker=markers[1], s=[100] )
plt.scatter( [v2[0]], [v2[1]], marker=markers[2], s=[100] )

Tv1 = [0,1]
Tv2 = [1,1]

A = getA( v1, v2, Tv1, Tv2 )

print v1, '->', Tv1
print v2, '->', Tv2
print A

# transformamos todos los vectores
ds_transformado = []
for patron in ds:
    vector = patron[0:2]
    vector_t = np.dot( A, vector )
    patron_t = np.concatenate((vector_t, [patron[2]]))
    ds_transformado += [ patron_t ]


plt.figure(1)
plt.scatter( [Tv1[0]], [Tv1[1]], marker=markers[1], s=[100] )
plt.scatter( [Tv2[0]], [Tv2[1]], marker=markers[2], s=[100] )

means = []
for k in range( 0, 3 ): # este for solo esta por que separo los coeficientes por clase
    x = []
    y = []
    clase = np.array([e for e in ds_transformado if e[2] == k])
    means += [ clase.mean( axis = 0 ) ]
    for patron in clase:
        x += [ patron[0] ]
        y += [ patron[1] ]
    plt.scatter( x, y, marker=markers[k] )

# segunda transformacion

v1 = [4.45, 1.46]
v2 = Tv2

Tv1 = Tv1
Tv2 = v2

A = getA(v1, v2, Tv1, Tv2)

ds_transformado2 = []
for patron in ds_transformado:
    vector = patron[0:2]
    vector_t = np.dot( A, vector )
    patron_t = np.concatenate((vector_t, [patron[2]]))
    ds_transformado2 += [ patron_t ]


plt.figure(2)
plt.scatter( [Tv1[0]], [Tv1[1]], marker=markers[1], s=[100] )
plt.scatter( [Tv2[0]], [Tv2[1]], marker=markers[2], s=[100] )

means = []
for k in range( 0, 3 ): # este for solo esta por que separo los coeficientes por clase
    x = []
    y = []
    clase = np.array([e for e in ds_transformado2 if e[2] == k])
    means += [ clase.mean( axis = 0 ) ]
    for patron in clase:
        x += [ patron[0] ]
        y += [ patron[1] ]
    plt.scatter( x, y, marker=markers[k] )


plt.show()
