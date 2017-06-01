import numpy as np

v1 = [ 2, -1 ]
v2 = [ 1,  2 ]

Tv1 = [ -2, 1 ]
Tv2 = [  1, 2 ]

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

print C
print D
print CD
print C_inv
print A

_Tv1 = np.dot( A, v1 )
_Tv2 = np.dot( A, v2 )

print Tv1, _Tv1
print Tv2, _Tv2

print np.dot( A, [ 5, -4 ] )
