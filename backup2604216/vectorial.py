import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
Y = iris.target

ds = iris.data

ds = np.column_stack((ds, Y))

c1 = np.array( [e for e in ds if e[4] == 0] )
c2 = np.array( [e for e in ds if e[4] == 1] )
c3 = np.array( [e for e in ds if e[4] == 2] )

max_index = np.argmax(c1, axis=0)
min_index = np.argmin(c1, axis=0)

max_x_c1 = c1[max_index[0]]
max_y_c1 = c1[max_index[1]]
min_x_c1 = c1[min_index[0]]
min_y_c1 = c1[min_index[1]]

print max_x_c1, max_y_c1, min_x_c1, min_y_c1

x = ds[:, 0]
y = ds[:, 1]

a = np.array([0.0, 0.0])
b = rnd.choice(c1)[0:2] #[1.3, 2.0]
c = rnd.choice(c1)[0:2] #[5.6, -1.0]

N = 150
#x = -5+np.random.rand(N)*9.1
#y = -1+np.random.rand(N)*3.1
z = []

Memory = np.array([b, c])

for i in range(N):
    test = np.array([x[i], y[i]])

    coefficients = np.linalg.solve(np.transpose( Memory ), test)
    variance = np.var( coefficients )
    z += [1 + np.pi * (1 * variance**2) ]

plt.scatter( x, y, s = z, c = ds[:, 4], alpha = 0.5 )

plt.plot( [a[0], b[0]], [a[1], b[1]], 'k-')
plt.plot( [a[0], c[0]], [a[1], c[1]], 'k-')

d = map(sum, zip(b,c))
plt.plot( [a[0], d[0]], [a[1], d[1]], 'k-')

plt.show()
