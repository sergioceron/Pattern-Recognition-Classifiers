import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
Y = iris.target*20

fig = plt.figure(1, figsize=(8, 6))

vzero = [0.0, 0.0, 0.0, 0.0]

"""
x1=[5.1,5.2,4.6,5.1]
y1=[3.7,4.1,3.4,3.4]
z1=[1.5,1.5,1.4,1.5]

x2=[7.0,6.0,5.7,6.5]
y2=[3.2,2.2,2.8,2.8]
z2=[4.7,4.0,4.1,4.6]

x3=[5.8,6.9,7.6,7.3]
y3=[2.7,3.1,3.0,2.9]
z3=[5.1,5.4,6.6,6.3]
"""
x1=[5.1,5.5,5.2,4.7]
y1=[3.8,3.5,3.4,3.2]
z1=[1.5,1.3,1.4,1.6]

x2=[4.9,6.1,5.2,6.1]
y2=[2.4,2.8,2.7,3.0]
z2=[3.3,4.7,3.9,4.6]

x3=[7.7,6.2,5.8,7.9]
y3=[2.6,3.4,2.7,3.8]
z3=[6.9,5.4,5.1,6.4]


s = [140 for n in range(len(x1))]

c1 = [0 for n in range(len(x1))]
c2 = [120 for n in range(len(x1))]
c3 = [240 for n in range(len(x1))]

ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(iris.data[:, 0], iris.data[:, 1], iris.data[:, 2], c=Y*120)
ax.scatter(x1, y1, z1, s=s, c=c1)
ax.scatter(x2, y2, z2, s=s, c=c2)
ax.scatter(x3, y3, z3, s=s, c=c3)

for i in range(4):
    ax.plot( [vzero[i], x1[i]], [vzero[i], y1[i]], zs=[vzero[i], z1[i]] )
    ax.plot( [vzero[i], x2[i]], [vzero[i], y2[i]], zs=[vzero[i], z2[i]] )
    ax.plot( [vzero[i], x3[i]], [vzero[i], y3[i]], zs=[vzero[i], z3[i]] )

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
