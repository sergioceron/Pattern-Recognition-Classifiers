import time
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.spatial
import scipy.stats
from sklearn import datasets
from decimal import *
from sklearn.cross_validation import LeaveOneOut

#pivote = 0 #2 for glass
ds = np.genfromtxt('./datasets/parkinsons.csv', delimiter=",", filling_values=0)
iris = datasets.load_iris()
Y = iris.target
ds = np.column_stack((iris.data, Y))
for i in range(4):
    for j in range(4):
        if i!=j:
            print 'Coor(',i,j,'):', scipy.stats.pearsonr(ds[:,i], ds[:,j])
#print scipy.stats.tstd(ds[:,15])