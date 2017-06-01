# TODO: EN OTRO ARCHIVO. PONER PESOS A LAS DISTANCIAS CON BASE EN LA CLASE CONOCIDA
# TAMBIEN PROBAR CON OTRO TIPO DE DISTANCIA, ALGO MAS REAL A LO QUE SIGNIFICAN LOS COEFICIENTES
# TAMBIEN USAR TRANSLACION DE EJES AL CENTROIDE, COMO EL CHAT (ALGO BUENO EN ESTAS MAMADAS)
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy import stats
import scipy
import urllib
import sys
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut, train_test_split
from pyswarm import pso

class prettyfloat(float):
    def __repr__(self):
        return "% 9.4f" % self

def get_memories_from_centroids( ds, num_clases, pattern_size ):
    classes = []
    means = []
    nearests = []
    memorias = []
    for i in range( num_clases ):
        classes += [np.array( [e[0:pattern_size] for e in ds if e[pattern_size] == i] )]
        means += [ scipy.stats.variation( classes[i], axis = 0 ) ]
        searchTree = scipy.spatial.cKDTree( np.copy( classes[i] ), leafsize = 100 )
        dummy, nearest = searchTree.query( means[i], k = pattern_size, p = pattern_size )
        nearests +=  [nearest]
        print 'Mean of class: ',i, means[i]
        print 'Memory', i, '(m-Nearest)', classes[i][ nearests[i] ]
        memorias += [np.array( classes[i][ nearests[i] ]) ]

    return memorias

def get_coeficientes( ds, memories ):
    c = len( memories )
    coeficientes = {}

    for i in range( c ):
        coeficientes[ str( i ) ] = []

    for i in range( len( ds ) ):
        test = np.array( ds[i, 0:m] )
        tclass = ds[i, m]
        cc = []
        for j in range( c ):
            try:
                cc = np.concatenate((cc, np.linalg.solve( np.transpose( memories[ int( j ) ] ), test ) ))
            except:
                cc = np.concatenate(( cc, np.zeros( m ) ))
                pass
        #print 'cc:', len(cc)
        coeficientes[ str( int( tclass ) ) ] += [ cc ]

    return coeficientes

# GENERAR EL VECTOR CTEST COMPLETO PARA CALCULAR LA DISTANCIA Y LUEGO
# CREAR UN ONE VECTOR CON ZERO EN LA COMPONENTE QUE REPRESENTA LA CLASE (ZERO POR EL MIN)
def get_distances( coeficientes, memories, test ):
    c = len( memories )

    ctest = []
    for j in range( c ):
        try :
            ctest = np.concatenate(( ctest, np.linalg.solve( np.transpose( memories[ j ] ), test ) ))
        except:
            ctest = np.concatenate(( ctest, np.zeros( len( test ) ) )) #m = len( test )

    print 'Coeficientes Prueba:', map( prettyfloat , ctest )
    distances = []
    for clase in range( c ):
        coeficiente = coeficientes[ str( clase ) ]

        dists = []
        for coef in coeficiente:
            dists += [ scipy.spatial.distance.cdist([ctest], [coef], 'cityblock')[0][0] ]
        distances += [ min( dists ) ]

    return distances

# MAIN CODE PROGRAM
iris = datasets.load_iris()
Y = iris.target
#raw_data = urllib.urlopen( "file:///Users/sergio/Downloads/glass.csv" )
#raw_data = urllib.urlopen( "http://goo.gl/j0Rvxq" )
# load the CSV file as a numpy matrix
#ds = np.column_stack((iris.data, Y))
#ds = np.loadtxt("/Users/sergio/Downloads/mammographic_filled.csv", delimiter=",") # ok
#ds = np.loadtxt("wine.csv", delimiter=",") # ok
#ds = np.loadtxt("parkinsons.csv", delimiter=",") # ok
#ds = np.loadtxt("w-breast-cancer.csv", delimiter=",")
#ds = np.loadtxt("tae.csv", delimiter=",")
#ds = np.loadtxt("ecoli.csv", delimiter=",")
#ds = np.loadtxt("habberman.csv", delimiter=",") # ok
#ds = np.loadtxt("heart-stat.csv", delimiter=",")
ds = np.loadtxt("xor.csv", delimiter=",")

m = len( ds[0] ) - 1                # size of pattern (ej. iris = 4)
c = len( np.unique( ds[ :, m ] ) )  # number of classes
N = len( ds )                       # size of dataset

train_ds = ds
test_ds = np.array([ds[ 0 ]])

test = np.array( test_ds[0, 0:m] )
tclass = test_ds[0, m]

memorias = get_memories_from_centroids( train_ds, c, m )

coeficientes = get_coeficientes( train_ds, memorias )
print 'New DS (Coeficientes):'
for key in coeficientes:
    for _coef in coeficientes[key]:
        print map( prettyfloat , np.array(_coef) )

distances = get_distances( coeficientes, memorias, test )
print 'Distancias: ', distances

minclass = np.argmin( distances )

print 'Clase Original:', int(tclass)
print 'Clase Calculada:', minclass
