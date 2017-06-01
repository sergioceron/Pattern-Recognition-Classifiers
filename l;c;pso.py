# ESTA PRUEBA CONSISTIO EN USAR PSO PARA CALCULAR EL SISTEMA GENERADOR, A NIVEL DE COMPONENTE
# SE REALIZARON PRUEBAS USANDO 1000 PARTICULAS Y HASTA 10 ITERACIONES, LO CUAL NO LLEVO A NINGUN BUEN RESULTADO
# CREO QUE AQUI SE PUEDE DEMOSTRAR QUE NO EXISTE NINGUN CONJUNTO DE VECTORES (DENTRO O FUERA) DEL CONJUNTO
# DE ENTRENAMIENTO QUE PERMITA SEPARAR LAS CLASES
# TAMBIEN SE HICIERON PRUEBAS CON KNN EN VEZ DE 1NN Y AUMENTO EL RENDIMIENTO COMO SE ESPERARIA
# PLUS, SE ANIMo LA GRAFICA DE SCATTER PARA IR VIENDO COMO AFECTABA EL S.G. AL CAMBIO DE BASE
import numpy as np
import numpy.random as rnd
from scipy import stats
import scipy
import urllib
import sys
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut, train_test_split
from pyswarm import pso
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ScatterIteration:
    def __init__( self, x, y, z, p ):
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)
        self.p = p

class Counter:
    def __init__( self ):
        self.i = 0
        self.scatters = []

    def next( self, scatter ):
        self.i += 1
        self.scatters += [ scatter ]

    def val( self ):
        return self.i

    def scatter( self, index ):
        return self.scatters[index]


def get_memories( ds, indexes, num_clases, pattern_size ):
    memory = []
    #for j in range( pattern_size ):
    #    index = int( indexes [ j ] ) # round lower
    #    memory += [ ds[ index ][0:pattern_size] ]
    for j in range( pattern_size ):
        vector = []
        for w in range( pattern_size ):
            vector += [ indexes [ j*pattern_size + w ] ]
        memory += [ vector ]

    return [ memory ]

def get_coeficientes( ds, memories, c ):
    coeficientes = {}

    for i in range( c ):
        coeficientes[ str( i ) ] = []

    for i in range( len( ds ) ):
        test = np.array( ds[i, 0:m] )
        tclass = ds[i, m]
        cc = []
        for j in range( len( memories ) ):
            try:
                cc += [ np.linalg.solve( np.transpose( memories[ int( j ) ] ), test ) ]
            except:
                cc += [ np.zeros( m ) ]
                pass

        coeficientes[ str( int( tclass ) ) ] += [ cc ]

    return coeficientes

def get_distances( coeficientes, memories, test, c ):
    ctest = []
    for j in range( len(memories) ):
        try :
            ctest += [ np.linalg.solve( np.transpose( memories[ j ] ), test ) ]
        except:
            ctest += [ np.zeros( len( test ) ) ]

    distances = []
    for clase in range( c ):
        coeficiente = coeficientes[ str( clase ) ]

        dists = []
        for coef in coeficiente:
            distances += [ (scipy.spatial.distance.cdist( [ctest[0]], [coef[0]], 'cityblock' )[0][0], clase ) ]

	dtype = [('dist', float), ('clase', int)]
	_dist = np.array( distances, dtype = dtype )
	_dist = np.sort( _dist, order = 'dist' )

	dist = np.zeros( c )
	for _d in _dist[0:5]:
		dist[ _d['clase'] ] += 1

    return dist

def colors(z):
    mapa = ['red', 'green', 'blue']
    colors = []
    for _z in z:
        colors += [ mapa[ int(_z) ] ]
    return colors

def J(p, *args):
    ds, c, m, N, plt, count = args
    #print p
    memorias = get_memories( ds, p, c, m )
    #coeficientes = get_coeficientes( ds, memorias, c )

    train, testds = train_test_split(ds, test_size=0.1, random_state=12)

    coeficientes = get_coeficientes( train, memorias, c )

    correct = 0
    for i in range( len( testds ) ):
        test = np.array( testds[i, 0:m] )
        tclass = testds[i, m]

        distances = get_distances( coeficientes, memorias, test, c )
        minclass = np.argmax( distances )

        if minclass == tclass:
            correct += 1

    performance = float((correct/float(len(testds))))

    #markers = [ 'o', 'v', 'x' ]
    #plt.figure( count.val() )
    x = []
    y = []
    z = []
    for k in range( c ):
        coef = coeficientes[str(k)]
        for _c in coef:
            x += [ _c[0][0] ]
            y += [ _c[0][1] ]
            z += [ k ]
        #plt.scatter( x, y, c = colors(z), marker=markers[k] )

    it = ScatterIteration( x, y, z, performance )

    count.next(it)

    #print 'Error: ', 1.0-performance
    return 1.0-performance

# MAIN CODE PROGRAM

iris = datasets.load_iris()
Y = iris.target
#ds = np.column_stack((iris.data, Y))
ds = np.loadtxt("./datasets/iris2d.csv", delimiter=",") # ok
#ds = np.loadtxt("mammographic_filled.csv", delimiter=",") # ok
#ds = np.loadtxt("./datasets/lung_cancer.csv.conflict", delimiter=",") # ok
#ds = np.loadtxt("parkinsons.csv", delimiter=",") # ok
#ds = np.loadtxt("w-breast-cancer.csv", delimiter=",")
#ds = np.loadtxt("tae.csv", delimiter=",")
#ds = np.loadtxt("ecoli.csv", delimiter=",")
#ds = np.loadtxt("habberman.csv", delimiter=",") # ok
#ds = np.loadtxt("heart-stat.csv", delimiter=",")

m = len( ds[0] ) - 1                # size of pattern (ej. iris = 4)
c = len( np.unique( ds[ :, m ] ) )  # number of classes
N = len( ds )                       # size of dataset

# TRAINING PHASE FIRST STAGE (SUPPORT PATTERNS)
correct = 0
loo = LeaveOneOut( N )
itera = 0
for train_index, test_index in loo:
    train_ds = ds[ train_index ]
    test_ds = ds[ test_index ]

    test = np.array( test_ds[0, 0:m] )
    tclass = test_ds[0, m]

    # ====================== SE CALCULAN LAS MEMORIAS ======================
    #plt.figure(0)
    #plt.scatter( train_ds[:, 0], train_ds[:, 1], c = colors(train_ds[:, -1]) )

    count = Counter()

    fig, ax = plt.subplots()
    scat = ax.scatter( train_ds[:, 0], train_ds[:, 1], c = colors(train_ds[:, -1]) )
    #global scat
    def animate(i):
        idx = i%count.val()
        print 'animation: ', idx, i, ', performance: ', count.scatters[idx].p
        #print count.scatters[idx].x
        #print count.scatters[idx].y
        print '-'*100
        ax.clear()
        scat = ax.scatter( count.scatter(idx).x, count.scatter(idx).y, c = count.scatter(idx).z )
        #scat.set_offsets( np.row_stack( (count.scatter(i).x, count.scatter(i).y) ) )
        #scat.set_array( count.scatter(i).z )

        return scat

    args = (train_ds, c, m, N - 1, plt, count) # -1 because LeaveOneOut
    # CACULAMOS EL UB Y LB
    lb = []
    ub = []
    for j in range( m ):
        for w in range( m ):
            lb += [ 0 ]
            ub += [ 100 ]

    xopt, fopt = pso(J, lb, ub, args=args, debug=False, minfunc=0.1, maxiter=1, swarmsize=100)


    ani = animation.FuncAnimation( fig, animate, np.arange(1, count.val()), interval = 500, blit = False, repeat=False )

    plt.show()

    memorias = get_memories( train_ds, xopt, c, m)
    print 'PSO Final Performance: ', 1.0-fopt
    #print 'PSO Final Memories: ', memorias

    coeficientes = get_coeficientes( train_ds, memorias, c )
    distances = get_distances( coeficientes, memorias, test, c )

    minclass = np.argmax( distances )

    print 'i: ', itera, ', res: ', tclass, '<->', minclass, '::', distances, '::', (minclass == tclass)
    itera += 1
    if minclass == tclass:
        correct += 1

    break

print 'Final Performance: %3.4f' %  float((correct/float(N)))
