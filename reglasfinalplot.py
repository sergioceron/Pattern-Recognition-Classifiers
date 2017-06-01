import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import scipy.spatial
import scipy.stats
from sklearn import datasets
from decimal import *
from sklearn.cross_validation import LeaveOneOut

px = [] #[0, 1, 1, 1, 2 ]
py = [] #[1, 0, 2, 3, 1 ]
pivote = 1 #2 for glass
NC = 3-1
ds = np.genfromtxt('./datasets/iris2d.csv', delimiter=",", filling_values=0)
#iris = datasets.load_iris()
#Y = iris.target
#ds = np.column_stack((iris.data, Y))

def plot(it, clase_a, clase_b, rules):
    print it
    print rules
    print '-'*50
    markers = [ 'o', 'v', 'x' ]
    hatches = ['.', '/']
    colors = [ '#00ffff', '#336699', '#ff00ff' ]

    fig = plt.figure(it)
    ax = fig.add_subplot(111, aspect='equal')
    ax.axis([3.2, 6.8, 1.1, 2.5])
    cc = 0
    if rules[0][0]: # feature 0,min
        x_max = rules[0][0]["value"]
        x_min = rules[0][0]["limit"]-1
        cc = int(rules[0][0]['belongs_to'])-1
    else:
        x_min = 0
        x_max = 0

    patch = patches.Rectangle((float(x_min), 0), float(x_max)-float(x_min), 6, hatch=hatches[cc], alpha=0.2, facecolor=colors[0])
    ax.add_patch(patch)

    cc = 0
    if rules[0][1]: # feature 0,max
        x_max = rules[0][1]["limit"]+1
        x_min = rules[0][1]["value"]
        cc = int(rules[0][1]['belongs_to'])-1
    else:
        x_max = 0
        x_min = 0

    patch = patches.Rectangle((float(x_min), 0), float(x_max)-float(x_min), 6, hatch=hatches[cc], alpha=0.2, facecolor=colors[0])
    ax.add_patch(patch)

    cc = 0
    if rules[1][0]: # feature 1,min
        y_min = rules[1][0]["limit"]-1
        y_max = rules[1][0]["value"]
        cc = int(rules[1][0]['belongs_to'])-1
    else:
        y_min = 0
        y_max = 0

    #patch = patches.Rectangle((0, float(y_min)), 7, float(y_max)-float(y_min), hatch=hatches[cc], alpha=0.2, facecolor=colors[1])
    #ax.add_patch(patch)

    cc = 0
    if rules[1][1]: # feature 1,max
        y_max = rules[1][1]["limit"]+1
        y_min = rules[1][1]["value"]
        cc = int(rules[1][1]['belongs_to'])-1
    else:
        y_max = 0
        y_min = 0

    #patch = patches.Rectangle((0, float(y_min)), 7, float(y_max)-float(y_min), hatch=hatches[cc], alpha=0.2, facecolor=colors[1])
    #ax.add_patch(patch)

    ax.scatter(clase_a[:, 0], clase_a[:, 1], marker=markers[0], color='0.5', s=50, facecolors='none', edgecolors='gray', cmap=plt.cm.Paired)
    ax.scatter(clase_b[:, 0], clase_b[:, 1], marker=markers[1], color='0.5', s=50, facecolors='none', edgecolors='gray', cmap=plt.cm.Paired)


def expandir( datos, size ):
    expandidos = []
    for patron in datos:
        expanded = patron.tolist()
        #for w in range( size ):
        #    for z in range( size ):
        #        if w == pivote and (z == 3 or z == 2):
        for w, z in zip(px, py):
            expanded += [ np.power( patron[w], 1/(patron[z]+1) ) ]
        expandidos += [ expanded ]
    return np.array( expandidos )

def classify( rules, pattern ):
    clases = []
    for j in range( len(rules) ):
        if rules[j][0]: # min
            clase = check_rule( pattern[j], rules[j][0] )
            if clase != -1:
                clases += [ clase ]
        if rules[j][1]: # max
            clase = check_rule( pattern[j], rules[j][1] )
            if clase != -1:
                clases += [ clase ]

    if len( clases ) == 0:
        return -1

    print 'votos: ', clases
    c1 = [c for c in clases if c == 0]
    c2 = [c for c in clases if c == 1]

    if len( c1 ) > len( c2 ):
        return 0
    else:
        return 1

def check_rule( value, rule ):
    clase = -1 # not classified
    if rule['action'] == 'min':
        if value < rule['value'] :#and value > rule['limit']:
            clase = rule['belongs_to'] - 1
    if rule['action'] == 'max' :#and value < rule['limit']:
        if value > rule['value']:
            clase = rule['belongs_to'] - 1
    return clase

def iteracion_general( clase1, clase2, size ):
    max1 = np.max( clase1, axis = 0 )
    min1 = np.min( clase1, axis = 0 )
    max2 = np.max( clase2, axis = 0 )
    min2 = np.min( clase2, axis = 0 )

    rules = []
    for i in range( size ):
        rulemin = {}
        rulemax = {}

        if min2[i] > max1[i]:
            rulemin['action'] = 'min'
            rulemin['value'] = ( max1[i] + min2[i] ) / 2.0
            rulemin['limit'] = min1[i]
            rulemin['belongs_to'] = 1
            rulemax['action'] = 'max'
            rulemax['value'] = ( max1[i] + min2[i] ) / 2.0
            rulemax['limit'] = max2[i]
            rulemax['belongs_to'] = 2
        elif max2[i] < min1[i]:
            rulemin['action'] = 'min'
            rulemin['value'] = ( max2[i] + min1[i] ) / 2.0
            rulemin['limit'] = min2[i]
            rulemin['belongs_to'] = 2
            rulemax['action'] = 'max'
            rulemax['value'] = ( max2[i] + min1[i] ) / 2.0
            rulemax['limit'] = max1[i]
            rulemax['belongs_to'] = 1
        else:
            if min1[i] < min2[i] :
                rulemin['action'] = 'min'
                rulemin['value'] = min2[i]#.quantize(Decimal('.01'), rounding=ROUND_DOWN)
                rulemin['limit'] = min1[i]
                rulemin['belongs_to'] = 1
            elif min1[i] > min2[i] :
                rulemin['action'] = 'min'
                rulemin['value'] = min1[i]#.quantize(Decimal('.01'), rounding=ROUND_DOWN)
                rulemin['limit'] = min2[i]
                rulemin['belongs_to'] = 2

            if max1[i] < max2[i] :
                rulemax['action'] = 'max'
                rulemax['value'] = max1[i]#.quantize(Decimal('.01'), rounding=ROUND_DOWN)
                rulemax['limit'] = max2[i]
                rulemax['belongs_to'] = 2
            elif max1[i] > max2[i] :
                rulemax['action'] = 'max'
                rulemax['value'] = max2[i]#.quantize(Decimal('.01'), rounding=ROUND_DOWN)
                rulemax['limit'] = max1[i]
                rulemax['belongs_to'] = 1

        rules += [ [ rulemin, rulemax ] ]

    C1 = 0
    C2 = 0
    NOclase1 = []
    NOclase2 = []
    for i in range( len( clase1 ) ):
        clase = -1
        for j in range( size ):
            if rules[j][0]:
                clase = check_rule( clase1[i, j], rules[j][0] )
                if clase != -1:
                    break
            if rules[j][1]:
                clase = check_rule( clase1[i, j], rules[j][1] )
                if clase != -1:
                    break

        if clase == -1 :
            C1 += 1
            NOclase1 += [ i ]

    for i in range( len( clase2 ) ):
        clase = -1
        for j in range( size ):
            if rules[j][0]:
                clase = check_rule( clase2[i, j], rules[j][0] )
                if clase != -1:
                    break
            if rules[j][1]:
                clase = check_rule( clase2[i, j], rules[j][1] )
                if clase != -1:
                    break

        if clase == -1 :
            C2 += 1
            NOclase2 += [ i ]

    return rules, clase1[ NOclase1 ], clase2[ NOclase2 ]

correctas = 0
N = len( ds )
psize = len( ds[0] ) - 1
loo = LeaveOneOut( N )
for train_index, test_index in loo:
    train_ds = ds[ train_index ]
    test_ds = ds[ test_index ]

    __clase = -1
    for m in range( 1, NC ):
        o_clasea = np.array([e[0:psize] for e in train_ds if e[psize] == m])
        o_claseb = np.array([e[0:psize] for e in train_ds if e[psize] > m])
        #print ' ------ TRAINING ------ '
        #print 'ITERACION 0'
        chain_rules = []
        rules, _clasea, _claseb = iteracion_general( expandir(o_clasea, psize), expandir(o_claseb, psize), psize + len(px) )

        plot( 0, o_clasea, o_claseb, rules )

        chain_rules += [ rules ]
        for it in range( 1, 10 ):
            if len( _clasea ) == 0 or len( _claseb ) == 0:
                break;
            #print 'ITERACION ', it
            ca = _clasea
            cb = _claseb
            rules, _clasea, _claseb = iteracion_general( _clasea, _claseb, psize + len(px) )
            chain_rules += [ rules ]
            plot( it, ca, cb, rules )

        plt.show()
        break
        #print np.array( chain_rules )
        #print ' ------ TESTING ------ '
        prueba = expandir([test_ds[0, 0:psize]], psize)[0]
        clase = -1
        for rules in chain_rules:
            clase = classify( rules, prueba )
            if clase != -1:
                break;

        print m,' vs others - class: ',clase
        # esta regla esta cabrona de entender
        if m < NC - 1:
            if clase == 0 :
                __clase = m
                break;
        else:
            if clase == 0 :
                __clase = m
            else:
                __clase = m+1
            break;
    break
    print prueba, ' -> ', test_ds[0, psize], ' :: ', __clase
    if test_ds[0, psize] == __clase:
        correctas += 1
    print '-'*100

print 'Performance: ', (correctas / float(N))*100, '%'
