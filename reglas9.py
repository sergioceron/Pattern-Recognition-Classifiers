import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.spatial
import scipy.stats
from sklearn import datasets
from decimal import *
from sklearn.cross_validation import LeaveOneOut

ds = np.genfromtxt('./datasets/breast-cancer-wisconsin.csv', delimiter=",", filling_values=0)
iris = datasets.load_iris()
Y = iris.target
ds = np.column_stack((iris.data, Y))

def expandir( datos, size=4 ):
    expandidos = []
    for patron in datos:
        expanded = patron.tolist()
        for w in range( size ):
            for z in range( size ):
                if w==1 and w != z:
                    expanded += [ np.power( patron[w], 1/(patron[z]+1) ) ]
        expandidos += [ expanded ]
    return np.array( expandidos )

def classify( rules, pattern ):
    clases = []
    for j in range( len(rules) ):
        if rules[j][0]:
            clase = check( pattern[j], rules[j][0] )
            if clase != -1:
                clases += [ clase ]
        if rules[j][1]:
            clase = check( pattern[j], rules[j][1] )
            if clase != -1:
                clases += [ clase ]

    if len( clases ) == 0:
        return -1

    c1 = [c for c in clases if c == 0]
    c2 = [c for c in clases if c == 1]
    print 'votos: ', clases
    if len( c1 ) > len( c2 ):
        return 0
    else:
        return 1

def check( value, rule ):
    clase = -1 # not classified
    if rule['action'] == 'min':
        if value < rule['value']:
            clase = rule['belongs_to'] - 1
    if rule['action'] == 'max':
        if value > rule['value']:
            clase = rule['belongs_to'] - 1
    return clase

def iteracion_general( clase1, clase2, size=4 ):
    max1 = np.max( clase1, axis = 0 )
    min1 = np.min( clase1, axis = 0 )
    max2 = np.max( clase2, axis = 0 )
    min2 = np.min( clase2, axis = 0 )
    print min1, max1
    print min2, max2
    rules = []
    for i in range( size ):
        rulemin = {}
        rulemax = {}
        
        if min2[i] > max1[i]:
            rulemin['action'] = 'min'
            rulemin['value'] = ( max1[i] + min2[i] ) / 2.0
            rulemin['belongs_to'] = 1
            rulemax['action'] = 'max'
            rulemax['value'] = ( max1[i] + min2[i] ) / 2.0
            rulemax['belongs_to'] = 2
            print 'this rule'
        elif max2[i] < min1[i]:
            rulemin['action'] = 'min'
            rulemin['value'] = ( max2[i] + min1[i] ) / 2.0
            rulemin['belongs_to'] = 2
            rulemax['action'] = 'max'
            rulemax['value'] = ( max2[i] + min1[i] ) / 2.0
            rulemax['belongs_to'] = 1
        else:
            if min1[i] < min2[i] :
                rulemin['action'] = 'min'
                rulemin['value'] = Decimal(min2[i]).quantize(Decimal('.01'), rounding=ROUND_DOWN)
                rulemin['belongs_to'] = 1
            elif min1[i] > min2[i] :
                rulemin['action'] = 'min'
                rulemin['value'] = Decimal(min1[i]).quantize(Decimal('.01'), rounding=ROUND_DOWN)
                rulemin['belongs_to'] = 2

            if max1[i] < max2[i] :
                rulemax['action'] = 'max'
                rulemax['value'] = Decimal(max1[i]).quantize(Decimal('.01'), rounding=ROUND_DOWN)
                rulemax['belongs_to'] = 2
            elif max1[i] > max2[i] :
                rulemax['action'] = 'max'
                rulemax['value'] = Decimal(max2[i]).quantize(Decimal('.01'), rounding=ROUND_DOWN)
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
                clase = check( clase1[i, j], rules[j][0] )
                if clase != -1:
                    break
            if rules[j][1]:
                clase = check( clase1[i, j], rules[j][1] )
                if clase != -1:
                    break

        if clase == -1 :
            C1 += 1
            NOclase1 += [ i ]

    for i in range( len( clase2 ) ):
        clase = -1
        for j in range( size ):
            if rules[j][0]:
                clase = check( clase2[i, j], rules[j][0] )
                if clase != -1:
                    break
            if rules[j][1]:
                clase = check( clase2[i, j], rules[j][1] )
                if clase != -1:
                    break

        if clase == -1 :
            C2 += 1
            NOclase2 += [ i ]

    print 'Clase 2 not classified (count): ', C1
    print 'Clase 3 not classified (count): ', C2

    return NOclase1, NOclase2, rules, clase1[ NOclase1 ], clase2[ NOclase2 ]

correctas = 0
N = len(ds)
psize = len(ds[0])-1
loo = LeaveOneOut( N )
for train_index, test_index in loo:
    train_ds = ds[ train_index ]
    test_ds = ds[ test_index ]

    o_clase2 = np.array([e[0:psize] for e in train_ds if e[psize] == 0])
    o_clase3 = np.array([e[0:psize] for e in train_ds if e[psize] == 1])

    chain_rules = []
    a, b, rules, _clase2, _clase3 = iteracion_general( expandir(o_clase2), expandir(o_clase3), 7 )
    
    chain_rules += [ rules ]
    for it in range( 10 ):
        if len( a ) == 0 or len( b ) == 0:
            break;
        print ' ------ ITERACION ', it, ' ------- '
        a, b, rules, _clase2, _clase3 = iteracion_general( _clase2, _clase3, 17 )
        chain_rules += [ rules ]
    
    print np.array(chain_rules)
    print ' ------ TESTING ------ '
    prueba = expandir([test_ds[0, 0:psize]])[0]
    clase = -1
    for rules in chain_rules:
        clase = classify( rules, prueba )
        if clase != -1:
            break;
    print prueba, ' -> ', test_ds[0, psize], ' :: ', clase
    if test_ds[0, psize] == clase:
        correctas += 1
    print '-'*100

print 'Performance: ', (correctas / float(N))*100, '%'
