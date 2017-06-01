import time
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.spatial
import scipy.stats
from sklearn import datasets
from decimal import *
from sklearn.cross_validation import LeaveOneOut

pivote = 0 #2 for glass
NC = 2-1
ds = np.genfromtxt('./datasets/parkinsons.csv', delimiter=",", filling_values=0)
#iris = datasets.load_iris()
#Y = iris.target
#ds = np.column_stack((iris.data, Y))

def expandir( datos, size ):
    expandidos = []
    for patron in datos:
        expanded = patron.tolist()
        for w in range( size ):
            for z in range( size ):
                if w == pivote and w != z:
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
                rulemin['value'] = Decimal(min2[i]).quantize(Decimal('.01'), rounding=ROUND_DOWN)
                rulemin['limit'] = min1[i]
                rulemin['belongs_to'] = 1
            elif min1[i] > min2[i] :
                rulemin['action'] = 'min'
                rulemin['value'] = Decimal(min1[i]).quantize(Decimal('.01'), rounding=ROUND_DOWN)
                rulemin['limit'] = min2[i]
                rulemin['belongs_to'] = 2

            if max1[i] < max2[i] :
                rulemax['action'] = 'max'
                rulemax['value'] = Decimal(max1[i]).quantize(Decimal('.01'), rounding=ROUND_DOWN)
                rulemax['limit'] = max2[i]
                rulemax['belongs_to'] = 2
            elif max1[i] > max2[i] :
                rulemax['action'] = 'max'
                rulemax['value'] = Decimal(max2[i]).quantize(Decimal('.01'), rounding=ROUND_DOWN)
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
    for m in range( NC ):
        o_clasea = np.array([e[0:psize] for e in train_ds if e[psize] == m])
        o_claseb = np.array([e[0:psize] for e in train_ds if e[psize] > m])
        #print ' ------ TRAINING ------ '
        #print 'ITERACION 0'
        start_time = time.time()
        chain_rules = []
        rules, _clasea, _claseb = iteracion_general( expandir(o_clasea, psize), expandir(o_claseb, psize), 2*psize-1 )

        chain_rules += [ rules ]
        for it in range( 1, 10 ):
            if len( _clasea ) == 0 or len( _claseb ) == 0:
                break;
            #print 'ITERACION ', it
            rules, _clasea, _claseb = iteracion_general( _clasea, _claseb, 2*psize-1 )
            chain_rules += [ rules ]
            
        print("Train time: %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        #print np.array( chain_rules )
        #print ' ------ TESTING ------ '
        prueba = expandir([test_ds[0, 0:psize]], psize)[0]
        clase = -1
        for rules in chain_rules:
            clase = classify( rules, prueba )
            if clase != -1:
                break;
        
        print("Test time: %s seconds ---" % (time.time() - start_time))

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
        
    if test_ds[0, psize] == __clase:
        correctas += 1
    print '-'*100
    break

#print 'Performance: ', (correctas / float(N))*100, '%'
