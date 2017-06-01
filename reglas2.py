import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.spatial
import scipy.stats
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut

iris = datasets.load_iris()
Y = iris.target
ds = np.column_stack((iris.data, Y))

o_clase2 = np.array([e[0:4] for e in ds if e[4] == 1])
o_clase3 = np.array([e[0:4] for e in ds if e[4] == 2])

def expandir( datos, size=4 ):
    expandidos = []
    for patron in datos:
        expanded = []
        for w in range( size ):
            for z in range( w+1, size ):
                expanded += [ patron[w] + patron[z] ]
        expandidos += [ expanded ]
    return np.array( expandidos )

def classify( rules, pattern ):
    clase = -1
    for j in range( len(rules) ):
        if rules[j][0]:
            clase = check( pattern[j], rules[j][0] )
            if clase != -1:
                break
        if rules[j][1]:
            clase = check( pattern[j], rules[j][1] )
            if clase != -1:
                break
    return clase

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

    rules = []
    for i in range( size ):
        rulemin = {}
        rulemax = {}
        if min1[i] < min2[i] :
            rulemin['action'] = 'min'
            rulemin['value'] = min2[i]
            rulemin['belongs_to'] = 1
        elif min1[i] > min2[i] :
            rulemin['action'] = 'min'
            rulemin['value'] = min1[i]
            rulemin['belongs_to'] = 2

        if max1[i] < max2[i] :
            rulemax['action'] = 'max'
            rulemax['value'] = max1[i]
            rulemax['belongs_to'] = 2
        elif max1[i] > max2[i] :
            rulemax['action'] = 'max'
            rulemax['value'] = max2[i]
            rulemax['belongs_to'] = 1

        rules += [ [ rulemin, rulemax ] ]

    print np.array(rules)

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

chain_rules = []
print ' --------------- ITERACION 1 ------------------ '
a, b, rules, _clase2, _clase3 = iteracion_general( o_clase2, o_clase3 )
chain_rules += [ rules ]
print ' --------------- ITERACION 2 ------------------ '
a, b, rules, _clase2, _clase3 = iteracion_general( _clase2, _clase3 )
chain_rules += [ rules ]
print ' --------------- ITERACION 3 ------------------ '
a, b, rules, _clase2, _clase3 = iteracion_general( _clase2, _clase3 )
chain_rules += [ rules ]
print ' --------------- ITERACION 4 ------------------ '
a, b, rules, _clase2, _clase3 = iteracion_general( _clase2, _clase3 )
chain_rules += [ rules ]

# PRUEBA CON RESUBSTITUTION
print ' --------------- RESUBSTITUTION ERROR ------------------ '
for prueba in o_clase2:
    clase = -1
    for rules in chain_rules:
        clase = classify( rules, prueba )
        if clase != -1:
            break;
    print prueba, ' -> ', 2, ' :: ', (clase + 2)

for prueba in o_clase3:
    clase = -1
    for rules in chain_rules:
        clase = classify( rules, prueba )
        if clase != -1:
            break;
    print prueba, ' -> ', 3, ' :: ', (clase + 2)
