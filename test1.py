import numpy as np
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut

from chat import CHAT
from chat_onehot import CHAT_ONEHOT
from encoder import BaseEncoder, IntegerEncoder

from collections import Counter

iris = datasets.load_iris()
Y = iris.target
ds = np.column_stack( (iris.data, Y) )
ds = np.genfromtxt( './datasets/glass.csv', delimiter=",", filling_values=0 )

# print encoder.positive()
# print encoder.integer()

N = np.size( ds, axis=0 )
loo = LeaveOneOut( N )
correct = 0
for train_index, test_index in loo:
	train_ds = ds[train_index]
	test = ds[test_index][0]

	integer_encoder = IntegerEncoder( train_ds[:, 0:-1] )
	train_ds_integer = integer_encoder.encode()
	test_integer = integer_encoder.encode_single( test[0:-1] )
	maximum = int( max( test_integer ) )
	classes = []
	for b in range( 1, 2 ):
		base_encoder = BaseEncoder( train_ds_integer, base=b )
		train_ds_based = base_encoder.encode()
		train_ds_encoded = np.column_stack( (train_ds_based, train_ds[:, -1]) )
		test_encoded = np.array( base_encoder.encode_single( test_integer ) )
		chat = CHAT_ONEHOT( train_ds_encoded )
		chat.train()
		c = chat.classify( test_encoded )
		classes += [c]

	print classes
	c = Counter( classes ).most_common( 1 )[0][0]
	if c == test[-1]:
		correct += 1
	#print test_encoded, int( test[-1] ), '->', c

print '\nAccuracy: ', float( correct ) / float( N )
