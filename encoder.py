import numpy as np
import math


class BaseEncoder:
	def __init__( self, ds, base=2 ):
		self.ds = ds
		self.base = base
		#self.padding_d = [math.log( x, self.base ) for x in np.max( ds, axis=0 )]
		if base == 1:
			self.padding = [x for x in np.max( ds, axis=0 )]
		else:
			self.padding = [int( np.floor( math.log( x, self.base ) ) + 1 ) for x in np.max( ds, axis=0 )]

	def encode( self ):
		new_ds = []
		for p in self.ds:
			new_p = [np.pad(self.toBase( x, base=self.base ).split(","), (z, 0), mode='constant')[-z:].tolist() for x, z in zip( p, self.padding )]
			new_p = [int( item ) if item else 0 for sublist in new_p for item in sublist]
			new_ds += [new_p]

		return np.array( new_ds )

	def encode_single( self, p ):
		new_p = [np.pad(self.toBase( x, base=self.base ).split(","), (z, 0), mode='constant')[-z:].tolist() for x, z in zip( p, self.padding )]
		return [int( item ) if item else 0 for sublist in new_p for item in sublist]

	def toBase( self, n, base=2 ):
		if base == 1:
			arr = ['1,' for _ in range(n)]
			return ''.join(arr)

		if n < base:
			return str( n )
		else:
			return self.toBase( n // base, base ) + "," + str( n % base )


class IntegerEncoder():
	def __init__( self, ds ):
		self.ds = ds
		self.decimals = []

	def positive( self ):
		min_by_feature = np.min( self.ds, axis=0 )
		new_ds = []
		for p in self.ds:
			new_ds += [np.add( p, min_by_feature )]
		return np.array( new_ds )

	def encode( self ):

		for i in range( np.size( self.ds, axis=1 ) ):
			feature_array = self.ds[:, i]
			max_dec = 0
			for j in feature_array:
				dec = len( str( j ).split( '.' )[1] )
				if dec > max_dec:
					max_dec = dec
			self.decimals += [max_dec]

		new_ds = []
		for p in self.ds:
			new_ds += [[int( m * (10 ** x) ) for x, m in zip( self.decimals, p )]]

		return np.array( new_ds )

	def encode_single( self, p ):
		return [int( m * (10 ** x) ) for x, m in zip( self.decimals, p )]

