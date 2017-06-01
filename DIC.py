import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class DIC:
	def __init__( self, pivot=0, _lambda=1, max_iter=5, plot=False ):
		self.pivot = pivot
		self.plot = plot
		self.nodes = []
		self._lambda = _lambda
		self.max_iter = max_iter

	def pdf( self, data ):
		density = gaussian_kde( np.unique( data ) )
		density.covariance_factor = lambda: self._lambda
		density._compute_covariance()
		return density

	def plot_ini( self, ax, f ):
		ax.tick_params( labelsize=9 )
		ax.set_title( 'Feature ' + str( f ), fontsize=9 )

	def plot_end( self, ax ):
		ax.legend( loc='upper right', fontsize=9 )

	def plot_pdf( self, pdf, class_set, class_num, ax ):
		xs = np.linspace( min( class_set ) - 1, max( class_set ) + 1, 200 )
		ax.plot( xs, pdf( xs ), label="Class " + str( class_num ) )
		ax.plot( class_set, pdf( class_set ), 'o' )

	def get_nodes( self, dataset, level ):
		c1 = np.array( [e[0:-1] for e in dataset if e[-1] == self.pivot] )
		c2 = np.array( [e[0:-1] for e in dataset if e[-1] > self.pivot] )
		max1 = np.max( c1, axis=0 )
		min1 = np.min( c1, axis=0 )
		max2 = np.max( c2, axis=0 )
		min2 = np.min( c2, axis=0 )

		size = np.size( dataset, axis=1 ) - 1

		c1set = np.array( [] )
		c2set = np.array( [] )

		if self.plot: fun, axs = plt.subplots( size )

		nodes = []
		for f in range( size ):
			if min2[f] > max1[f]:
				threshold = (max1[f] + min2[f]) / 2.0
				c1set = np.array( [e[f] for e in c1 if e[f] < threshold] )
				c2set = np.array( [e[f] for e in c2 if e[f] > threshold] )
			elif max2[f] < min1[f]:
				threshold = (max2[f] + min1[f]) / 2.0
				c1set = np.array( [e[f] for e in c1 if e[f] > threshold] )
				c2set = np.array( [e[f] for e in c2 if e[f] < threshold] )
			else:
				upper = 0
				lower = 0
				if min1[f] < min2[f]:
					lower = min2[f]
					c1set = np.array( [e[f] for e in c1 if min1[f] <= e[f] < min2[f]] )
				elif min1[f] > min2[f]:
					lower = min1[f]
					c2set = np.array( [e[f] for e in c2 if min2[f] <= e[f] < min1[f]] )

				if max1[f] < max2[f]:
					upper = max1[f]
					c2set = np.array( [e[f] for e in c2 if max1[f] < e[f] <= max2[f]] )
				elif max1[f] > max2[f]:
					upper = max2[f]
					c1set = np.array( [e[f] for e in c1 if max2[f] < e[f] <= max1[f]] )

				intersection = np.array( [e for e in dataset if lower <= e[f] <= upper] )

			if self.plot: self.plot_ini( axs[f], f )

			if np.unique( c1set ).size > 1:
				c1density = self.pdf( c1set )
				if self.plot: self.plot_pdf( c1density, c1set, 1, axs[f] )
				nodes += [{
					'f': f,
					'l': level,
					'm': max( c1density( c1set ) ),
					'd': c1density,
					'c': 0
				}]
			if np.unique( c2set ).size > 1:
				c2density = self.pdf( c2set )
				if self.plot: self.plot_pdf( c2density, c2set, 2, axs[f] )
				nodes += [{
					'f': f,
					'l': level,
					'm': max( c2density( c2set ) ),
					'd': c2density,
					'c': 1
				}]

			if self.plot: self.plot_end( axs[f] )

		return nodes, intersection

	def classify( self, pattern ):
		classes = [0, 0]
		for node in self.nodes:
			#classes[node['c']] += (1.0/float(node['l'])) * (float(node['d']( pattern[node['f']] ))/float(node['m']))
			classes[node['c']] += float(node['d']( pattern[node['f']] ))/float(node['m'])

		return np.argmax( classes )

	def train( self, dataset ):
		intersection = dataset
		iteration = 1
		while intersection.size > 0 and iteration < self.max_iter:
			nodes_in_level, intersection = self.get_nodes( intersection, iteration )
			self.nodes += nodes_in_level
			iteration += 1
		if self.plot:
			plt.tight_layout()
			plt.show()
