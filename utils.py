def open_irisds():
	iris = datasets.load_iris()
        Y = iris.target
        return np.column_stack( (iris.data, Y) )

def open_dataset( file_name, missing_values=0 ):
	ds = np.genfromtxt( file_name, delimiter=",", filling_values=missing_values )

def get_statistics( dataset ):
	card = np.unique(dataset[:,-1]).size
	size = dataset.size[0]
	return (card, size)
 

