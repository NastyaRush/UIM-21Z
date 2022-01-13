# mean shift clustering
from numpy import unique, where
from matplotlib import pyplot
from sklearn.cluster import AffinityPropagation
import sys
import os
actual_path = (os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")).replace('models', 'data')
sys.path.insert(1, actual_path)
import make_dataset


def affinity_propagation():
	# read processed data 
	x = make_dataset.read_data()
	# define the model
	model = AffinityPropagation(damping=0.7)
	# fit model and predict clusters
	yhat = model.fit_predict(x)
	# retrieve unique clusters
	clusters = unique(yhat)
	return yhat, clusters, x
