# birch clustering
from numpy import unique
from numpy import where
from sklearn.cluster import Birch
from matplotlib import pyplot
from sklearn.preprocessing import scale
import sys
import os
# birch clusteringx
from sklearn.cluster import Birch
actual_path = (os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")).replace('models', 'data')
sys.path.insert(1, actual_path)
import make_dataset


def birch():
	# read processed data 
	x = make_dataset.read_data()
	# define the model
	model = Birch(threshold=0.05, n_clusters=2)
	# fit the model
	yhat = model.fit_predict(x)
	# retrieve unique clusters
	clusters = unique(yhat)
	return yhat, clusters, x

# # create scatter plot for samples from each cluster
# for cluster in clusters:
# 	# get row indexes for samples with this cluster
# 	row_ix = where(yhat == cluster)
# 	# create scatter of these samples
# 	pyplot.scatter(x[row_ix, 0], x[row_ix, 1])
# # show the plot

# # loading the dataset
# print("davies: ", davies_bouldin_score(x, yhat))
# print("silhouette: ", silhouette_score(x, yhat))


# pyplot.show()