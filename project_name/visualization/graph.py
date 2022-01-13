from matplotlib import pyplot as plt
from numpy import unique
from numpy import where



def graph(yhat, clusters, x, path, name):
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
        # show the plot
        plt.savefig(fr'{path}'+name)
        #plt.show()
