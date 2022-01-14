from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import json


def write_results(data, path, mode):
    with open(fr'{path}results.txt', mode) as f:
        f.write(data)

# davies_bouldin_score
def db_score(x, yhat, path, name, write_mode='w'):
    write_results(f'davies_bouldin_score ({name}): ' + str(davies_bouldin_score(x, yhat))+"\n", path, write_mode)


# silhouette score 
def silh_score(x, yhat, path, name, write_mode='w'):
    write_results(f'silhouette ({name}): ' + str(silhouette_score(x, yhat))+"\n", path, write_mode)

