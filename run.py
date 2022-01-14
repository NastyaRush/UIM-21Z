from project_name.visualization import graph, evaluate_algorithm
from project_name.data import make_dataset
from project_name.models import affinity_propagation, birch
from pathlib import Path
import os
import sys




def main():
    make_dataset.main()
    a_yhat, a_clusters, a_data = affinity_propagation.affinity_propagation()
    b_yhat, b_clusters, b_data = birch.birch()
    reports_path = str(os.path.dirname(os.path.realpath(__file__)).replace("\\", "/"))+'/reports/'
    # print(reports_path+'affinity_propagation_graph.png')
    graph.graph(a_yhat, a_clusters, a_data, reports_path, 'affinity_propagation_graph.png')
    graph.graph(b_yhat, b_clusters, b_data, reports_path, 'birch_graph.png')
    evaluate_algorithm.silh_score(a_data, a_yhat, reports_path, "affinity_propagation", 'w')
    evaluate_algorithm.db_score(a_data, a_yhat, reports_path, "affinity_propagation", 'a')
    evaluate_algorithm.silh_score(b_data, b_yhat, reports_path, "birch", 'a')
    evaluate_algorithm.db_score(b_data, b_yhat, reports_path, "birch", 'a')



if __name__ == "__main__":
    main()