import pymetis
import numpy as np

adjacency_list = [np.array([1, 2]), np.array([0, 2]), np.array([0, 1])]

num_clusters = 2
a = pymetis.part_graph(num_clusters, adjacency=adjacency_list)
