import numpy as np
from sgl import LearnGraphTopolgy
from metrics import ModelSelection, Metrics
import networkx as nx
import matplotlib.pyplot as plt
import csv

with open('data/cancer/data.csv', 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    line_count = 0
    data = [[row[i] for i in range(len(row))] for row in csv_reader]
data = np.asarray(data)
data = data[1:, 1:]
data = data.astype('float')
n = data.shape[0]
p = data.shape[1]
print(data.shape)
with open('data/cancer/labels.csv', 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    names = [[str(row[i]) for i in range(len(row))] for row in csv_reader]
names = np.asarray(names)
names = names[1:, 1]

# cov_emp = (data.T @ data)/ n

sgl = LearnGraphTopolgy( data , maxiter=10, record_objective = False, record_weights = False)
graph = sgl.learn_k_component_graph(w0 = 'qp', k=5, beta=1e2)
# build network
A = graph['adjacency']
G = nx.from_numpy_matrix(A)
print('Graph statistics:')
print('Nodes: ', G.number_of_nodes(), 'Edges: ', G.number_of_edges() )