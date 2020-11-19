import numpy as np
from sgl import LearnGraphTopolgy
from metrics import ModelSelection, Metrics
import networkx as nx
import matplotlib.pyplot as plt

with open('data/animals.txt', 'r') as f:
    data = [[int(num) for num in line.split(',')] for line in f]
data = np.asarray(data).T
n = data.shape[0]
p = data.shape[1]
with open('data/animals_names.txt', 'r') as f:
    names = [[str(num) for num in line.split(',')] for line in f]
names = names[0]

cov_emp = (data.T @ data)/ n

sgl = LearnGraphTopolgy(cov_emp + (1/3)*np.eye(p), maxiter=400, record_objective = True, record_weights = True)
graph = sgl.learn_k_component_graph(w0 = 'qp', k=5, beta=1)
# build network
A = graph['adjacency']
G = nx.from_numpy_matrix(A)
print('Graph statistics:')
print('Nodes: ', G.number_of_nodes(), 'Edges: ', G.number_of_edges() )

# normalize edge weights to plot edges strength
all_weights = []
for (node1,node2,data) in G.edges(data=True):
    all_weights.append(data['weight'])
max_weight = max(all_weights)
norm_weights = [3* w / max_weight for w in all_weights]
norm_weights = norm_weights

labeldict = {}
for i, name in enumerate(names):
    labeldict[i] = name

pos = nx.spring_layout(G)
nx.draw(G, labels=labeldict, pos=pos, width=norm_weights, with_labels = True)
plt.show()