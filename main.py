import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sgl import LearnGraphTopolgy

plots_dir = './plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

'''Visual results on two moon dataset 

'''
np.random.seed(0)
n = 50  # number of nodes per cluster
k = 2   # number of components
X, y = make_moons(n_samples=n*k, noise=.05, shuffle=True)
# X, y = make_blobs(n_samples=n*k, centers=k, n_features=2, random_state=0)
# dict to store position of nodes
pos = {}
for i in range(n*k):
    pos[i] = X[i]
# Visualization of original data
fig = plt.figure()
plt.scatter(X[:,0], X[:,1], c=y )
plt.title("Two moon dataset")
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
fig.savefig('plots/two_moon_dataset.eps', format='eps')
fig.savefig('plots/two_moon_dataset.png')

# compute sample correlation matrix
S = np.dot(X, X.T)

# estimate underlying graph
sgl = LearnGraphTopolgy(S, maxiter=1000, record_objective = True, record_weights = True)
graph = sgl.learn_k_component_graph(k=2, beta=0.1 )

nll = graph['negloglike']
print('NLL: ', min(nll))
objective = graph['obj_fun']
print('Objective: ', min(objective))

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

# plot graph
fig = plt.figure(figsize=(15,15)) 
nx.draw_networkx(G,pos, width=norm_weights)
plt.title("Learned graph for two moon dataset")
plt.suptitle('components k=2')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
filename = 'plots/learned_graph_k='+ str(k) +'.eps'
png_filename = 'plots/learned_graph_k='+ str(k) +'.png'
fig.savefig(filename, format='eps')
fig.savefig(png_filename,)