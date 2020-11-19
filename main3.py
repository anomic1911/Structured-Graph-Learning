import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import make_sparse_spd_matrix
from sgl import LearnGraphTopolgy
from metrics import ModelSelection, Metrics
from utils import Operators
from scipy.linalg import block_diag

plots_dir = './plots'
outs_dir = './outs'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
if not os.path.exists(outs_dir):
    os.makedirs(outs_dir)
np.random.seed(3)

def generate_bipartite_data(n1, n2, n_samples):
    n_features = n1+n2
    W = np.random.rand(n1, n2)
    # uncomment for unweighted graph
    # W = np.ones((n1, n2))
    B = np.ones((n1, n2)) * W
    A = np.hstack((np.zeros((n1, n1)), B))
    A_ = np.hstack((B.T, np.zeros((n2, n2))))
    A_true = np.vstack((A, A_))
    L_true = np.diag(np.sum(A_true, axis=1)) - A_true
    # print(L_true, A_true)
    cov_true = np.linalg.pinv(L_true)
    # sample from GMRF
    X = np.random.multivariate_normal(np.zeros(n_features), cov_true, size=n_samples)
    # X -= X.mean(axis=0)
    # X /= X.std(axis=0)
    # plot laplacian
    fig = plt.figure(figsize=(15,15)) 
    plt.title('True Laplacian')
    plt.set_cmap('Blues')
    plt.imshow(L_true)
    plt.colorbar()
    filename = 'plots/bipartite_true_laplacian.png'
    fig.savefig(filename, format='png')
    plt.close()
    # plot adjacency
    fig = plt.figure(figsize=(15,15)) 
    plt.title('True Adjacency')
    plt.imshow(A_true)
    plt.colorbar()
    filename = 'plots/bipartite_true_adj.png'
    fig.savefig(filename, format='png')
    plt.close()
    return X, L_true, cov_true

def empirical_estimate(X, n_samples, plot=True):
    ''' Empirical estimation '''
    print('##########  Empirical Estimation  ##########')
    # Sample Covariance matrix
    cov_emp = np.dot(X.T, X) / n_samples
    prec_emp = np.linalg.pinv(cov_emp)
    A = np.diag(np.diag(prec_emp)) - prec_emp
    # uncomment for thresholding in unweighted graph
    # A[A>eps] = 1
    # A[A<eps] = 0
    # prec_emp = np.diag(np.sum(A, axis=1)) - A
    metric = Metrics(L_true, prec_emp)
    print('Rel error:', metric.relative_error())
    print('F1 score:', metric.f1_score())

    if plot:
        fig = plt.figure(figsize=(15,15)) 
        plt.title('Estimated Laplacian empirical')
        plt.imshow(prec_emp)
        plt.colorbar()
        filename = 'plots/bipartite_estimated_Laplacian_empirical.png'
        fig.savefig(filename, format='png')
        plt.close()

        fig = plt.figure(figsize=(15,15)) 
        A = np.diag(np.diag(prec_emp)) - prec_emp
        plt.title('Estimated Adjacency empirical')
        plt.imshow(A)
        plt.colorbar()
        filename = 'plots/bipartite_estimated_adj_empirical.png'
        fig.savefig(filename, format='png')
        plt.close()
    return prec_emp, cov_emp

n = 1600
p1 = 10
p2 = 6
p = p1+p2
X, L_true, cov_true = generate_bipartite_data(p1, p2, n)
L_emp, cov_emp = empirical_estimate(X, n)
# check for bipartite graph
print('##########  Assumed Graph structure: connected bipartite graph  ##########')
sgl = LearnGraphTopolgy(cov_emp, maxiter=5000, record_objective = True, record_weights = True)
graph = sgl.learn_bipartite_graph(w0 = 'qp', z = 4, nu=1e4)
A_sga = graph['adjacency']
eps = 1e-3
A_sga[A_sga<eps] = 0
L_sga = graph['laplacian']
# plot laplacian
fig = plt.figure(figsize=(15,15)) 
plt.title('Estimated Laplacian Bipartite')
plt.imshow(L_sga)
plt.colorbar()
filename = 'plots/bipartite_estimated_Laplacian.png'
fig.savefig(filename, format='png')
plt.close()
# plot adjacency
fig = plt.figure(figsize=(15,15)) 
plt.title('Estimated Adjacency Bipartite')
plt.imshow(A_sga)
plt.colorbar()
filename = 'plots/bipartite_estimated_adj.png'
fig.savefig(filename, format='png')
plt.close()

mod_selection = ModelSelection()
ebic = mod_selection.ebic(L_sga, cov_emp, n, p)
metrics = Metrics(L_true, L_sga)
print('train objective:', min(graph['obj_fun']), 'train NLL:', min(graph['nll']) )
print('Rel error: {} F1 score: {}'.format(metrics.relative_error(), metrics.f1_score()))
print('eBIC score:', ebic)


# def SGL_EBIC(cov_emp, K = 7, plot=True):
#     ''' SGL + EBIC '''
#     eps = 1e-4
#     precs = []
#     adjs = []
#     ebics = []
#     m = ModelSelection()
#     sgl = LearnGraphTopolgy(cov_emp, maxiter=5000, record_objective = True, record_weights = True)
#     # check for k-component graph
#     print('##########  Assumed Graph structure: k-component graph  ##########')
#     if K < 1:
#         raise Exception('Increase k or number of components')
#     for k in range(1, K+1):
#         print('===> k =', k)
#         # estimate underlying graph
#         graph = sgl.learn_k_component_graph(k=k, beta=1e4)
#         L = graph['laplacian']
#         # thresholding
#         A = np.diag(np.diag(L)) - L
#         A[A>eps] = 1
#         A[A<eps] = 0
#         adjs.append(A)
#         L = np.diag(np.sum(A, axis=1)) - A
#         precs.append(L)
#         metric = Metrics(prec_true, L)
#         ebic = m.ebic(L, cov_emp, n_samples, n_features)
#         ebics.append(ebic)
#         print('train objective:', min(graph['obj_fun']), 'train NLL:', min(graph['nll']) )
#         print('Rel error: {} F1 score: {}'.format(metric.relative_error(), metric.f1_score()))
#         print('eBIC score:', ebic)

#     # check for bipartite graph
#     print('##########  Assumed Graph structure: connected bipartite graph  ##########')
#     graph = sgl.learn_bipartite_graph(z = 4, nu=1e4)
#     A = graph['adjacency']
#     A[A>eps] = 1
#     A[A<eps] = 0
#     adjs.append(A)
#     L = np.diag(np.sum(A, axis=1)) - A
#     precs.append(L)
#     metric = Metrics(prec_true, L)
#     ebic = m.ebic(L, cov_emp, n_samples, n_features)
#     ebics.append(ebic)
#     print('train objective:', min(graph['obj_fun']), 'train NLL:', min(graph['nll']) )
#     print('Rel error: {} F1 score: {}'.format(metric.relative_error(), metric.f1_score()))
#     print('eBIC score:', ebic)
#     # check for multi-component bipartite graph
#     print('##########  Assumed Graph structure: multi-component bipartite graph  ##########')


#     if plot:
#         # plot k-component graphs
#         for i in range(K):
#             fig = plt.figure(figsize=(15,15)) 
#             L = precs[i]
#             plt.title('Estimated Laplacian k=' + str(i+1))
#             plt.imshow(L)
#             plt.colorbar()
#             filename = 'plots/estimated_Laplacian_k=' + str(i+1) + '.png'
#             fig.savefig(filename, format='png')
#             plt.close()

#             fig = plt.figure(figsize=(15,15)) 
#             A = adjs[i]
#             plt.title('Estimated Adjacency k=' + str(i+1))
#             plt.imshow(A)
#             plt.colorbar()
#             filename = 'plots/estimated_adj_k=' + str(i+1) + '.png'
#             fig.savefig(filename, format='png')
#             plt.close()
#         # plot bipartite graph
#         fig = plt.figure(figsize=(15,15)) 
#         L = precs[K]
#         plt.title('Estimated Laplacian Bipartite')
#         plt.imshow(L)
#         plt.colorbar()
#         filename = 'plots/estimated_Laplacian_bipartite.png'
#         fig.savefig(filename, format='png')
#         plt.close()

#         fig = plt.figure(figsize=(15,15)) 
#         A = adjs[K]
#         plt.title('Estimated Adjacency Bipartite')
#         plt.imshow(A)
#         plt.colorbar()
#         filename = 'plots/estimated_adj_bipartite.png'
#         fig.savefig(filename, format='png')
#         plt.close()
#         # plot multi-component graphs

#     # save precision matrices and corresponding ebic scores
#     precs, ebics = np.asarray(precs), np.asarray(ebics)
#     with open('outs/outs.npy', 'wb') as f:
#         np.save(f, precs)
#         np.save(f, ebics)

# if __name__ == "__main__":
#     # actual graph bipartite example
#     n_samples = 200
#     n1 = 10
#     n2 = 6
#     n_features = n1+n2
#     X, prec_true, cov_true = generate_bipartite_data(n1, n2, n_samples)
#     prec_emp, cov_emp = empirical_estimate(X, n_samples)
#     SGL_EBIC(cov_emp, K=8)


    # with open('test.npy', 'rb') as f:
    #     precs = np.load(f)
    #     ebics = np.load(f)
    # k_ebic = ebics.index(max(ebics))
    # precs[k_ebic], np.linalg.pinv(precs[k_ebic]), k_ebic + 1