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
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def generate_data(n_samples, p, k_true):
    '''Makes k-component data and generate samples from GMRF
    
    Parameters
    ----------
    n_samples : int Number of samples 

    p : int Number of nodes per component

    k_true : int Number of components

    Returns
    -------
    X : data 2D ndarray (n_samples ⨉ n_features)

    cov_true : true covariance matrix  2D ndarray (n_features ⨉ n_features)

    prec_true : true precision matrix  2D ndarray (n_features ⨉ n_features)
    '''
    n_features = p*k_true
    a = int((p*(p-1))/2)
    # Assume each component fully connected
    w = np.ones(a)
    # uncomment to specify edge weights
    # w = np.ones(a) * np.random.rand(a,1)
    op = Operators()
    lw = op.L(w)
    prec_true = block_diag(*[lw]*k_true)
    cov_true = np.linalg.pinv(prec_true)
    A_true = np.diag(np.diag(prec_true)) - prec_true
    # normalization
    d = np.sqrt(np.diag(cov_true))
    cov_true /= d
    cov_true /= d[:, np.newaxis]
    prec_true *= d
    prec_true *= d[:, np.newaxis]
    # sample from GMRF
    prng = np.random.RandomState(3)
    X = prng.multivariate_normal(np.zeros(n_features), cov_true, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    # plot laplacian
    fig = plt.figure(figsize=(15,15)) 
    plt.title('True Laplacian')
    plt.imshow(prec_true)
    plt.colorbar()
    filename = 'plots/true_laplacian.png'
    fig.savefig(filename, format='png')
    # plot adjacency
    fig = plt.figure(figsize=(15,15)) 
    plt.title('True Adjacency')
    plt.imshow(A_true)
    plt.colorbar()
    filename = 'plots/true_adj.png'
    fig.savefig(filename, format='png')
    return X, prec_true, cov_true

def empirical_estimate(X, n_samples, plot=True):
    ''' Empirical estimation '''
    print('##########  Empirical Estimation  ##########')
    eps = 1e-4
    # Sample Covariance matrix
    cov_emp = np.dot(X.T, X) / n_samples
    prec_emp = np.linalg.pinv(cov_emp)

    metric = Metrics(prec_true, prec_emp)
    print('Rel error:', metric.relative_error())
    print('F1 score:', metric.f1_score())

    if plot:
        fig = plt.figure(figsize=(15,15)) 
        plt.title('Estimated Laplacian empirical')
        plt.imshow(prec_emp)
        plt.colorbar()
        filename = 'plots/estimated_Laplacian_empirical.png'
        fig.savefig(filename, format='png')

        fig = plt.figure(figsize=(15,15)) 
        A = np.diag(np.diag(prec_emp)) - prec_emp
        plt.title('Estimated Adjacency empirical')
        plt.imshow(A)
        plt.colorbar()
        filename = 'plots/estimated_adj_empirical.png'
        fig.savefig(filename, format='png')
    return prec_emp, cov_emp


def SGL_EBIC(cov_emp, K = 7, plot=True):
    ''' SGL + EBIC '''
    if K < 1:
        raise Exception('Increase K')
    eps = 1e-4
    sgl = LearnGraphTopolgy(cov_emp, maxiter=10000, record_objective = True, record_weights = True)
    m = ModelSelection()
    precs = []
    graphs = []
    ebic_scores = []
    op = Operators()
    for k in range(1, K):
        print('########## k =', k, '##########')
        # estimate underlying graph
        graph = sgl.learn_k_component_graph(k=k, beta=1e4)
        graphs.append(graph)
        w = op.Linv(graph['laplacian'])
        w[w>eps] = 1
        L = op.L(w)
        precs.append(L)
        metric = Metrics(prec_true, L)
        dof = n_features*(n_features-k)/(2*k)
        ebic_score = m.ebic(L, cov_emp, dof, n_samples, n_features)
        ebic_scores.append(ebic_score)
        print('train obj:', min(graph['obj_fun']), 'train nll:', min(graph['negloglike']) )
        print('Rel error:', metric.relative_error())
        print('F1 score:', metric.f1_score())
        print('eBIC score:', ebic_score)

    if plot:
        for i, graph in enumerate(graphs):
            fig = plt.figure(figsize=(15,15)) 
            w = op.Linv(graph['laplacian'])
            w[w>eps] = 1
            L = op.L(w)
            plt.title('Estimated Laplacian k=' + str(i+1))
            plt.imshow(L)
            plt.colorbar()
            filename = 'plots/estimated_Laplacian_k=' + str(i+1) + '.png'
            fig.savefig(filename, format='png')
            
            fig = plt.figure(figsize=(15,15)) 
            A = np.diag(np.diag(L)) - L
            plt.title('Estimated Adjacency k=' + str(i+1))
            plt.imshow(A)
            plt.colorbar()
            filename = 'plots/estimated_adj_k=' + str(i+1) + '.png'
            fig.savefig(filename, format='png')
    # find index of maxmimum EBIC
    k_ebic = ebic_scores.index(max(ebic_scores))
    return  precs[k_ebic], np.linalg.pinv(precs[k_ebic]), k_ebic + 1

if __name__ == "__main__":
    n_samples = 200
    p = 10
    k_true = 3
    n_features = p*k_true
    X, prec_true, cov_true = generate_data(n_samples, p, k_true)
    prec_emp, cov_emp = empirical_estimate(X, n_samples)
    prec_ebic, cov_ebic, k_ebic = SGL_EBIC(cov_emp, K=7)