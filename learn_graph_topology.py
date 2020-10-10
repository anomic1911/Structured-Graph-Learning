import numpy as np
import math
import time
from utils import L, Linv, Lstar, A
from optimizer import *
from tqdm import tqdm
from objective_function import objective, negloglikelihood, prior
# import cvxopt

def learn_k_component_graph(S, k=1, is_data_matrix=False, w0='naive', rho=1e-2, verbose=False, alpha=0,
beta=1e4, fix_beta=True, beta_max = 1e6, lb=0, ub=1e4, maxiter=1000, abstol = 1e-6, reltol = 1e-4,
eigtol = 1e-9, eps = 1e-4, record_objective = False, record_weights = False): 
    # number of nodes
    n = S.shape[0]
    # find an appropriate inital guess
    if is_data_matrix:
        raise Exception('Not implemented yet!')
    else:
        Sinv = np.linalg.pinv(S)
    # if w0 is either "naive" or "qp", compute it, else return w0
    w0 = w_init(w0, Sinv)
    # compute quantities on the initial guess
    Lw0 = L(w0)
    # l1-norm penalty factor
    H = alpha * (np.zeros((n,n)) - np.ones((n, n)))
    K = S + H
    U0 = U_update(Lw = Lw0, k = k)
    lamda0 = lamda_update(lb = lb, ub = ub, beta = beta, U = U0, Lw = Lw0, k = k)
    # save objective function value at initial guess
    if record_objective:
        ll0 = negloglikelihood(Lw = Lw0, lamda = lamda0, K = K)
        fun0 = ll0 + prior(beta = beta, Lw = Lw0, lamda = lamda0, U = U0)
        fun_seq = [fun0]
        ll_seq = [ll0]

    beta_seq = [beta]
    if record_weights:
        w_seq = [w0]
    time_seq = [0]
    # if (verbose)
    #     pb = progress::progress_bar$new(format = "<:bar> :current/:total  eta: :eta  beta: :beta  kth_eigval: :kth_eigval relerr: :relerr",
    #                                     total = maxiter, clear = FALSE, width = 120)
    start_time = time.time()
    for _ in tqdm(range(maxiter)):
        w = w_update(w = w0, Lw = Lw0, U = U0, beta = beta, lamda = lamda0, K = K)
        Lw = L(w)
        U = U_update(Lw = Lw, k = k)
        lamda = lamda_update(lb = lb, ub = ub, beta = beta, U = U, Lw = Lw, k = k)
        
        # compute negloglikelihood and objective function values
        if record_objective:
            ll = negloglikelihood(Lw = Lw, lamda = lamda, K = K)
            fun = ll + prior(beta = beta, Lw = Lw, lamda = lamda, U = U)
            ll_seq.append(ll)
            fun_seq.append(fun)
        if record_weights:
            w_seq.append(w)
        
        # check for convergence
        werr = np.abs(w0 - w)
        has_w_converged = all(werr <= .5 * reltol * (w + w0)) or all(werr <= abstol)
        time_seq.append( time.time() - start_time )
        if not fix_beta or verbose:
            eigvals, _ = np.linalg.eigh(Lw)
        
        # if verbose:
        #     pb$tick(token = list(beta = beta, kth_eigval = eigvals[k],
        #     relerr = 2 * max(werr / (w + w0), na.rm = 'ignore')))
        # }
        
        if not fix_beta:
            n_zero_eigenvalues = np.sum(np.abs(eigvals) < eigtol)
            if k <= n_zero_eigenvalues:
                beta = (1 + rho) * beta
            elif k > n_zero_eigenvalues:
                beta = beta / (1 + rho)
            if beta > beta_max:
                beta = beta_max
            beta_seq.append(beta)
        
        if has_w_converged:
            break
        
        # update estimates
        w0 = w
        U0 = U
        lamda0 = lamda
        Lw0 = Lw
        K = S + H / (-Lw + eps)
    
    # compute the adjancency matrix
    Aw = A(w)
    results = {'laplacian' : Lw, 'adjacency' : Aw, 'w' : w, 'lamda' : lamda, 'U' : U,
    'elapsed_time' : time_seq, 'convergence' : has_w_converged, 'beta_seq' : beta_seq }
    if record_objective:
        results['obj_fun'] = fun_seq
        results['negloglike'] = ll_seq

    if record_weights:
        results['w_seq'] = w_seq

    return results