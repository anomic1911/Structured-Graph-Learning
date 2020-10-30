import numpy as np
import math
import time
from utils import Operators
from optimizer import Optimizer
from tqdm import tqdm
from objectives import Objectives
# import cvxopt

class LearnGraphTopolgy:
    def __init__(self, S, is_data_matrix=False, alpha=0, w0='naive', maxiter=10000, abstol = 1e-6, reltol = 1e-4,
    record_objective = False, record_weights = False):
        self.S = S
        self.is_data_matrix = is_data_matrix
        self.alpha = alpha
        self.w0 = w0
        self.maxiter = maxiter
        self.abstol = abstol
        self.reltol = reltol
        self.record_objective = record_objective
        self.record_weights = record_weights
        self.op = Operators()
        self.optimizer = Optimizer()
        self.obj = Objectives()
        self.bic = 0

    def learn_k_component_graph(self, k=1, rho=1e-2, beta=1e4, fix_beta=True, beta_max = 1e6,
    lb=0, ub=1e10, eigtol = 1e-9, eps = 1e-4): 
        # number of nodes
        n = self.S.shape[0]
        # find an appropriate inital guess
        if self.is_data_matrix:
            raise Exception('Not implemented yet!')
        else:
            Sinv = np.linalg.pinv(self.S)
        # if w0 is either "naive" or "qp", compute it, else return w0
        w0 = self.optimizer.w_init(self.w0, Sinv)
        # compute quantities on the initial guess
        Lw0 = self.op.L(w0)
        # l1-norm penalty factor
        H = self.alpha * (np.zeros((n,n)) - np.ones((n, n)))
        K = self.S + H
        U0 = self.optimizer.U_update(Lw = Lw0, k = k)
        lamda0 = self.optimizer.lamda_update(lb = lb, ub = ub, beta = beta, U = U0, Lw = Lw0, k = k)
        # save objective function value at initial guess
        if self.record_objective:
            nll0 = self.obj.negloglikelihood(Lw = Lw0, lamda = lamda0, K = K)
            fun0 = nll0 + self.obj.prior(beta = beta, Lw = Lw0, lamda = lamda0, U = U0)
            fun_seq = [fun0]
            nll_seq = [nll0]

        beta_seq = [beta]
        if self.record_weights:
            w_seq = [w0]
        time_seq = [0]
        
        start_time = time.time()
        for _ in tqdm(range(self.maxiter)):
            w = self.optimizer.w_update(w = w0, Lw = Lw0, U = U0, beta = beta, lamda = lamda0, K = K)
            Lw = self.op.L(w)
            U = self.optimizer.U_update(Lw = Lw, k = k)
            lamda = self.optimizer.lamda_update(lb = lb, ub = ub, beta = beta, U = U, Lw = Lw, k = k)
            
            # compute negloglikelihood and objective function values
            if self.record_objective:
                nll = self.obj.negloglikelihood(Lw = Lw, lamda = lamda, K = K)
                fun = nll + self.obj.prior(beta = beta, Lw = Lw, lamda = lamda, U = U)
                nll_seq.append(nll)
                fun_seq.append(fun)
            if self.record_weights:
                w_seq.append(w)
            
            # check for convergence
            werr = np.abs(w0 - w)
            has_w_converged = all(werr <= .5 * self.reltol * (w + w0)) or all(werr <= self.abstol)
            time_seq.append( time.time() - start_time )
            if not fix_beta:
                eigvals, _ = np.linalg.eigh(Lw)
            
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
            K = self.S + H / (-Lw + eps)
        
        # compute the adjancency matrix
        Aw = self.op.A(w)
        results = {'laplacian' : Lw, 'adjacency' : Aw, 'w' : w, 'lamda' : lamda, 'U' : U,
        'elapsed_time' : time_seq, 'convergence' : has_w_converged, 'beta_seq' : beta_seq }
        if self.record_objective:
            results['obj_fun'] = fun_seq
            results['negloglike'] = nll_seq
            results['bic'] = 0

        if self.record_weights:
            results['w_seq'] = w_seq

        return results
    
    def learn_bipartite_graph(self, z = 0, nu = 1e4, m=7):
        # number of nodes
        n = self.S.shape[0]
        # find an appropriate inital guess
        if self.is_data_matrix:
            raise Exception('Not implemented yet!')
        else:
            Sinv = np.linalg.pinv(self.S)