import numpy as np
import math
import time
from utils import Operators
from optimizer import Optimizer
from tqdm import tqdm
from objectives import Objectives

class LearnGraphTopolgy:
    def __init__(self, S, is_data_matrix=False, alpha=0, maxiter=10000, abstol = 1e-6, reltol = 1e-4,
    record_objective = False, record_weights = False):
        self.S = S
        self.is_data_matrix = is_data_matrix
        self.alpha = alpha
        self.maxiter = maxiter
        self.abstol = abstol
        self.reltol = reltol
        self.record_objective = record_objective
        self.record_weights = record_weights
        self.op = Operators()
        self.obj = Objectives()
        self.optimizer = Optimizer()
        self.bic = 0

    def learn_k_component_graph(self, k=1, rho=1e-2, beta=1e4, w0='naive', fix_beta=True, beta_max = 1e6,
    lb=0, ub=1e10, eigtol = 1e-9, eps = 1e-4):
        # number of nodes
        n = self.S.shape[0]
        # find an appropriate inital guess
        if self.is_data_matrix or self.S.shape[0] != self.S.shape[1]:
            raise Exception('Not implemented yet!')
        else:
            Sinv = np.linalg.pinv(self.S)
        # if w0 is either "naive" or "qp", compute it, else return w0
        w0 = self.optimizer.w_init(w0, Sinv)
        # compute quantities on the initial guess
        Lw0 = self.op.L(w0)
        # l1-norm penalty factor
        H = self.alpha * (np.eye(n) - np.ones((n, n)))
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
            results['nll'] = nll_seq
            results['bic'] = 0

        if self.record_weights:
            results['w_seq'] = w_seq
        return results
    
    def learn_bipartite_graph(self, z = 0, nu = 1e4, m=7, w0='naive'):
        # number of nodes
        n = self.S.shape[0]
        # find an appropriate inital guess
        if self.is_data_matrix or self.S.shape[0] != self.S.shape[1]:
            raise Exception('Not implemented yet!')
        else:
            Sinv = np.linalg.pinv(self.S)
        # note now that S is always some sort of similarity matrix
        J = np.ones((n, n))*(1/n)
        # l1-norm penalty factor
        H = self.alpha * (2*np.eye(n) - np.ones((n, n)))
        K = self.S + H
        # if w0 is either "naive" or "qp", compute it, else return w0
        w0 = self.optimizer.w_init(w0, Sinv)
        Lips = 1 / min(np.linalg.eigvals(self.op.L(w0) + J)) 
        # compute quantities on the initial guess
        Aw0 = self.op.A(w0)
        V0 = self.optimizer.V_update(Aw0, z)
        psi0 = self.optimizer.psi_update(V0, Aw0)
        Lips_seq = [Lips]
        time_seq = [0]
        start_time = time.time()
        ll0 = self.obj.bipartite_nll(Lw = self.op.L(w0), K = K, J = J)
        fun0 = ll0 + self.obj.bipartite_prior(nu = nu, Aw = Aw0, psi = psi0, V = V0)
        fun_seq = [fun0]
        nll_seq = [ll0]
        if self.record_weights:
            w_seq = [w0]
        for _ in tqdm(range(self.maxiter)):
            # we need to make sure that the Lipschitz constant is large enough
            # in order to avoid divergence
            while(1):
                # compute the update for w
                w = self.optimizer.bipartite_w_update(w = w0, Aw = Aw0, V = V0, nu = nu, psi = psi0,
                K = K, J = J, Lips = Lips)
                # compute the objective function at the updated value of w
                fun_t = self.obj.bipartite_obj(Aw = self.op.A(w), Lw = self.op.L(w), V = V0, psi = psi0,
                K = K, J = J, nu = nu)
                # check if the previous value of the objective function is
                # smaller than the current one
                Lips_seq.append(Lips)
                if fun0 < fun_t:
                    # in case it is in fact larger, then increase Lips and recompute w
                    Lips = 2 * Lips
                else:
                    # otherwise decrease Lips and get outta here!
                    Lips = .5 * Lips
                    if Lips < 1e-12:
                        Lips = 1e-12
                    break
            Lw = self.op.L(w)
            Aw = self.op.A(w)
            V = self.optimizer.V_update(Aw = Aw, z = z)
            psi = self.optimizer.psi_update(V = V, Aw = Aw)
            # compute negloglikelihood and objective function values
            ll = self.obj.bipartite_nll(Lw = Lw, K = K, J = J)
            fun = ll + self.obj.bipartite_prior(nu = nu, Aw = Aw, psi = psi, V = V)
            # save measurements of time and objective functions
            time_seq.append(time.time() - start_time)
            nll_seq.append(ll)
            fun_seq.append(fun)
            # compute the relative error and check the tolerance on the Adjacency
            # matrix and on the objective function
            if self.record_weights:
                w_seq.append(w)
             # check for convergence
            werr = np.abs(w0 - w)
            has_w_converged = all(werr <= .5 * self.reltol * (w + w0)) or all(werr <= self.abstol)
            
            if has_w_converged:
                break
            # update estimates
            fun0 = fun
            w0 = w
            V0 = V
            psi0 = psi
            Aw0 = Aw

        results = {'laplacian' : Lw, 'adjacency' : Aw, 'w' : w, 'psi' : psi, 'V' : V,
        'elapsed_time' : time_seq, 'Lips_seq' : Lips_seq, 'convergence' : has_w_converged, 'nu' : nu }
        
        if self.record_objective:
            results['obj_fun'] = fun_seq
            results['nll'] = nll_seq
            results['bic'] = 0

        if self.record_weights:
            results['w_seq'] = w_seq

        return results
       
    # def learn_bipartite_k_component_graph <- function(S, is_data_matrix = FALSE, z = 0, k = 1,
    #                                           w0 = "naive", m = 7, alpha = 0., beta = 1e4,
    #                                           rho = 1e-2, fix_beta = TRUE, beta_max = 1e6, nu = 1e4,
    #                                           lb = 0, ub = 1e4, maxiter = 1e4, abstol = 1e-6,
    #                                           reltol = 1e-4, eigtol = 1e-9,
    #                                           record_weights = FALSE, record_objective = FALSE, verbose = TRUE)