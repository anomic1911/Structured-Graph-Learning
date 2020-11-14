import numpy as np
import math
import time
from utils import Operators

class Optimizer:
    def __init__(self):
        self.op = Operators()

    def w_init(self, w0, Sinv):
        if w0.isalpha():
            if w0 == 'qp':
                raise Exception('Not implemented yet!')
            elif w0 == 'naive':
                w0 = self.op.Linv(Sinv)
                w0[w0<0] = 0
        return w0

    def w_update(self, w, Lw, U, beta, lamda, K):
        c = self.op.Lstar(np.matmul(np.matmul(U , np.diag(lamda)), U.T)  - K / beta)
        grad_f = self.op.Lstar(Lw) - c
        M_grad_f = self.op.Lstar(self.op.L(grad_f))
        wT_M_grad_f = sum(w * M_grad_f)
        dwT_M_dw = sum(grad_f * M_grad_f)
        # exact line search
        t = (wT_M_grad_f - sum(c * grad_f)) / dwT_M_dw
        w_update = w - t * grad_f
        w_update[w_update < 0] = 0
        return w_update

    def bipartite_w_update(self, w, Aw, V, nu, psi, K, J, Lips):
        grad_h = 2 * w - self.op.Astar(V @ np.diag(psi) @ V.T) #+ Lstar(K) / beta
        w_update = w - (self.op.Lstar(np.linalg.inv(self.op.L(w) + J) + K) + nu * grad_h) / (2 * nu + Lips)
        w_update[w_update < 0] = 0
        return w_update


    def U_update(self, Lw, k):
        _, eigvec = np.linalg.eigh(Lw)
        p = Lw.shape[1]
        return eigvec[:, k:p]
    
    def V_update(self, Aw, z):
        p = Aw.shape[1]
        _, V = np.linalg.eigh(Aw)
        return np.hstack((V[:, 0:int(.5*(p - z))], V[:, int(.5*(p + z)):p])) 

    def psi_update(self, V, Aw, lb = float('-inf'), ub = float('inf')):
        c = np.diag(V.T @ Aw @ V)
        n = c.shape[0]
        temp = c[int(n/2):n]
        temp = temp[::-1]
        c_tilde = .5 * (temp - c[0:int(n/2)])
        from sklearn.isotonic import IsotonicRegression
        iso_reg = IsotonicRegression().fit(list(range(1, c_tilde.shape[0]+1)) , c_tilde)
        x = iso_reg.predict(list(range(1, c_tilde.shape[0]+1)))
        x = np.concatenate((-x[::-1], x))
        x[x < lb] = lb
        x[x > ub] = ub
        return x

    def lamda_update(self, lb, ub, beta, U, Lw, k):
        q = Lw.shape[1] - k
        opt = np.matmul(np.matmul(U.T, Lw), U)
        d = np.diag(opt)
        # unconstrained solution as initial point
        lamda = 0.5 * (d + np.sqrt(d**2 + 4 / beta))
        eps = 1e-9
        condition = list((lamda[1:q-1] - lamda[0:(q-2)]) >= -eps)
        condition.append((lamda[q-1] - ub) <= eps)
        condition.append((lamda[0] - lb) >= -eps) 
        if all(condition):
            return (lamda)
        else:
            greater_ub = lamda > ub
            lesser_lb = lamda < lb
            lamda[greater_ub] = ub
            lamda[lesser_lb] = lb

        condition = list((lamda[1:q-1] - lamda[0:(q-2)]) >= -eps)
        condition.append((lamda[q-1] - ub) <= eps)
        condition.append((lamda[0] - lb) >= -eps) 
        if all(condition):
            return lamda
        else:
            print(lamda)
            raise Exception("eigenvalues are not in increasing order, consider increasing the value of beta")