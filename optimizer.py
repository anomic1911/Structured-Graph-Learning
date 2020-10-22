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
        temp = np.zeros((lamda.shape[0],lamda.shape[0]))
        np.fill_diagonal(temp, lamda)
        c = self.op.Lstar(np.matmul(np.matmul(U , temp), U.T)  - K / beta)
        grad_f = self.op.Lstar(Lw) - c
        M_grad_f = self.op.Lstar(self.op.L(grad_f))
        wT_M_grad_f = sum(w * M_grad_f)
        dwT_M_dw = sum(grad_f * M_grad_f)
        # exact line search
        t = (wT_M_grad_f - sum(c * grad_f)) / dwT_M_dw
        w_update = w - t * grad_f
        w_update[w_update < 0] = 0
        return(w_update)

    def U_update(self, Lw, k):
        _, eigvec = np.linalg.eigh(Lw)
        p = Lw.shape[1]
        return eigvec[:, k:p]

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