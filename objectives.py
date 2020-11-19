import numpy as np
import math

class Objectives:
    def objective(self, Lw, U, lamda, K, beta):
        return self.negloglikelihood(Lw, lamda, K) + self.prior(beta, Lw, lamda, U)

    def negloglikelihood(self, Lw, lamda, K):
        return np.sum(-np.log(lamda)) + np.trace(K @ Lw)

    def prior(self, beta, Lw, lamda, U):
        return 0.5 * beta * np.linalg.norm(Lw - U @ np.diag(lamda) @ U.T)**2

    def bipartite_nll(self, Lw, K, J):
        return np.sum(-np.log(np.linalg.eigvals(Lw + J)) + np.diag(K @ Lw))

    def bipartite_prior(self, nu, Aw, psi, V):
        return 0.5 * nu * np.linalg.norm(Aw - V @ np.diag(psi) @ V.T) ** 2
    
    def bipartite_obj(self, Aw, Lw, V, psi, K, J, nu):
        return self.bipartite_nll(Lw = Lw, K = K, J = J) + self.bipartite_prior(nu = nu, Aw = Aw, psi = psi, V = V)
