import numpy as np
import math

class Objectives:
    def objective(self, Lw, U, lamda, K, beta):
        return self.negloglikelihood(Lw, lamda, K) + self.prior(beta, Lw, lamda, U)

    def negloglikelihood(self, Lw, lamda, K):
        return(np.sum(-np.log(lamda)) + np.trace(np.dot(K, Lw)))

    def prior(self, beta, Lw, lamda, U):
        temp = np.zeros((lamda.shape[0],lamda.shape[0]))
        np.fill_diagonal(temp, lamda)
        return 0.5 * beta * np.linalg.norm(Lw - np.matmul(np.matmul(U,temp),U.T))**2