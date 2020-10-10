import numpy as np
import math

def objective(Lw, U, lamda, K, beta):
    return negloglikelihood(Lw, lamda, K) + prior(beta, Lw, lamda, U)

def negloglikelihood(Lw, lamda, K):
    return(np.sum(-np.log(lamda)) + np.trace(np.dot(K, Lw)))

def prior(beta, Lw, lamda, U):
    temp = np.zeros((lamda.shape[0],lamda.shape[0]))
    np.fill_diagonal(temp, lamda)
    return 0.5 * beta * np.linalg.norm(Lw - np.matmul(np.matmul(U,temp),U.T))**2