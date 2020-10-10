import numpy as np
import math

'''Usage: print(L([1,0,1]))
'''
def L(w):
    k = len(w)
    n = int(0.5*(1+ math.sqrt(1+8*k)))
    Lw = np.zeros((n,n))
    k=0
    for i in range(0, n):
        for j in range(i+1,n):
            Lw[i][j] = -w[k]
            k = k + 1 
    Lw = Lw + Lw.T
    row,col = np.diag_indices_from(Lw)
    Lw[row,col] = -Lw.sum(axis=1)
    return Lw

# def vecLmat(n):
#     ncols = 0.5*n*(n-1)
#     nrows = n*n

'''Computes the inverse of the L operator.
    @param M Laplacian matrix
    @return w the weight vector of the graph
    Usage: 
    a = [[1, -1, 0], [-1, 2, -1], [0, -1, 1]]
    print(Linv(np.array(a)))
'''
def Linv(M):
    N = M.shape[0]
    k = int( 0.5*N*(N-1))
    l = 0
    w = np.zeros(k)
    for i in range(0,N):
        for j in range(i+1, N):
            w[l] = -M[i][j]
            l = l+1
    return w

def Lstar(M):
    N = M.shape[0]
    k = int( 0.5*N*(N-1))
    w = np.zeros(k)
    j=0
    l=1
    for i in range(0,k):
        w[i] = M[j][j] + M[l][l] -(M[l][j] + M[j][l])
        if l==(N-1):
            j = j+1
            l = j+1
        else:
            l = l+1
    return w

'''Computes the Adjacency linear operator which maps a vector of weights into a valid Adjacency matrix.
    @param w weight vector of the graph
    @return Aw the Adjacency matrix
    Usage: print(A(np.array([1,0,1])))
'''
def A(w):
    k = w.shape[0]
    n = int(0.5 * (1 + math.sqrt(1 + 8 * k)))
    Aw = np.zeros((n,n))
    k=0
    for i in range(0, n):
        for j in range(i+1,n):
            Aw[i][j] = w[k]
            k = k + 1 
    Aw = Aw + Aw.T
    return Aw