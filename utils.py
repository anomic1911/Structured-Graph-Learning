import numpy as np
import math

class Operators:
    def L(self, w):
        '''Usage: print(L([1,0,1]))
        
        '''
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

    def vecLmat(self, n):
        ncols = int(0.5*n*(n-1))
        nrows = int(n*n)
        R = np.zeros((nrows, ncols))
        e = np.zeros(ncols)
        e[0] = 1
        R[:, 0] = self.L(e).flatten()
        for j in range(1,ncols):
            e[j-1] = 0
            e[j] = 1
            R[:, j] = self.L(e).flatten()
        return R

    def Linv(self, M):
        '''Computes the inverse of the L operator
        Parameters
        ----------
        M : Laplacian matrix
        
        Returns
        -------
        w : the weight vector of the graph
        
        Usage
        -------
        a = [[1, -1, 0], [-1, 2, -1], [0, -1, 1]]
        Linv(np.array(a))
        '''
        N = M.shape[0]
        k = int( 0.5*N*(N-1))
        l = 0
        w = np.zeros(k)
        for i in range(0,N):
            for j in range(i+1, N):
                w[l] = -M[i][j]
                l = l+1
        return w

    def A(self, w):
        '''Computes the Adjacency linear operator which maps a vector of weights into a valid Adjacency matrix.
        
        Parameters
        ----------
        w : weight vector of the graph
        
        Returns
        -------
        Aw : the Adjacency matrix
        
        Usage
        -------
        A(np.array([1,0,1]))
        '''
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
    
    def Lstar(self, M):
        N = M.shape[1]
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

    def Astar(self, M):
        N = M.shape[1]
        k = int( 0.5*N*(N-1))
        w = np.zeros(k)
        j=0
        l=1

        for i in range(0,k):
            w[i] = M[l][j] + M[j][l]
            if l==(N-1):
                j = j+1
                l = j+1
            else:
                l = l+1
        return w