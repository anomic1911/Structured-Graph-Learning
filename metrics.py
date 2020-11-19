import numpy as np

class Metrics:
    def __init__(self, Lw_true, Lw_est, eps=1e-4):
        self.Lw_true = Lw_true
        self.Lw_est = Lw_est
        self.eps = eps
        n = self.Lw_true.shape[0]
        self.tp, self.fp, self.fn, self.tn = 0, 0, 0, 0
        for i in range(0, n-1):
            for j in range(i+1, n):
                is_edge = abs(self.Lw_true[i][j]) > eps
                is_est_edge = abs(self.Lw_est[i][j]) > eps
                if is_edge and is_est_edge:
                    self.tp += 1
                elif not is_edge and is_est_edge:
                    self.fp += 1
                elif is_edge and not is_est_edge:
                    self.fn += 1
                else:
                    self.tn += 1

    def relative_error(self):
        """Relative Error between the final estimated laplacian and the true reference
        graph Laplacian matrix. 

        Parameters
        ----------
        Lw_true : true Laplacian
        Lw_est : estimated laplacian
        """
        return np.linalg.norm(self.Lw_true-self.Lw_est, ord=2) / np.linalg.norm(self.Lw_true, ord=2)
        
    def f1_score(self):
        return 2 * self.tp / (2 * self.tp + self.fn + self.fp)
    
    def recall(self):
        return self.tp / (self.tp + self.fn)
    
    def specificity(self):
        self.tn / (self.tn + self.fp)

    def accuracy(self):
        (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def negative_predictive_value(self):
        return (self.tn / (self.tn + self.fn))

    def false_discovery_rate(self):
        return (self.fp / (self.fp + self.tp))

class ModelSelection:
    def gdet(self, L):
        eps = 1e-4
        vals = np.linalg.eigvals(L)
        nz_eigvals = vals[vals > eps]
        return np.prod(nz_eigvals)

    def bic(self, L, S, n, p):
        """Bayesian information criterion
        eBIC(\theta) = log gdet(\theta) − Tr(S\theta) − |E|log n
        """
        A = np.diag(np.diag(L)) - L 
        num_edges = np.sum(A[A>0])/2
        return 2*np.log(self.gdet(L)) - np.trace(S*L) - num_edges * np.log(n)

    def ebic(self, L, S, n, p):
        """Extended Bayesian information criterion
        eBIC(\theta) = log gdet(\theta) − Tr(S\theta) − |E|log n − 4γ|E|log p
        """
        # if n/p > 500:
        #     gamma = 0
        # elif n/p > 100:
        #     gamma = 0.5
        # else:
        gamma = 0
        A = np.diag(np.diag(L)) - L 
        num_edges = np.sum(A[A>0])/2
        ll = np.log(self.gdet(L)) - np.trace(S @ L)
        print(2*ll, num_edges * np.log(n), num_edges)
        return 2*ll - num_edges * np.log(n) - 4*gamma*num_edges*np.log(p)

    # def aic(self, L, S, n, p):
    #     """Akaike information criterion
    #     eBIC(\theta) = log gdet(\theta) − Tr(S\theta) − |E|log n − 4γ|E|log p
    #     """
    #     return -2 * ll + 2 * k

    # def aicc(self, ll, n, k):
    #     """Akaike information criterion corrected

    #     """
    #     return self.aic(ll,n, k) + 2*k*(k+1)/(n-k-1)