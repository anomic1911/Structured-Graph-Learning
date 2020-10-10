import numpy as np

def bic(ll, n, k):
    """Bayesian information criterion

    """
    return -2 * ll + k * np.log(n)

def aic(ll, n, k):
    """Akaike information criterion

    """
    return -2 * ll + 2 * k

def aicc(ll, n, k):
    """Akaike information criterion corrected

    """
    return aic(ll,n, k) + 2*k*(k+1)/(n-k-1)