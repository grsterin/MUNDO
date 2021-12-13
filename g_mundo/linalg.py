import numpy as np
import networkx as nx
from numpy.linalg import inv
from scipy.linalg import eig, eigh
from typing import Dict, List, NewType, Tuple, Set
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel

ndarray = NewType('numpy ndarray', np.ndarray)

def rkhs(H: ndarray)-> ndarray:
    """
    Given a matrix $A$, perform the decomposition $A = X D X^T$ 
    Resulting in the final matrix $X D^{1/2}$.
    """
    def is_symmetric(A: ndarray, rtol: float=1e-05, atol: float=1e-08)->bool:
        """
        Checks if the matrix is symmetric or not.
        """
        return np.allclose(A, A.T, rtol=rtol, atol=atol)

    if not is_symmetric(A):
        print('[!] Cannot embed asymmetric kernel into RKHS. Closing...')
        exit()        
    eigenvalues, eigenvectors = eig(H)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues))


def best_gamma_kernel(distance_matrix: ndarray)-> ndarray:
    """
    Find the best r for which the laplacian kernel produces the best
    affinity matrix.
    """
    r = None
    for i in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        rbf = laplacian_kernel(distance_matrix, gamma=i)
        n, r = rbf.shape[0], rbf.copy()
        if n * 1000 <= np.count_nonzero(rbf):
            return rbf
    return r


def turbo_dsd(adjacency: ndarray, nRw: int):
    """
    Function that takes in adjacency matrix, and return the 
    resulting Turbo DSD matrix from that.
    """
    adjacency = np.asmatrix(adjacency)
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    p = adjacency / degree
    if nRw >= 0:
        c = np.eye(n)
        for i in xrange(nRw):
            c = np.dot(c, p) + np.eye(n)
        return squareform(pdist(c,metric='cityblock'))
    else:
        pi = degree / degree.sum()
        return squareform(pdist(inv(np.eye(n) - p - pi.T),metric='cityblock'))

    
def compute_pinverse_diagonal(diag: ndarray)-> ndarray:
    """
    Compute inverse of a diagonal matrix.
    """
    m = diag.shape[0]
    i_diag = np.zeros((m,m))
    for i in range(m):
        di = diag[i, i]
        if di != 0.0:
            i_diag[i, i] = 1 / float(di)
    return i_diag
    
    
def compute_dsd_normalized(adj: ndarray, deg: ndarray, nrw: int= -1, lm: int= 1, is_normalized: bool=False)-> ndarray:
    """
    Function to compute DSD Matrix : choose `is_normalized` = False, and 
    `lm` = 1.
    """
    deg_i = compute_pinverse_diagonal(deg)
    P = np.matmul(deg_i, adj)
    Identity = np.identity(adj.shape[0])
    e = np.ones((adj.shape[0], 1))

    # Compute W
    scale = np.matmul(e.T, np.matmul(deg, e))[0, 0]
    W = np.multiply(1 / scale, np.matmul(e, np.matmul(e.T, deg)))

    up_P = np.multiply(lm, P - W)
    X_ = Identity - up_P
    X_i = np.linalg.pinv(X_)

    if nrw > 0:
        LP_t = Identity - np.linalg.matrix_power(up_P, nrw)
        X_i = np.matmul(X_i, LP_t)
    if is_normalized == False:
        return X_i
    
    # Normalize with steady state
    SS = np.sqrt(np.matmul(deg, e))
    SS = compute_pinverse_diagonal(np.diagflat(SS))
    return np.matmul(X_i, SS)


##### Thresholding and RBF kernel


def best_threshold(rbf: ndarray)-> ndarray:
    """
    Best threshold for RBF kernel
    """
    t = None
    for i in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        thresh = rbf.mean() - i*rbf.std()
        t_rbf = np.where((rbf < thresh) | (rbf == 1), 0, rbf)
        n, t = rbf.shape[0], t_rbf.copy()
        if n*1000 <= np.count_nonzero(t_rbf):
            return t_rbf
    return t


def compute_rbf(A: ndarray, gamma: float=None, t: float=None)-> ndarray:
    """
    Compute the RBF kernel
    """
    pd = pairwise_distance_matrix(A)
    rbf = laplacian_kernel(pd, gamma) if gamma else best_gamma_kernel(pd)
    t_rbf = np.where((rbf < t) | (rbf == 1), 0, rbf) if t else best_threshold(rbf)
    clean(t_rbf)
    return t_rbf
