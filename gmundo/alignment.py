import numpy as np
from numpy.linalg import pinv, norm

################### ISORANK CODE #############################
def isorank(G1, G2, row_map, col_map, alpha, matches, E = None, iterations = 5):
    """
    Compute the ISORANK matches from G1 and G2, two networkx graphs.
    E is the sequence based similarity score.
    
    row_map : dict{G1_nodes -> G1_ID}, size = m
    col_map : dict{G2_nodes -> G2_ID}, size = n
    
    E : numpy matrix {m x n}, sequence similarity score
    """
    def _isorank_compute_A():
        """
        Function to compute the isorank score for the networks G1 and G2.
        G1: networkx graph
        G2: networkx graph
        A : A[i, j][u, v]
        """
        A    = np.zeros((len(row_map), len(col_map), len(row_map), len(col_map)))
        for i, i_id in row_map.items():
            for j, j_id in col_map.items():
                for u, u_id in row_map.items():
                    for v, v_id in col_map.items():
                        A[i, j, u, v] = (1. / (len(G1[r]) * len(G2[c])) if 
                                        (G1.has_edge(i, u) and G2.has_edge(j, v)) 
                                         else 0)
        return A
    
    def _isorank_compute_next_r(A, R, E = None, alpha = 1):
        """
        Performs the R = AR operation, where R is a matrix and A is a 4 dimensional tensor.
        """
        m, n   = R.shape
        R_next = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                R_next[i, j] = alpha * np.sum(A[i, j] * R)
        if E is not None:
            R_next += (1-alpha) * E
        return R_next / norm(R_next)
    
    def _isorank_one_to_one(R_final):
        """
        Find the best pairings from the obtained R_final
        """
        R_temp        = np.copy(R_final)
        best_pairings = []
        for i in range(matches):
            p, q = np.unravel_index(np.argmax(R_temp, axis = None), R_final.shape)
            R_temp[p, :] = -100
            R_temp[:, q] = -100
            best_pairings.append((p, q))
        return best_pairings
    
    m      = len(row_map)
    n      = len(col_map)
    R      = np.eye(m, n)
    errors = []
    
    # Compute Isorank matrix
    A      = _isorank_compute_A()
    for i in range(iterations):
        R_next = _isorank_compute_next_r(A, R, E, alpha)
        errors.append(np.linalg.norm(R - R_next, ord = "fro"))
        R      = R_next
    
    # Find the best pairs
    best_pairs = _isorank_one_to_one(R)
    
    i_row_map  = {value: key for key, value in row_map.items()}
    i_col_map  = {value: key for key, value in col_map.items()}
    
    best_pairs = [(i_row_map[p], i_col_map[q]) for p, q in best_pairs]
    
    return best_pairs, R, errors
                      