import numpy as np
from numpy.linalg import pinv, norm
import subprocess
import pathlib
import os

################### ISORANK CODE #############################
def isorank(G1, G2, row_map, col_map, alpha, matches, E = None, iterations = 5):
    """
    Compute the ISORANK matches from G1 and G2, two networkx graphs.
    E is the sequence based similarity score.
    
    row_map : dict{G1_nodes -> G1_ID}, size = m
    col_map : dict{G2_nodes -> G2_ID}, size = n
    
    E : numpy matrix {m x n}, sequence similarity score
    """
    def _isorank_compute_Aij(i, j, m, n):
        A = np.zeros((m, n))             
        for u, u_id in row_map.items():
            for v, v_id in col_map.items():
                A[u_id, v_id] = (1. / (len(G1[u]) * len(G2[v])) if 
                          (G1.has_edge(i, u) and G2.has_edge(j, v)) 
                          else 0)
        return A
    
    def _isorank_compute_next_r(R, E = None, alpha = 1):
        """
        Performs the R = AR operation, where R is a matrix and A is a 4 dimensional tensor.
        """
        m, n   = R.shape
        R_next = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                R_next[i, j] = alpha * np.sum(_isorank_compute_Aij(i, j, m, n) * R)
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
    for i in range(iterations):
        R_next = _isorank_compute_next_r(R, E, alpha)
        errors.append(np.linalg.norm(R - R_next, ord = "fro"))
        R      = R_next
    
    # Find the best pairs
    best_pairs = _isorank_one_to_one(R)
    
    i_row_map  = {value: key for key, value in row_map.items()}
    i_col_map  = {value: key for key, value in col_map.items()}
    
    best_pairs = [(i_row_map[p], i_col_map[q]) for p, q in best_pairs]
    
    return best_pairs, R, errors


def hubalign(smaller_network_file_name: str,
             bigger_network_file_name: str,
             input_folder: str,
             output_folder: str,
             lmbda: float = 0.1,
             alpha: float = 1,
             blast_file: str = None) -> str:
    """
    Parameters:
        smaller_network_name - name of the network file with a smaller number of nodes
        bigger_network_name - name of the network file with a bigger number of nodes
        input_folder - location of the input network files
        output_folder - location for the output alignment file
        lmbda - parameter which controls importance of the edge weight compared to node weight
        alpha - parameter which controls importance of sequence similarity compared to topological similarity
        blast_file - path to a file containing space-separated node pairs and their BLAST similarity scores
    Returns:
        path to the alignment file
    """
    hubalign_binary_path = f"{pathlib.Path(__file__).parent.absolute()}/bin/hubalign-with-scores"
    hubalign_call_array = [hubalign_binary_path,
                           smaller_network_file_name,
                           bigger_network_file_name,
                           "-i", input_folder,
                           "-o", output_folder,
                           "-l", str(lmbda),
                           "-a", str(alpha)]

    if alpha != 1 and blast_file is not None:
        hubalign_call_array.extend(["-b", blast_file])
    elif alpha == 1 and blast_file is not None:
        raise Exception("HubAlign error: if alpha is equal to 1, blast file should not be passed"
                        "into function")
    elif alpha != 1 and blast_file is None:
        raise Exception("HubAlign error: if blast file is passed into function, alpha should not be 1")

    process = subprocess.Popen(hubalign_call_array,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode != 0:
        raise Exception(f"HubAlign error: process executed with errors. Return code is {process.returncode}. "
                        f"Error: {err.decode('UTF-8')}")

    alignment_file_path = f"{output_folder}/{smaller_network_file_name}-{bigger_network_file_name}.alignment"
    if not os.path.exists(alignment_file_path):
        raise Exception(f"HubAlign error: process executed with errors: alignment file doesn't "
                        f"exist in {output_folder}")
    return alignment_file_path
