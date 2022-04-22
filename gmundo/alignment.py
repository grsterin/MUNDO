import numpy as np
import pandas as pd
from numpy.linalg import pinv, norm
import subprocess
from tqdm import tqdm
import pathlib
import os

################### ISORANK CODE #############################
def isorank(G1, G2, row_map, col_map, alpha, matches = 100, E = None, iterations = 5, saveto = None, rowname = "source", colname = "target"):
    """
    Compute the ISORANK matches from G1 and G2, two networkx graphs. These graphs are labeled
    using the actual name of the nodes

    E is the sequence based similarity score.
    
    row_map : dict{G1_nodes -> G1_ID}, size = m
    col_map : dict{G2_nodes -> G2_ID}, size = n
    
    E : numpy matrix {m x n}, sequence similarity score
    """

    i_row_map  = {value: key for key, value in row_map.items()}
    i_col_map  = {value: key for key, value in col_map.items()}
    

    def _isorank_mult_R_A_ij(i, j, R):
        """
        Suffix ending with _ imply that the variable is made up of true label name.
        The indices i, j are the index labels, not true labels.
        """
        A  = np.zeros((m, n))
        i_ = i_row_map[i]
        j_ = i_col_map[j]
        i_neighbors_ = G1.neighbors(i_)
        j_neighbors_ = G2.neighbors(j_)
        
        i_neighbors  = [row_map[i_n_] for i_n_ in i_neighbors_]
        j_neighbors  = [col_map[j_n_] for j_n_ in j_neighbors_]

        A_ij_local   = np.zeros((len(i_neighbors), len(j_neighbors)))
        for id_i, ni in enumerate(i_neighbors):
            for id_j, nj in enumerate(j_neighbors):
                A_ij_local[id_i, id_j] = 1. / (len(G1[i_row_map[ni]]) * len(G2[i_col_map[nj]]))
        R_local      = R[np.ix_(i_neighbors, j_neighbors)]
        return np.sum(R_local * A_ij_local)
    
    def _isorank_compute_next_r(R, E = None, alpha = 1):
        """
        Performs the R = AR operation, where R is a matrix and A is a 4 dimensional tensor.
        """
        m, n   = R.shape
        R_next = np.zeros((m, n))

        for id_ in tqdm(range(m * n)):
            i, j = [int(id_ / n), id_ % n]
            R_next[i, j] = alpha * _isorank_mult_R_A_ij(i, j, R)
        if E is not None:
            R_next += (1-alpha) * E
        return R_next / norm(R_next)
    
    def _isorank_one_to_one(R_final):
        """
        Find the best pairings from the obtained R_final, using GREEDY method
        """
        R_temp        = np.copy(R_final)
        best_pairings = []
        ps, qs        = np.unravel_index(np.argsort(-R_temp.flatten()), R_final.shape)

        row_used      = set()
        col_used      = set()

        pairings = min(*R.shape) if matches is None else min(matches, min(*R.shape))
        print(f"Number of pairings... {pairings}")
        print(f"Len(ps) = {len(ps)} Len(qs) = {len(qs)}")
            
        for p, q in zip(ps, qs):
            if p in row_used or q in col_used:
                continue
            row_used.add(p)
            col_used.add(q)

            best_pairings.append((p, q, R_final[p, q]))
            if len(best_pairings) >= pairings:
                print("Break complete")
                break
                
        return best_pairings
    
    m      = len(row_map)
    n      = len(col_map)
    # Random Initialization
    R      = np.random.rand(m, n)
    # Normalize
    R      = R / norm(R)
    
    errors = []
    
    # Compute Isorank matrix
    R_next     = None
    for i in range(iterations):
        print(f"Running iterations {i}...")
        R_next = _isorank_compute_next_r(R, E, alpha)
        errors.append(np.linalg.norm(R - R_next, ord = "fro"))
        R      = R_next
    
    # Find the best pairs
    best_pairs = _isorank_one_to_one(R)
    print(len(best_pairs))
    if saveto:
        # convert back to true label
        best_pairs = [(i_row_map[p], i_col_map[q], w) for p, q, w in best_pairs]
        mappings   = pd.DataFrame(best_pairs, columns = [rowname, colname, "weight"])
        mappings.to_csv(saveto, index = None, sep = "\t")
        
    # The output is going to be the best pairings from 
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
