import numpy as np
from .utils import *
from .linalg import rkhs
import networkx as nx
from typing import Any, Dict, List, NewType, Tuple, Set
import sys
import numpy
from numpy.linalg import pinv

## Define New Types
ndarray = NewType("numpy ndarray", np.ndarray)
Graph   = NewType("networkx Graph Object", nx.Graph)


def embed_matrices(source_rkhs: ndarray,
                   target_diff: ndarray,
                   landmark_id: List[Tuple[int, int]])-> ndarray:
    """
    This function performs MUNK embedding. Let the `source_rkhs` be represented as
    $C_1: m x m$, `target_diff` as $D_2: n x n$. The `landmark_id` represents the 
    mapping between the orthologs of source and target PPI. Let $C_{1L}$ be the rows 
    of $C_1$ corresponding to the landmark ids (same as $D_{2L}). Then, the output 
    embedding is going to be:
    
    $$
    D_2 = C_{1L}^{\dagger} D_{2L}^T : m x n  
    $$
    """
    s_landmark_id, t_landmark_id = zip(*landmark_id)
    return pinv(source_rkhs[s_landmark_id,:]) @ (target_diff[t_landmark_id,:])

def coembed_networks(source_dsd: ndarray,
                     target_dsd: ndarray,
                     landmark_indices: List[Tuple[int, int]], verbose: bool = True)-> ndarray:
    def log(strng):
        if verbose:
            print(strng)
            
    log('\tComputing RKHS for source network... ')
    source_rkhs = rkhs(source_dsd) # m x m 
    log('\tEmbedding matrices... ')
    target_rkhs_hat = embed_matrices(source_rkhs, target_dsd, landmark_indices) # m x n
    log('\tCreating final munk matrix... ')
    munk_matrix = np.dot(source_rkhs, target_rkhs_hat) # m x [m m] x n => m x n 
    
    return munk_matrix.T # n x m ; where n represents the target node-count.

