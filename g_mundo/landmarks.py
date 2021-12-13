import numpy as np
import networkx as nx
from typing import Any, Dict, List, NewType, Tuple, Set
from blast_tools import BlastParser

## Define New Types
ndarray = NewType("numpy ndarray", np.ndarray)
Graph   = NewType("networkx Graph Object", nx.Graph)
Hit     = NewType('Hit object defined in blast_tools.py', Generic)

############################# CODE HERE #################################
def index_landmarks(source_indexed_nodes: Dict[str, int],
                    target_indexed_nodes: Dict[str, int],
                    reciprocal_best_hits: Set[Tuple[str, str]])-> List[Tuple[str, str]]:
    """
    Function that takes in a name->id source and target index mapping, and a 
    list of (name, name) best hits, and returns (id, id) best hits
    """
    return [(source_indexed_nodes[src_query], target_indexed_nodes[tgt_match])
            for src_query, tgt_match in reciprocal_best_hits]


## TODO: Add ISORANK


## TODO: Add HUBALIGN
