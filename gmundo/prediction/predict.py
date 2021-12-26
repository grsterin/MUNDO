from typing import Tuple, Dict, List, Callable
import numpy as np

def get_neighbors_split(protein: str,
                        target_map: Dict[str, List[str]],
                        n_neighbors: int,
                        target_go_dict # changes made here
                       ) -> Tuple[int, int]:
    """
    This method would allow us to use a single parameter for a number of neighbors instead of 2 parameter, which would simplify grid search.
    It also allows to independently determine number of neighbors to use in knn for each node.
    Parameters:
        protein         - protein in target network for which we want to determine num of neighbors to use in knn voting
        target_map      - Dict[protein(target), [protein(target)]] which is sorted protein neighbors by their similarity for the target species (lower index in the list means the protein is more similar)
        n_neighbors     - No of neighbors for source plus target.
        target_go_f     - function that takes target protein and returns the GO labels associated with it
    Returns:
        Tuple[num_of_dsd_neighbors, num_of_munk_neighbors], where num_of_dsd_neighbors + num_of_munk_neighbors = n_neighbors
    """

    def get_n_closest_neighbors() -> List[str]:
        neighbors = target_map[protein] if protein in target_map else []
        return neighbors[:n_neighbors]

    def is_protein_annotated(prot: str) -> bool:
        go_annotations = target_go_dict[prot] if prot in target_go_dict else []
        return len(go_annotations) != 0

    target_closest_neighbors = get_n_closest_neighbors()
    num_closest_neighbors_annotated = len([neighbor for neighbor in target_closest_neighbors if is_protein_annotated(neighbor)])
    t = num_closest_neighbors_annotated / n_neighbors

    num_of_dsd_neighbors = int(t * n_neighbors)
    num_of_munk_neighbors = int((1 - t) * n_neighbors)

    return num_of_dsd_neighbors, num_of_munk_neighbors


def get_weight_coefficient(networks_hubalign_score: float,
                           networks_reference_score1: float,
                           networks_reference_score2: float,
                           alpha1: float,
                           alpha2: float):
    """
    Let's assume that there's a linear dependency between network similarity score and best weight factor alpha.
    In this case if we empirically determine good weight factors for some 2 network pairs, we'll be able to compute weight factor for any pair of networks knowing their similarity score.

    Parameters:
        networks_hubalign_score - mean HubAlign score of 300 highest scored node pairs of a pair of networks we want to find weights for
        networks_reference_score1 - mean HubAlign score of 300 highest scored node pairs of a pair of networks (e.g., highly similar Human and Mouse)
        networks_reference_score2 - mean HubAlign score of 300 highest scored node pairs of a different pair of networks (e.g., less similar Human and Baker's yeast)
        alpha1 - empirically determined best weight factor for the pair of networks used in networks_reference_score1
        alpha2 - empirically determined best weight factor for the pair of networks used in networks_reference_score2
    Returns:
        alpha: weight factor for the target network DSD votes for the case of interest.
    """

    # using notation y = kx + b
    k = (alpha2 - alpha1) / (networks_reference_score2 - networks_reference_score1)
    b = alpha1 - networks_reference_score1 * k	
    alpha = k * networks_hubalign_score + b
    
    return alpha


def MUNDO_predict(target_map: Dict[str, List[str]],
                  MUNK_map: Dict[str, List[str]],
                  n_neighbors: int,
                  target_go_dict, # change it here
                  source_go_dict,
                  MUNK_weight: float = 0.25) -> Dict[str, List[Tuple[str, float]]]:
    """
    Performs prediction on the target network,
    Parameters:
        target_map      - Dict[protein(target), [protein(target)]] which is sorted protein neighbors by their similarity for the target species(lower index in the list means the protein is more similar)
        MUNK_map        - Dict[protein(target), [protein(source)]] where the protein in this case is from the source species.
        n_neighbors     - integer values. No of neighbors for source plus target.
        target_go_f     - both functions that takes target protein and returns the GO labels associated with it.
        source_go_f     - both functions that takes source protein and returns the GO labels associated with it.
        MUNK_weight     - weight to use for MUNK votes.
    Returns:
        Dictionary containing mapping from target proteins to the sorted (in descending order) list of associated labels and their confidence values
    """

    def vote(target_voters: List[str], MUNK_voters: List[str]) -> List[Tuple[str, float]]:
        """
        Returns a list of (GO, confidence value) tuples
        """
        go_map = {}
        
        # Work on target 
        for v in target_voters:
            go_labels = target_go_dict[v] if v in target_go_dict else []
            for g in go_labels:
                go_map[g] = 1.0 if g not in go_map else go_map[g] + 1.0
        
        # Work on source
        for v in MUNK_voters:
            go_labels = source_go_dict[v] if v in source_go_dict else []
            for g in go_labels:
                go_map[g] = MUNK_weight if g not in go_map else go_map[g] + MUNK_weight

        return sorted(go_map.items(), reverse=True, key=lambda k: k[1])  # [(go_label, vote), ... ] format, vote is float
    
    # Get all the target proteins
    if type(target_map) is dict:
        proteins = target_map.keys()
    elif type(target_map) is np.ndarray:
        proteins = range(target_map.shape[0])
    protein_labels = {}
    
    for p in proteins:
        n_target_neighbors, n_MUNK_neighbors = get_neighbors_split(p, target_map, n_neighbors, target_go_dict)

        label = target_go_dict[p] if p in target_go_dict else None
        
        # Only happens if training proteins, skip them 
        if label != None:
            protein_labels[p] = label
            continue
        
        target_voters = target_map[p][:n_target_neighbors]
        MUNK_voters   = MUNK_map[p][:n_MUNK_neighbors]
        
        protein_labels[p] = vote(target_voters, MUNK_voters)

    return protein_labels
