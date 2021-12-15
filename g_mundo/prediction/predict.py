def get_num_of_voting_neighbors(node, dsd_matrix, z: int) -> Tuple[int, int]:
"""
This method would allow us to use a single parameter for a number of neighbors instead of 2 parameter, which would simplify grid search.
It also allows to independently determine number of neighbors to use in knn for each node.  

Parameters:
    node - node in target network for which we want to determine num of neighbors to use in knn voting
dsd_matrix - dsd matrix of target network
    z - cumulative number of DSD and MUNK neighbors to use in knn

Returns:
    Tuple[num_of_dsd_neighbors, num_of_munk_neighbors], where num_of_dsd_neighbors + num_of_munk_neighbors = z
"""

    z_dsd_neighbors = get_z_closest_neighbors(node, dsd_matrix)
    num_annotated_in_z_neighborhood = len([neighbor for neighbor in z_dsd_neighbors if is_node_annotated(neighbor)])
    t = num_annotated_in_z_neighborhood / z

    num_of_dsd_neighbors = int(t * z)
    num_of_munk_neighbors = int((1 - t) * z)
    return num_of_dsd_neighbors,  num_of_munk_neighbors

def get_weight_coefficient(networks_hubalign_score: float, networks_reference_score1: float, networks_reference_score2: float, alpha1: float, alpha2: float):
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

def MUNDO_predict(target_map, 
                  MUNK_map, 
                  n_target_neighbors, 
                  n_MUNK_neighbors,
                  target_go_f,
                  MUNK_go_f,
                  MUNK_weight = 0.25):
    """
    Performs prediction on the target network, 
    target_map: map protein(target) -> [protein (source)] which is sorted protein neighbors by their similarity for the target species(lower index in the list means the protein is more similar)
    
    MUNK map: map protein(target) -> [protein (source)] where the protein in this case is from the source species.
    
    n_target_neighbors, n_MUNK_neighbors -> integer values. No of neighbors for both
    source and target.
    
    target_go_f, MUNK_go_f : both functions that takes target (or source if the function is MUNK_go_f) and returns the GO labels associated with them.
    """
    def vote(target_voters, MUNK_voters):
        """
        Returns a map GO -> confidence value.
        """
        go_map = {}
        
        # Work on target 
        for v in target_voters:
            go_labels = target_go_f[v]
            for g in go_labels:
                go_map[g] = 1 if g not in go_map else go_map[g] + 1
        
        # Work on source
        for v in MUNK_voters:
            go_labels = MUNK_go_f[v]
            for g in go_labels:
                go_map[g] = MUNK_weight if g not in go_map else go_map[g] + 1
        
        return sorted(go_map.items(), reverse = True, lambda k: k[1])
    
    # Get all the target proteins
    proteins       = target_map.keys()
    protein_labels = {}
    
    for p in proteins:
        label = target_go_f(p)
        
        # Only happens if training proteins, skip them 
        if label != None:
            protein_labels[p] = label
            continue
        
        target_voters = target_map[p][:n_target_neighbors]
        MUNK_voters   = MUNK_map[p][:n_target_neighbors]
        
        protein_labels[p] = vote(target_voters, MUNK_voters)
    return protein_labels[p]