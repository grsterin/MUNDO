from typing import Tuple, Dict, List
import numpy as np
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def get_neighbors_split(protein: int,
                        target_map: Dict[int, List[int]],
                        n_neighbors: int,
                        target_go_dict) -> Tuple[int, int]:
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

    def get_n_closest_neighbors() -> List[int]:
        neighbors = target_map[protein] if protein in target_map else []
        return neighbors[:n_neighbors]

    def is_protein_annotated(prot: int) -> bool:
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


def mundo_predict(target_map: Dict[int, List[int]],
                  munk_map: Dict[int, List[int]],
                  n_neighbors: int,
                  target_go_dict: Dict[int, str],
                  source_go_dict: Dict[int, str],
                  munk_weight: float = 0.25) -> Dict[str, List[Tuple[str, float]]]:
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

    def vote(target_voters: List[int], munk_voters: List[int]) -> List[Tuple[str, float]]:
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
        for v in munk_voters:
            go_labels = source_go_dict[v] if v in source_go_dict else []
            for g in go_labels:
                go_map[g] = munk_weight if g not in go_map else go_map[g] + munk_weight

        return sorted(go_map.items(), reverse=True, key=lambda k: k[1])  # [(go_label, vote), ... ] format, vote is float

    proteins = {}
    if type(target_map) is dict:
        proteins = target_map.keys()
    elif type(target_map) is np.ndarray:
        proteins = range(target_map.shape[0])

    protein_labels = {}
    for p in proteins:
        n_target_neighbors, n_munk_neighbors = get_neighbors_split(p, target_map, n_neighbors, target_go_dict)

        target_voters = target_map[p][:n_target_neighbors]
        munk_voters = munk_map[p][:n_munk_neighbors] if munk_weight != 0.0 else []
        
        protein_labels[p] = vote(target_voters, munk_voters)

    return protein_labels



def perform_binary_OVA(E, labels, params = {}, clf_type="LR", confidence = True):
    """
    Perform binary svc on embedding and return the new labels
    @param E: Embedding of size n x k
    @param labels: A dictionary that maps the index in the row of embedding to labels. An index can have many labels
    @param params:
    @return labels: Since the dictionary labels is incomplete (some of the indices donot have any labels associated with it), this function performs SVC for each labels and completels the labels dictionary, and returns it.
    """
    def convert_labels_to_dict(lls):
        """
        This function takes in a list of labels associated with a protein embedding, and returns the dictionary that is keyed .
        by the index of protein embedding with the value 
        """
        l_dct = {}
        for k in lls:
            ll       = lls[k]
            l_dct[k] = {i: True for i in ll}
        return l_dct

    labels_dct = convert_labels_to_dict(labels)
    
    def transpose_labels(labels):
        """
        Returns a dict with go labels as keys with values being a list of 
        proteins with that label
        """
        transpose = {}
        for protein in labels:
            lls = labels[protein]
            for ll in lls:
                if not ll in transpose:
                    transpose[ll] = []
                transpose[ll].append(protein)
        return transpose

    preprocess = False
    if "preprocess" in params:
        from sklearn.pipeline import make_pipeline
        preprocess = params["preprocess"]
        print(f"Preprocess set to {preprocess}")
    # perform a filter on the label classes we are going to use
    t_labels                     = transpose_labels(labels)
    print(f"The number of All the Labels {len(t_labels)}")

    """
    #[used_labels, unused_labels] = jaccard_filter(t_labels, 0.1)
    [used_labels, assoc_dict]   = jaccard_filter_added_unused(t_labels, threshold=thres)
    print(f"The number of Used Labels {len(used_labels)}")
    """

    samples    = {}
    n          = E.shape[0]

    # Adding Positive samples
    for i in labels:
        lls = labels[i]
        for ll in lls:
            """
            # ignore the labels that we aren't considering
            if ll not in used_labels:
                continue
            """
            if ll not in samples:
                samples[ll] = {"positive" : [], "negative" : [], "null" : [], "clf": None}
                if clf_type != "LR":
                    samples[ll]["clf"] = (SVC(gamma = "auto", probability=True, max_iter = 10000) 
                                          if not preprocess else
                                          make_pipeline(StandardScaler(), SVC(probability = True)))
                else:
                    samples[ll]["clf"] = LogisticRegression(random_state = 0)
            samples[ll]["positive"].append(i)

    # Adding Negative samples and creating null set (unlabeled data)
    null_set = []
    for i in range(n):
        if i not in labels:
            null_set.append(i)
        else:
            for j in samples:
                if j not in labels_dct[i]:
                    samples[j]["negative"].append(i)
    null_set = np.array(null_set)

    # Balance negative and positive samples
    # and train the probabilistic SVMs
    for s in samples:
        n_pos = len(samples[s]["positive"])
        n_neg = len(samples[s]["negative"])
        n_val = n_pos if n_pos < n_neg else n_neg
        samples[s]["positive"] = np.array(samples[s]["positive"][:n_val])
        samples[s]["negative"] = np.array(samples[s]["negative"][:n_val])
        lbls                   = np.zeros((2 * n_val, ))
        lbls[:n_pos]           = 1
        inputs                 = np.concatenate([samples[s]["positive"],
                                                 samples[s]["negative"]])
        samples[s]["clf"].fit(E[inputs], lbls)
    
    # iterate over the classes computing the probability for each point in
    # the null set for each class
    probabilities = np.zeros(( len(null_set), len(samples) ))
    sample_keys = list(samples.keys())
    for i, s in enumerate(sample_keys):
        # predict_proba returns a matrix, each row is the prediction for a datapoint
        # and each column is for a different class. col 0 is negative, col 1 is positive
        probabilities[:,i] = samples[s]["clf"].predict_proba(E[null_set])[:,1]


    for i in range(len(null_set)):
        prbs = probabilities[i]
        e_id = null_set[i]
        if confidence:
            labels[e_id] = [(s, p) for s, p in zip(sample_keys, prbs)]
        else:
            prb_id = np.argmax(prbs)
            labels[e_id] = sample_keys[prb_id]
    return labels

