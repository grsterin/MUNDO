import random
import numpy as np
from goatools.obo_parser import GODag
from goatools.associations import read_gaf
from goatools.semantic import TermCounts, get_info_content
from goatools.semantic import resnik_sim
import pkg_resources



"""
Four metrics are used for FUNCTION prediction:
ACCURACY
F1
RESNIK metric  - Described in our new GLIDER paper
RESNIK average - Described in  Pandey et. al.
        https://academic.oup.com/bioinformatics/article/24/16/i28/201569
"""

def score_cv(test_nodes, test_labelling, real_labelling):
    """
    Scores cross validation by counting the number of test nodes that
    were accurately labeled after their removal from the true
    labelling.
    """
    correct = 0
    total   = 0
    
    for node in test_nodes:
        """
        Only check for nodes in the test list
        """
        
        if node not in test_labelling:
            continue
    
        test_label = test_labelling[node]
        
        if type(test_label) is list:
            """
            Highest confidence label
            """
            try:
                test_label = test_label[0][0]
            except:
                test_label = None
        if test_label in real_labelling[node]:
            correct += 1
        total += 1
    return float(correct) / float(total)

def kfoldcv(k, 
            labels, 
            prediction_algorithm, 
            randomized=True, 
            degree_vec = None, 
            reverse = False):
    """Performs k-fold cross validation (for metric ACCURACY).

    Args:
      - A number of folds k
      - A labeling for the nodes. (protein labels)
      - An algorithm that takes the training labels
      and outputs a predicted labelling.

    The prediction algorithm should take a map: training protein -> training labels
    and return a map: protein -> labels, where the map adds both test and train proteins
    
    Output: 
      - A list where each element is the accuracy of the
      learning algorithm holding out one fold of the data.
    """
    
    nodes = list(labels.keys())
    if randomized:
        random.shuffle(nodes)
    accuracies = []
    for i in range(0, k):
        inc = int(len(nodes) / k)
        
        x = inc * i
        y = inc * (i + 1)
        
        if i + 1 == k:
            y = len(nodes)
        
        """
        In k-folds CV if the `reverse` option is set to true, then the 
        training and testing nodes is flipped. That means, the k-1 blocks is
        now the test block and the remaining 1 block is the train block.
        """
        if not reverse:
            training_nodes = nodes[:x] + nodes[y:]
            test_nodes = nodes[x:y]
        else:
            training_nodes  = nodes[x:y]
            test_nodes      = nodes[:x] + nodes[y:]
        
        training_labels = {n: labels[n] for n in training_nodes}
        test_labelling  = prediction_algorithm(training_labels)
        
        accuracy = score_cv(test_nodes, 
                            test_labelling, 
                            labels)
        
        accuracies.append(accuracy)
    return accuracies

def kfoldcv_with_pr(k, 
                    labels, 
                    prediction_algorithm, 
                    randomized=True, 
                    reverse = False):
    """Performs k-fold cross validation.

    Args:
      - A number of folds k
      - A labeling for the nodes.
      - An algorithm that takes the training labels
      and outputs a predicted labelling.
      
      The prediction should take a map: prot_train -> [label] where `prot_train` represents proteins in training set and return,
      prot_test -> [(label, conf)], where `prot_test` are proteins for testing, and conf is the confidence ratio.

    Output: 
      - A list where each element is the accuracy of the
      learning algorithm holding out one fold of the data.
    """
    nodes = list(labels.keys())
    if randomized:
        random.shuffle(nodes)
    fscores = []
    for i in range(0, k):
        inc = int(len(nodes) / k)
        x = inc * i
        y = inc * (i + 1)
        if i + 1 == k:
            y = len(nodes)
            
        """
        The meaning of `reverse` is the same here.
        """
        
        if not reverse:
            training_nodes = nodes[:x] + nodes[y:]
            test_nodes = nodes[x:y]
        else:
            training_nodes  = nodes[x:y]
            test_nodes      = nodes[:x] + nodes[y:]
            
        training_labels = {n: labels[n] for n in training_nodes}
        test_labelling  = prediction_algorithm(training_labels)
        
        fmax = score_cv_pr(test_nodes, 
                           test_labelling, 
                           labels)
        fscores.append(fmax)
    return fscores


def score_cv_pr(test_nodes, 
                test_labelling, 
                real_labelling, 
                ci = 1000):
    """Scores cross validation by counting the number of test nodes that
    were accurately labeled after their removal from the true
    labelling.
    """
    
    def compute_fmax(precs, recalls):
        f1 = [2 * (p * r) / (p+r) if p+r != 0 else 0 for (p, r) in zip(precs, recalls)]
        return np.max(f1)

    """
    Computation of precision and recall is done here.
    """
    cis = [i / ci for i in range(ci)]
    precision = []
    recall    = []
    for c in cis:
        prec_counter = 0
        rec_counter  = 0
        prs          = 0
        rcs          = 0
        for node in test_nodes:
            if node not in test_labelling:
                continue
            
            pred_labels = set([t for (t,c1) in test_labelling[node] if c1 >= c])
            if len(pred_labels) != 0:
                prec_counter += 1
            true_labels = set(real_labelling[node])
            if len(true_labels) != 0:
                rec_counter += 1
            prs += len(pred_labels.intersection(true_labels)) / float(len(pred_labels)) if len(pred_labels) != 0 else 0 
            rcs += len(pred_labels.intersection(true_labels)) / float(len(true_labels)) if len(true_labels) != 0 else 0
            
        prs     = prs / prec_counter if prec_counter != 0 else 0
        rcs     = rcs / rec_counter  if rec_counter  != 0 else 0
        precision.append(prs)
        recall.append(rcs)
    fmax  = compute_fmax(precision, recall)
    return fmax 


def kfoldcv_sim(k, 
                labels, 
                prediction_algorithm, 
                randomized=True, 
                reverse = False,
                namespace = "MF",
                ci = 20):
    """Performs k-fold cross validation.

    Args:
      - A number of folds k
      - A labeling for the nodes.
      - An algorithm that takes the training labels
      and outputs a predicted labelling.

    Output: 
      - A list where each element is the accuracy of the
      learning algorithm holding out one fold of the data.
    """
    gdagfile = pkg_resources.resource_filename('glide', 'data/go-basic.obo.dat')
    assoc_f  = pkg_resources.resource_filename('glide', 'data/go-human.gaf.dat')
    godag    = GODag(gdagfile)
    assoc    = read_gaf(assoc_f, namespace = namespace)
    t_counts = TermCounts(godag, assoc)
    nodes = list(labels.keys())
    if randomized:
        random.shuffle(nodes)
    fscores = []
    for i in range(0, k):
        inc = int(len(nodes) / k)
        x = inc * i
        y = inc * (i + 1)
        if i + 1 == k:
            y = len(nodes)
        if not reverse:
            training_nodes = nodes[:x] + nodes[y:]
            test_nodes = nodes[x:y]
        else:
            training_nodes  = nodes[x:y]
            test_nodes      = nodes[:x] + nodes[y:]
            
        training_labels = {n: labels[n] for n in training_nodes}
        test_labelling  = prediction_algorithm(training_labels)
        
        fmax = score_cv_sim(test_nodes, 
                           test_labelling, 
                            labels, 
                            godag, 
                            t_counts,
                            ci = ci)
        fscores.append(fmax)
    return fscores



def score_cv_sim(test_nodes, 
                 test_labelling, 
                 real_labelling,
                 go_dag,
                 term_counts,
                 ci = 1000):
    """
    Scores cross validation by counting the number of test nodes that
    were accurately labeled after their removal from the true
    labelling.
    """
    def sem_similarity_(go_id, go_ids, avg = False):
        """
        If avg == True, compute the average Resnik Similarity Instead.
        """
        sims = [resnik_sim(go_id, go_i, go_dag, term_counts) for go_i in go_ids]
        if avg:
            return np.average(sims)
        return np.max(sims)
    def sem_similarity(gois_1, gois_2, avg = False):
        """
        If avg == True, use the average Resnik Similarity, provided in Pandey et. al.
        https://academic.oup.com/bioinformatics/article/24/16/i28/201569
        """
        if avg:
            sims = [sem_similarity_(g1, gois_2) for g1 in gois_1]
            return np.average(sims)
        
        sims1 = [sem_similarity_(g1, gois_2) for g1 in gois_1]
        sims2 = [sem_similarity_(g2, gois_1) for g2 in gois_2]
        n_1   = len(sims1)
        n_2   = len(sims2)
        return (np.sum(sims1) + np.sum(sims2)) / float(n_1 + n_2)

    cis  = [i / ci for i in range(ci)]
    sims = []
    for c in cis:
        sim_counter  = 0
        sim          = 0
        for node in test_nodes:
            if node not in test_labelling:
                continue
            pred_labels = set([t for (t,c1) in test_labelling[node] if c1 >= c])
            true_labels = set(real_labelling[node])
            if len(true_labels) != 0:
                sim_counter += 1
            else:
                continue
            if len(pred_labels) != 0:
                sim    += sem_similarity(pred_labels, true_labels)
        sims.append(sim / float(sim_counter))
    return np.max(sims) 

