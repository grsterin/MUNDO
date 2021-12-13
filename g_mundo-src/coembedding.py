import json
import nump as np
from glob import glob
import networkx as nx
from utils import compute_dsd
import argparse
import pandas as pd
from g_mundo.linalg import compute_rbf
from g_mundo.coembed import embed_matrices
from utils import compute_dsd, get_unpickled_networks_url, get_unpickled_network, get_dsd_url, get_dsd_matrices

def get_args():
    """
    Get the command line parameters.
    """
    parser = argparse.ArgumentParser()
    
    """
    Input and output urls in the command line
    """
    parser.add_argument( "--input_folder", 
                        help="The folder containing input files", 
                        type=str, 
                        nargs='?', 
                        default = None)
    parser.add_argument("--inprefix",
                        help ="The prefix of the input file contents",
                        type = "str"
                        default="")
    parser.add_argument("--precomputedprefix", 
                       help = "If dsd files are already precomputed, this is the identifier",
                        default = None)
    parser.add_argument("-o", "--output_folder", 
                        help="name of directory where network embeddings should be saved.", 
                        type=str, 
                        required=True)
    parser.add_argument("--outprefix", help="name of current run - appended to all outputted matrix files", 
                        type=str, 
                        nargs= '?', 
                        default= '')
    
    """
    The output files the program is going to output
    """
    parser.add_argument("--compute_munk", 
                        action = "store_true", 
                        default = False, 
                        help = "Compute the MUNK embedding in the curr folder")
    parser.add_argument("--compute_target_dsd", 
                        action = "store_true", 
                        default = False, 
                        help = "Compute the Target DSD embedding in the curr folder")
    parser.add_argument("--compute_source_dsd", 
                        action = "store_true", 
                        default = False, 
                        help = "Compute the Source DSD embedding in the curr folder")
    
    """
    Hyperparameters
    """
    
    parser.add_argument('-n', 
                        "--normalized", 
                        help="Use normalized DSD - default is not normalized (y/n). ", 
                        action = "store_true",
                        default = False)
    parser.add_argument('-g', 
                        "--gamma", 
                        help="value of gamma to use for RBF Kernel. If compute = \'d\', gamma is ignored", 
                        type= float, 
                        nargs= '?', 
                        default= None)
    parser.add_argument('-t', 
                        "--thres", 
                        help="value of threshold to use for RBF Kernel. If compute = \'d\', threshold is ignored", 
                        type= float, 
                        nargs= '?', 
                        default= None)
    """
    Util command. To print out the intermediate and debugging messages
    """
    parser.add_argument("-v", 
                        "--verbose", help="print status updates (y/n)", 
                        type=str, 
                        nargs='?', 
                        default = 'y', 
                        choices=['y','n'])
    return parser.parse_args()


def get_network_nodes(graph):
    """
    Takes the netwokx graph annotated by protein symbols and returns all the present
    nodes.
    Sorts the nodes by degree and removes the version number in the protein names (reomve the .* part)
    """
    deg_sorted_nodes = [node for node, d in sorted(graph.degree, key=lambda x: x[1], reverse=True)]
    indexed_nodes = {node.split('.')[0]:i for i, node in enumerate(sorted_nodelist)} 
    return deg_sorted_nodes, indexed_nodes

def save_source_target_maps(folder, prefix, source_lst, target_lst):
    """
    Save the source and target lst in the given folder with prefix applied
    """
    source_file = f"{folder}/source_labels{prefix}.json"
    target_file = f"{folder}/target_labels{prefix}.json"
    with open(source_file, "w") as sp:
        json.dump({i: node for i, node in enumerate(source_lst)}, sp)
    with open(target_file, "w") as tp:
        json.dump({i: node for i, node in enumerate(target_lst)}, tp)
    
    
def main(args):
    verbose = args.verbose
    
    def log(strng):
        if verbose:
            print(strng)
            
    log("Retrieving pickled networks...")
    source, target         = get_unpickled_networks_url(args.input_folder, 
                                                        args.inprefix)
    source_net, target_net = get_unpickled_networks(source, target)
    
    log("Retrieving reciprocal best hits...")
    rbh_df = set(pd.read_csv(f"{args.input_folder}/reciprocal_best_hits.txt", 
                             sep = "\t", 
                             header = None).to_records(index = False))
    
    sort_s_nodes, index_s_nodes = get_network_nodes(source_net)
    sort_t_nodes, index_t_nodes = get_network_nodes(target_net)
    
    ### Index landmarks
    indexed_landmarks = [(index_s_nodes[p], index_t_nodes[q]) for 
                         p, q in rbh_df]
    
    if args.precomputedprefix:
        source, target = get_dsd_url(args.input_folder,
                                     args.precomputedprefix)
        source_dsd, target_dsd      = get_dsd_matrices(source, 
                                                       target)
    else:
        log("Computing DSD for source network...")
        source_dsd                  = compute_dsd(source_net, sort_s_nodes)
        log("Computing DSD for target network...")
        target_dsd                  = compute_dsd(target_net, sort_t_nodes)
    
    log("Generating the RBF kernel...")
    s_rbf = compute_rbf(source_dsd, args.gamma, args.thres)
    t_rbf = compute_rbf(target_dsd, args.gamma, args.thres)
    
    log("Saving row and column labels...")
    save_source_target_maps(args.input_folder, 
                            args.input_prefix, 
                            sort_s_nodes,
                            sort_t_nodes)
    
    if args.compute_munk:
        log("Computing MUNK embedding and saving it...")
        munk_mat = embed_matrices(s_rbf,
                                  t_rbf,
                                  indexed_landmarks)
        np.save(f"{args.output_folder}/munk_matrix_{args.outprefix}.npy", 
                munk_mat)
    
    if args.compute_source_dsd:
        log("Saving source dsd matrix...")
        np.save(f"{args.output_folder}/source_dsd_matrix_{args.outprefix}.npy",
                source_dsd)
        
    if args.compute_target_dsd:
        log("Saving target dsd matrix...")
        np.save(f"{args.output_folder}/target_dsd_matrix_{args.outprefix}.npy",
                target_dsd)
    