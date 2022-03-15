#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import argparse
import sys
import os
sys.path.append(os.getcwd()) 
sys.path.append(f"{os.getcwd()}/src")
from unimundo_utils import read_network_file, get_go_lab, get_prot_go_dict, get_go_lab_src
from gmundo.prediction.scoring import kfoldcv, kfoldcv_with_pr
from gmundo.prediction.predict import mundo_predict
import pandas as pd
import numpy as np
import json


def construct_predictor_mundo(target_neighbors, munk_neighbors, source_prot_go, n_neighbors=20, n_neighbors_munk = 10, alpha = 0.25):
    def predictor(target_prot_go):
        return mundo_predict(target_neighbors,
                             munk_neighbors,
                             n_neighbors,
                             target_prot_go,
                             source_prot_go,
                             alpha,
                             split_source_target = False,
                             n_neigbors_munk = n_neighbors_munk)
    return predictor


def convert_to_dict(npy_neighbors):
    ndict = {}
    n, _ = npy_neighbors.shape
    for i in range(n):
        ndict[i] = npy_neighbors[i, :]
    return ndict


"""
We are assuming the networks are always represented using the ENTREZ protein ids, which are essentially integers.
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help = "The folder where the input files exist")
    parser.add_argument("--go_folder")
    parser.add_argument("--output_folder", help  = "The name of the output folder")
    parser.add_argument("--network_source", help = "The name of the source network, which is inside the input_folder: no extensions on the name. If the name of the source file is file.txt, you input only file here")
    parser.add_argument("--network_target", help = "The name of the target network, which is inside the output folder as well: same naming convention as --network_source")
    parser.add_argument("--munk_name", help = "The name of the Munk coembedding network, without extension")
    parser.add_argument("--go_type", default = "F", choices = ["P", "F", "C"])
    parser.add_argument("--min_level_tar", default = 5, type = int)
    parser.add_argument("--min_prot_tar", default = 50, type = int)
    parser.add_argument("--src_org_id", type = int)
    parser.add_argument("--tar_org_id", type = int)
    parser.add_argument("--n_neighbors", type = int, default = 20)
    parser.add_argument("--n_neighbors_munk", type = int, default = 10)
    parser.add_argument("--rbf_smoothing", default = "0.1")
    parser.add_argument("--verbose", action = "store_true", default = False)
    parser.add_argument("--alpha", default = 0.25, type = float)
    return parser.parse_args()


def main(args):
    def log(strng):
        if args.verbose:
            print(strng)
    
    munk_url = f"{args.input_folder}/{args.munk_name}"
    tar_url = f"{args.input_folder}/{args.network_target}"
    src_url = f"{args.input_folder}/{args.network_source}"
    
    src_dsd_map = {}
    tar_dsd_map = {}
    
    if not os.path.exists(f"{tar_url}.dsd.rbf_{args.rbf_smoothing}.npy"):
        log(f"{tar_url}.dsd.rbf_{args.rbf_smoothing}.npy not found! RBF for {args.rbf_smoothing} not yet computed for the target network")
        exit(1)
    else:
        tar_rbf = np.load(f"{tar_url}.dsd.rbf_{args.rbf_smoothing}.npy")
        with open(f"{tar_url}.dsd.json", "r") as jf:
            tar_dsd_map = json.load(jf)
            
    if not os.path.exists(f"{src_url}.dsd.json"):
        print("Source map not found!")
        exit(1)
    else:
        with open(f"{src_url}.dsd.json", "r") as jf:
            src_dsd_map = json.load(jf)
    
    if not os.path.exists(f"{munk_url}.npy"):
        print("Munk coembedding not found")
        exit(1)
    else:
        munk_mat = np.load(f"{munk_url}.npy")
        tar_dsd_map = {}
        with open(f"{tar_url}.dsd.json", "r") as jf:
            tar_dsd_map = json.load(jf)
    
    r_src_dsd_map = {val:key for key, val in src_dsd_map.items()}
    r_tar_dsd_map = {val:key for key, val in tar_dsd_map.items()}
    
    src_nlist = [int(r_src_dsd_map[i]) for i in range(len(r_src_dsd_map)) if r_src_dsd_map[i].isnumeric()]
    tar_nlist = [int(r_tar_dsd_map[i]) for i in range(len(r_tar_dsd_map)) if r_tar_dsd_map[i].isnumeric()]
    
    """
    src_labels, src_go_prots_dict = get_go_lab(args.go_type, 
                                               args.min_level_src, 
                                               args.min_prot_src,
                                               args.src_org_id,
                                               args.go_folder,
                                               src_nlist)
    """
    
    tar_labels, tar_go_prots_dict = get_go_lab(args.go_type, 
                                               args.min_level_tar, 
                                               args.min_prot_tar, 
                                               args.tar_org_id,
                                               args.go_folder,
                                               tar_nlist)

    src_labels, src_go_prots_dict = get_go_lab_src(args.go_type, 
                                               args.src_org_id,
                                               args.go_folder,
                                               set(tar_labels),
                                               src_nlist)
    
    src_prot_go = get_prot_go_dict(src_go_prots_dict, src_dsd_map)
    tar_prot_go = get_prot_go_dict(tar_go_prots_dict, tar_dsd_map)
    
    
    # Get neighbors
    tar_neighbors = np.argsort(-tar_rbf, axis = 1)[:, :args.n_neighbors]
    munk_neighbors = np.argsort(-munk_mat, axis = 1)[:, :args.n_neighbors]
    
    
    munk_neigh_dict = convert_to_dict(munk_neighbors)
    tar_neigh_dict  = convert_to_dict(tar_neighbors)
    
    results = {}         
    accs = kfoldcv(5,
              tar_prot_go,
              construct_predictor_mundo(tar_neigh_dict,
                                        munk_neigh_dict,
                                        src_prot_go,
                                        n_neighbors = args.n_neighbors,
                                        n_neighbors_munk = args.n_neighbors_munk,
                                        alpha = args.alpha)
              )
    log(f"Accuracies: mean= {np.average(accs)}, std= {np.std(accs)}")
    results["acc"] = accs
    
    f1  = kfoldcv_with_pr(5,
              tar_prot_go,
              construct_predictor_mundo(tar_neigh_dict,
                                        munk_neigh_dict,
                                        src_prot_go,
                                        n_neighbors = args.n_neighbors,
                                        n_neighbors_munk = args.n_neighbors_munk,
                                        alpha = args.alpha)
              )
    log(f"F1max: mean= {np.average(f1)}, std= {np.std(f1)}")
    results["f1"] = f1
    
    res = pd.DataFrame(results)
    res.to_csv(f"{args.output_folder}/{args.go_type}_k_{args.n_neighbors}_alpha_{args.alpha}.tsv", sep = "\t")
    
if __name__ == "__main__":
    main(parse_args())
