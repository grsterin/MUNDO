import argparse
from dsd_utils import get_go_lab, get_prot_go_dict
import os
from gmundo.prediction.scoring import kfoldcv, kfoldcv_with_pr
from gmundo.prediction.predict import mundo_predict
import pandas as pd
import numpy as np
import json


def construct_predictor_dsd(target_neighbors, n_neighbors=20):
    def predictor(target_prot_go):
        """
        MUNDO with munk weight set to 0 is basically pure DSD,
        so we can just reuse mundo_predict method here
        """
        return mundo_predict(target_neighbors,
                             {},
                             n_neighbors,
                             target_prot_go,
                             {},
                             0.0)
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
    parser.add_argument("--organism_name", help = "The name of the organism network, which is inside the input_folder: no extensions on the name. If the name of the source file is file.txt, you input only file here")
    parser.add_argument("--go_type", default = "F", choices = ["P", "F", "C"])
    parser.add_argument("--min_level_tar", default = 5, type = int)
    parser.add_argument("--min_prot_tar", default = 50, type = int)
    parser.add_argument("--org_id", type = int)
    parser.add_argument("--n_neighbors", type = int, default = 20)
    parser.add_argument("--rbf_smoothing", default = "0.1")
    parser.add_argument("--verbose", action = "store_true", default = False)
    return parser.parse_args()


def main(args):
    def log(strng):
        if args.verbose:
            print(strng)
    
    network_url = f"{args.input_folder}/{args.organism_name}"

    if not os.path.exists(f"{network_url}.dsd.rbf_{args.rbf_smoothing}.npy"):
        log(f"{network_url}.dsd.rbf_{args.rbf_smoothing}.npy not found! RBF for {args.rbf_smoothing} not yet computed for the f{args.organism_name} network")
        exit(1)
    else:
        tar_rbf = np.load(f"{network_url}.dsd.rbf_{args.rbf_smoothing}.npy")
        with open(f"{network_url}.dsd.json", "r") as jf:
            tar_dsd_map = json.load(jf)
            
    r_dsd_map = {val: key for key, val in tar_dsd_map.items()}
    node_list = [int(r_dsd_map[i]) for i in range(len(r_dsd_map)) if r_dsd_map[i].isnumeric()]

    tar_labels, tar_go_prots_dict = get_go_lab(args.go_type,
                                               args.min_level_tar,
                                               args.min_prot_tar,
                                               args.org_id,
                                               args.go_folder,
                                               node_list)

    tar_prot_go = get_prot_go_dict(tar_go_prots_dict, tar_dsd_map)
    
    tar_neighbors = np.argsort(-tar_rbf, axis=1)[:, :args.n_neighbors]
    tar_neigh_dict = convert_to_dict(tar_neighbors)
    results = {}
    accs = kfoldcv(5,
                   tar_prot_go,
                   construct_predictor_dsd(tar_neigh_dict, n_neighbors=args.n_neighbors)
                   )
    log(f"Accuracies: mean= {np.average(accs)}, std= {np.std(accs)}")
    results["acc"] = accs
    
    f1 = kfoldcv_with_pr(5,
                         tar_prot_go,
                         construct_predictor_dsd(tar_neigh_dict, n_neighbors=args.n_neighbors)
                        )
    log(f"F1max: mean= {np.average(f1)}, std= {np.std(f1)}")
    results["f1"] = f1
    
    res = pd.DataFrame(results)
    res.to_csv(f"{args.output_folder}/{args.go_type}_k_{args.n_neighbors}.tsv", sep = "\t")


if __name__ == "__main__":
    main(parse_args())
