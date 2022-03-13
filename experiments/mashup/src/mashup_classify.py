import argparse
import os
import sys
sys.path.append(os.getcwd()) 
sys.path.append(f"{os.getcwd()}/src")

from mashup_utils import read_network_file, get_go_lab, get_prot_go_dict
from gmundo.prediction.scoring import kfoldcv, kfoldcv_with_pr
from gmundo.prediction.predict import perform_binary_OVA
import pandas as pd
import numpy as np
import json

def construct_predictor(E, params = {}, confidence = True):
    def predictor(training_labels):
        tlabels_f = lambda i: (training_labels[i] if i in training_labels else [])
        labels_dict = {}
        for i in range(E.shape[0]):
            l = tlabels_f(i)
            if len(l) != 0:
                labels_dict[i] = l
        return perform_binary_OVA(E, 
                                  labels_dict, 
                                  params = params, 
                                  confidence = confidence)
    return predictor

"""
We are assuming the networks are always represented using the ENTREZ protein ids, which are essentially integers.
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help = "The folder where the input files exist")
    parser.add_argument("--mashup_emb")
    parser.add_argument("--go_folder")
    parser.add_argument("--output_folder", help  = "The name of the output folder")
    parser.add_argument("--go_type", default = "F", choices = ["P", "F", "C"])
    parser.add_argument("--min_level", default = 5, type = int)
    parser.add_argument("--min_prot", default = 50, type = int)
    parser.add_argument("--org_id", type = int, help = "Organism ID")
    parser.add_argument("--verbose", action = "store_true", default = False)
    return parser.parse_args()


def main(args):
    def log(strng):
        if args.verbose:
            print(strng)
    mash_map = {}
    with open(f"{args.input_folder}/{args.mashup_emb}.json", "r") as jf:
        mash_map = json.load(jf)
    
    E        = np.abs(np.load(f"{args.input_folder}/{args.mashup_emb}.npy"))
    
    r_mash_map = {val:key for key, val in mash_map.items()}
    
    nlist = [int(r_mash_map[i]) for i in range(len(r_mash_map))]
    
    labels, go_prots_dict = get_go_lab(args.go_type, 
                                       args.min_level, 
                                       args.min_prot,
                                       args.org_id,
                                       args.go_folder,
                                       nlist)
    
    
    prot_go = get_prot_go_dict(go_prots_dict, mash_map)
                        
    results = {}         
    accs = kfoldcv(5,
              prot_go,
              construct_predictor(E,
                                  confidence = False)
              )
    log(f"Accuracies: mean= {np.average(accs)}, std= {np.std(accs)}")
    results["acc"] = accs
    
    f1  = kfoldcv_with_pr(5,
              prot_go,
              construct_predictor(E)
              )
    log(f"F1max: mean= {np.average(f1)}, std= {np.std(f1)}")
    results["f1"] = f1
    
    res = pd.DataFrame(results)
    res.to_csv(f"{args.output_folder}/{args.go_type}.tsv", sep = "\t")
        
if __name__ == "__main__":
    main(parse_args())
    
                          
                                              
