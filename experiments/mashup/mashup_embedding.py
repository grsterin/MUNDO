#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
from mashup_utils import generate_As, compute_mashup
import argparse
import numpy as np
import json 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_folder", help = "Working folder containing the files")
    parser.add_argument("--network", help = "Name of the network")
    parser.add_argument("--dim", type = int, default = 1000)
    return parser.parse_args()

def main(args):
    A, nodemap = generate_As([f"{args.working_folder}/{args.network}.tsv"], verbose = True)
    E = compute_mashup(A, reduced_dim = args.dim)
    np.save(f"{args.working_folder}/{args.network}.mashup.dim_{args.dim}.npy", E)
    with open(f"{args.working_folder}/{args.network}.mashup.dim_{args.dim}.json", "w")  as jf:
        json.dump(nodemap, jf)
    
if __name__ == "__main__":
    main(parse_args())
