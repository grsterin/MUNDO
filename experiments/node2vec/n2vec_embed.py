#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import os
import sys
sys.path.append(os.getcwd()) 
from n2vec_utils import compute_embedding
import pandas as pd
import numpy as np
import json 
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_folder", help = "File where the network is present")
    parser.add_argument("--network")
    parser.add_argument("--dim", type = int, default = 100)
    parser.add_argument("--p", default = 1, type = float)
    parser.add_argument("--q", default = 1, type = float)
    parser.add_argument("--num_walks", default = 10, type = int)
    return parser.parse_args()

def main(args):
    net_url = f"{args.working_folder}/{args.network}.tsv"
    df      = pd.read_csv(net_url, sep = "\t")
    
    if len(df.columns) == 2:
        df["weight"] = 1
    
    df.columns = ["p", "q", "weight"]
    nodelist   = list(set(df["p"]).union(set(df["q"])))
    nodemap    = {k: i for i, k in enumerate(nodelist)}
    
    df = df.replace({"p": nodemap, "q":nodemap})
    edgelist = df.values.tolist()
    
    with open(f"{args.working_folder}/{args.network}_d_{args.dim}_p_{args.p}_q_{args.q}_nw_{args.num_walks}.json", "w") as jf:
        json.dump(nodemap, jf)
    
    nargs      = {}
    nargs["p"] = args.p
    nargs["q"] = args.q
    nargs["dimensions"] = args.dim
    nargs["num_walks"]  = args.num_walks
    nargs["intermediate_file_loc"] = f"{args.working_folder}/{args.network}_text_d_{args.dim}_p_{args.p}_q_{args.q}_nw_{args.num_walks}.n2vec.emb"
    nargs["final_emb"]             = f"{args.working_folder}/{args.network}_d_{args.dim}_p_{args.p}_q_{args.q}_nw_{args.num_walks}.n2vec.npy"
    
    embeddings = compute_embedding(edgelist, nargs)
    
    with open(nargs["intermediate_file_loc"], "r") as ef:
        n_nodes, n_emb = ef.readline().strip().split(" ")
        n_nodes        = int(n_nodes)
        n_emb          = int(n_emb)
        embeddings     = np.zeros((n_nodes, n_emb))
        
        for line in ef:
            emb       = line.strip().split(" ")
            index     = int(emb[0])
            features  = [float(f_i) for f_i in emb[1: ]]
            embeddings[index] = features
    np.save(nargs["final_emb"], embeddings)
    
if __name__ == "__main__":
    main(parse_args())
    
    
    
    
