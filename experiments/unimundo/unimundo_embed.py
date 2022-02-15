import networkx as nx
import pandas as pd
import numpy as np
import argparse 
from gmundo.linalg import compute_dsd_embedding
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import laplacian_kernel
from unimundo_utils import read_mapping 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_folder", help = "Current working folder.")
    parser.add_argument("--src_name")
    parser.add_argument("--target_name")
    parser.add_argument("--mapping")
    parser.add_argument("--construct_dsd", default = False, action = "store_true")
    parser.add_argument("--construct_dsd_dist", default = False, action = "store_true")
    parser.add_argument("--construct_kernel", default = False, action = "store_true")
    parser.add_argument("--laplacian_param", default = 0.1, type = Float)
    parser.add_argument("--construct_coembed", default = False, action = "store_true")
    parser.add_argument("--verbose", action = "store_true", default = False)
    return parser.parse_args()

def main(args):
    
    def log(strng):
        if args.verbose:
            print(strng)
            
    src_file = f"{args.working_folder}/{args.src_name}"
    target_file = f"{args.working_folder}/{args.target_name}"
    
    gsrc = read_network_file(f"{src_file}.txt")
    gtar = read_network_file(f"{target_file}.txt")
    
    src_nodelist = list(gsrc.nodes())
    tar_nodelist = list(gtar.nodes())
    
    src_map  = {val: i for i, val in enumerate(src_nodelist)}
    tar_map  = {val: i for i, val in enumerate(tar_nodelist)}
    
    with open(f"{src_file}.dsd.json", "w") as jf:
        json.dump(src_map, jf)
    
    with open(f"{tar_file}.dsd.json", "w") as jf:
        json.dump(tar_map, jf)
    
    src_dsd = comptue_dsd_embedding(gsrc, src_nodelist)
    tar_dsd = compute_dsd_embedding(gtar, tar_nodelist)
    
    if args.construct_dsd:
        np.save(f"{src_file}.dsd.npy", src_dsd)
        np.save(f"{tar_file}.dsd.npy", tar_dsd)
        
        
    src_ddist = squareform(pdist(src_dsd))
    tar_ddist = squareform(pdist(tar_dsd))
    
    if args.construct_dsd_dist:
        np.save(f"{src_file}.dsd.dist.npy", src_ddist)
        np.save(f"{tar_file}.dsd.dist.npy", tar_ddist)
    
    src_ker = laplacian_kernel(src_ddist, gamma = args.gamma)
    tar_ker = laplacian_kernel(tar_ddist, gamma = args.gamma)
    
    if args.construct_kernel:
        np.save(f"{src_file}.dsd.rbf_{args.gamma}.npy", src_ker)
        np.save(f"{tar_file}.dsd.rbf_{args.gamma}.npy", tar_ker)
    
    mapping = read_mapping(f"{args.working_folder}/{args.mapping}.tsv", src_map, tar_map)
    munk_mat = coembed_networks(src_ker, tar_ker, mapping, verbose = True)
    
    if args.construct_coembed:
        np.save(f"{args.working_folder}/{args.mapping}_lap_ker_{args.gamma}.munk.npy", munk_mat)

if __name__ == "__main__":
    main(parse_args())
    
    
    