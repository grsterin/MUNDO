#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/src")
from gmundo.prediction.predict import mundo_predict
from gmundo.prediction.scoring import kfoldcv, kfoldcv_with_pr
from glide_utils import get_go_lab, get_prot_go_dict
import json
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd

def compute_DSD_RBF(network_file, 
                    t=-1, 
                    lm=1, 
                    normalized=False, 
                    donot_recompute = False):
    save_loc = f"{network_file}.dsd.rbf_0.1.npy"
    A        = None
    if os.path.exists(save_loc) and network_file != "":
        """
        Already Precomputed
        """
        with open(f"{network_file}.dsd.json", "r") as jf:
            nodemap = json.load(jf)
        return np.load(save_loc), nodemap
    elif donot_recompute:
        print("Previous computation of DSD not found! Exiting...")
        sys.exit(1)
    # Read the network file
    df = pd.read_csv(f"{network_file}.txt", sep = "\t", header = None)
    nodemap = {k:i for i, k in enumerate(set(df[0]).union(set(df[1])))}
    edges = pd.replace({0: nodemap, 1:nodemap})[[0,1]].values

    # Compute adjacency matrix
    n = len(nodemap)
    A = np.zeros((n, n))
    for p, q in edges:
        A[p, q] = 1
        A[q, p] = 1

    # Compute X
    n, _ = A.shape
    d = A @ np.ones((n, 1))
    P = A / d
    Identity = np.identity(A.shape[0])
    e = np.ones((A.shape[0], 1))

    D = np.diag(d.flatten())
    # Compute W
    scale = np.matmul(e.T, np.matmul(D, e))[0, 0]
    W = np.multiply(1 / scale, np.matmul(e, np.matmul(e.T, D)))

    up_P = np.multiply(lm, P - W)
    X_ = Identity - up_P
    X_i = np.linalg.pinv(X_)

    if t > 0:
        LP_t = Identity - np.linalg.matrix_power(up_P, t)
        X_i = np.matmul(X_i, LP_t)

    # Compute R
    R =  rbf_kernel(squareform(pdist(X_i)), gamma = 0.1)
    np.save(save_loc, R)
    return R, nodemap


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
                             0.0,
                             split_source_target = False)
    return predictor


def convert_to_dict(npy_neighbors):
    ndict = {}
    n, _ = npy_neighbors.shape
    for i in range(n):
        ndict[i] = npy_neighbors[i, :]
    return ndict


def target_source_neighbor(source_name, 
                        target_name, 
                        landmark_file, # Use it to map target landmarks to source
                        target_RBF, # Use it to find the closest landmarks
                        source_RBF,
                        source_map, # source symbol -> id
                        target_map, # target symbol -> id
                        no_landmarks,
                        no_source_neighbors 
                        ):
    
    # Target to source map
    land_df = pd.read_csv(landmark_file, sep = "\t").head(no_landmarks)
    tar_src_ent = {k:i for k, i in land_df[[target_name, source_name]].values} # target symbol landmarks -> source symbol landmarks

    # Get all the indices of the target landmarks
    target_ids = [target_map[k] for k in target_src_ent.keys()] # id in target space
    target_RBF_land = target_RBF[:, target_ids] 

    # Get the closest target landmark 
    r_target_map   = {i:k for k, i in target_map.items()} # target id -> symbol
    target_RBF_ids = [r_target_map[target_ids[k]] for k in np.argmax(target_RBF_land, axis = 1).flatten()] # symbol in target space. For target protein, find out which landmark is the 
    # closest. And for that target landmark, return its corresponding id.

    # map the target landmark to the source landmark
    target_to_source_landmarks = [source_map[tar_src_ent[k]] for k in target_RBF_ids] # source id 

    # source_landmark_ids 
    src_landmark_ids = [source_map[k] for k in tar_src_ent.values()] # Get all source landmarks
    src_landmark_idmap = {k:i for i, k in enumerate(src_landmark_ids)} # source id to source idd.
    src_landmark_RBF = source_RBF[src_landmark_ids, :] # For each landmark, get `k=no_source_neighbors` nearest neighbors
    src_landmark_RBF_neighbors = np.argsort(-src_landmark_RBF, axis = 1)[:no_source_neighbors]

    """
    Here the munk algorithm does not imply the actual munk algorithm. Just an algorithm that finds the closest source neighbors
    """
    m, _ = target_RBF.shape
    munk_neighbors = np.zeros((m, no_source_neighbors))
    for k in range(m):
        munk_neighbors[m] = src_landmark_RBF_neighbors[src_landmark_idmap[target_to_source_landmarks[m]]]
    return munk_neighbors

"""
We are assuming the networks are always represented using the ENTREZ protein ids, which are essentially integers.

python glide_create_embedding_and_classify.py --input_folder net --go_folder go --output_folder . --network_source bakers_yeast_biogrid --network_target fission_yeast_biogrid --landmark_file fission-yeast-bakers-yeast-with-blast.alignment.tsv 
--landmark_no 100 
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help = "The folder where the input files exist")
    parser.add_argument("--go_folder")
    parser.add_argument("--output_folder", help  = "The name of the output folder")
    parser.add_argument("--network_source", help = "The name of the source network, which is inside the input_folder: no extensions on the name. If the name of the source file is file.txt, you input only file here")
    parser.add_argument("--network_target", help = "The name of the target network, which is inside the output folder as well: same naming convention as --network_source")
    parser.add_argument("--landmark_file", help = "The name of the Munk coembedding network, without extension")
    parser.add_argument("--landmark_no", help = "Landmark number")
    parser.add_argument("--go_type", default = "F", choices = ["P", "F", "C"])
    parser.add_argument("--min_level_tar", default = 5, type = int)
    parser.add_argument("--min_prot_tar", default = 50, type = int)
    parser.add_argument("--src_org_id", type = int)
    parser.add_argument("--tar_org_id", type = int)
    parser.add_argument("--source_neighbors", type = int, default = 10)
    parser.add_argument("--target_neighbors", type = int, default = 20)
    parser.add_argument("--verbose", action = "store_true", default = False)
    parser.add_argument("--alpha", default = 0.25, type = float)
    return parser.parse_args()



def main(args):
    def log(strng):
        if args.verbose:
            print(strng)
    
    tar_url = f"{args.input_folder}/{args.network_target}"
    src_url = f"{args.input_folder}/{args.network_source}"
    landmark_file = f"{args.input_folder}/{args.landmark_file}"
    
    # Compute source and target RBF file
    source_R, s_nodemap = compute_DSD_RBF(src_url, donot_recompute = True)
    target_R, t_nodemap = compute_DSD_RBF(tar_url, donot_recompute = True)
    
    r_s_nodemap = {val:key for key, val in s_nodemap.items()}
    r_t_nodemap = {val:key for key, val in t_nodemap.items()}
    
    src_nlist = [int(r_s_nodemap[i]) for i in range(len(r_s_nodemap)) if r_s_nodemap[i].isnumeric()]
    tar_nlist = [int(r_t_nodemap[i]) for i in range(len(r_t_nodemap)) if r_t_nodemap[i].isnumeric()]
    
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
    tar_neighbors = np.argsort(-source_R, axis = 1)[:, :args.n_neighbors]
    munk_neighbors = target_source_neighbor(args.network_source, 
                        args.network_target, 
                        landmark_file, # Use it to map target landmarks to source
                        target_R, # Use it to find the closest landmarks
                        source_R,
                        s_nodemap, # source symbol -> id
                        t_nodemap, # target symbol -> id
                        args.landmark_no,
                        args.source_neighbors 
                        )
    
    
    munk_neigh_dict = convert_to_dict(munk_neighbors)
    tar_neigh_dict  = convert_to_dict(tar_neighbors)
    
    results = {}         
    accs = kfoldcv(5,
              tar_prot_go,
              construct_predictor_mundo(tar_neigh_dict,
                                        munk_neigh_dict,
                                        src_prot_go,
                                        n_neighbors = args.target_neighbors,
                                        alpha = args.alpha,
                                        n_neighbors_munk = args.source_neighbors)
              )
    log(f"Accuracies: mean= {np.average(accs)}, std= {np.std(accs)}")
    results["acc"] = accs
    
    f1  = kfoldcv_with_pr(5,
              tar_prot_go,
              construct_predictor_mundo(tar_neigh_dict,
                                        munk_neigh_dict,
                                        src_prot_go,
                                        n_neighbors = args.target_neighbors,
                                        alpha = args.alpha,
                                        n_neighbors_munk = args.source_neighbors)
              )
    log(f"F1max: mean= {np.average(f1)}, std= {np.std(f1)}")
    results["f1"] = f1
    
    res = pd.DataFrame(results)
    res.to_csv(f"{args.output_folder}/{args.go_type}_k_{args.n_neighbors}_alpha_{args.alpha}_landmark_{args.landmark_no}.tsv", sep = "\t")
    
if __name__ == "__main__":
    main(parse_args())
