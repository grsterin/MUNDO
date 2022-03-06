#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import os
import sys
sys.path.append(os.getcwd()) 
import numpy as np
import argparse 
from gmundo.linalg import compute_dsd_embedding
from gmundo.coembed import coembed_networks
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import laplacian_kernel
from unimundo_utils import read_mapping
import json
from gmundo.network_op import read_network_from_tsv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_folder", help="Current working folder.")
    parser.add_argument("--biogrid_tsv_folder", help="Folder with biogrid networks converted to tsv files.")
    parser.add_argument("--source_organism_name", help="Source organism name, e.g. 'human'")
    parser.add_argument("--target_organism_name", help="Target organism name, e.g. 'mouse'")
    parser.add_argument("--mapping", help="Name of network mapping file without extension, e.g. 'mouse-human-no-blast.alignment'")
    parser.add_argument("--mapping_num_of_pairs", type=int, default=300,
                        help="Number of aligned node pairs to be used for coembedding")
    parser.add_argument("--construct_dsd", action="store_true")
    parser.add_argument("--construct_dsd_dist", action="store_true")
    parser.add_argument("--construct_kernel", action="store_true")
    parser.add_argument("--laplacian_param", default=0.1, type=float)
    parser.add_argument("--save_munk_matrix", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def check_all_files(files):
    for f in files:
        if not os.path.exists(f):
            print(f"File {f} not found, exiting!")
            sys.exit(1)

def main(args):

    def log(strng):
        if args.verbose:
            print(strng)

    check_all_files([f"{args.biogrid_tsv_folder}/{args.source_organism_name}.tsv",
                     f"{args.biogrid_tsv_folder}/{args.target_organism_name}.tsv",
                     f"{args.working_folder}/{args.mapping}.tsv"])
                    
    log("Reading networks from BIOGRID files")
    g_source = read_network_from_tsv(f"{args.biogrid_tsv_folder}/{args.source_organism_name}.tsv")
    g_target = read_network_from_tsv(f"{args.biogrid_tsv_folder}/{args.target_organism_name}.tsv")
    
    source_base_name = f"{args.working_folder}/{args.source_organism_name}"
    target_base_name = f"{args.working_folder}/{args.target_organism_name}"

    
    source_json = f"{source_base_name}.dsd.json"
    target_json = f"{target_base_name}.dsd.json"

    src_map     = None
    tar_map     = None
    if (os.path.exists(source_json) and os.path.exists(target_json)):
        with open(source_json, "r") as jf:
            src_map   = json.load(jf)
            r_src_map = {v:k for k, v in src_map.items()} 
        with open(target_json, "r") as jf:
            tar_map = json.load(jf)
            r_tar_map = {v:k for k, v in tar_map.items()}
        src_nodelist = [r_src_map[i] for i in range(len(r_src_map))]
        tar_nodelist = [r_tar_map[i] for i in range(len(r_tar_map))]
    else:
        src_nodelist = list(g_source.nodes())
        tar_nodelist = list(g_target.nodes())

        src_map = {val: i for i, val in enumerate(src_nodelist)}
        tar_map = {val: i for i, val in enumerate(tar_nodelist)}

        with open(source_json, "w") as jf:
            json.dump(src_map, jf)
    
        with open(target_json, "w") as jf:
            json.dump(tar_map, jf)

    ###################################### COMPUTING DSD #####################################################3

    source_dsd_name = f"{source_base_name}.dsd.npy"
    target_dsd_name = f"{target_base_name}.dsd.npy"
    
    if (os.path.exists(source_dsd_name)):
        log("Source dsd file already exists! Loading...")
        src_dsd = np.load(source_dsd_name)
    else:
        log("Computing source DSD embedding")
        src_dsd = compute_dsd_embedding(g_source, src_nodelist)
        if args.construct_dsd:
            log("\tSaving...")
            np.save(source_dsd_name, src_dsd)
    if (os.path.exists(target_dsd_name)):
        log("Target dsd file already exists! Loading...")
        tar_dsd = np.load(target_dsd_name)
    else:
        log("Computing target DSD embedding")
        tar_dsd = compute_dsd_embedding(g_target, tar_nodelist)
        if args.construct_dsd:
            log("\tSaving...")
            np.save(target_dsd_name, tar_dsd)

            
    ###################################### COMPUTING DSD DIST ####################################################3#
    
    source_dist_name = f"{source_base_name}.dsd.dist.npy"
    target_dist_name = f"{target_base_name}.dsd.dist.npy"

    if (os.path.exists(source_dist_name) and os.path.exists(target_dist_name)):
        log("Dist files already exists! Loading...")
        src_ddist    = np.load(source_dist_name)
        tar_ddist    = np.load(target_dist_name)
    else:
        log("Converting source and target DSD matrices to square form pairwise distance matrices")
        src_ddist = squareform(pdist(src_dsd))
        tar_ddist = squareform(pdist(tar_dsd))
        if args.construct_dsd_dist:
            log(f"\tSaving...")
            np.save(source_dist_name, src_ddist)
            np.save(target_dist_name, tar_ddist)

            
    ###################################### COMPUTING LAPLACIAN ######################################################
    
    source_lap_name  = f"{source_base_name}.dsd.rbf_{args.laplacian_param}.npy"
    target_lap_name  = f"{target_base_name}.dsd.rbf_{args.laplacian_param}.npy"

    if (os.path.exists(source_lap_name) and os.path.exists(target_lap_name)):
        log("Laplacian files already exists! Loading...")
        src_ker    = np.load(source_lap_name)
        tar_ker    = np.load(target_lap_name)
    else:
        log("Computing source and target laplacian kernel")
        src_ker = laplacian_kernel(src_ddist, gamma=args.laplacian_param)
        tar_ker = laplacian_kernel(tar_ddist, gamma=args.laplacian_param)
        if args.construct_kernel:
            log(f"\tSaving...")
            np.save(source_lap_name, src_ker)
            np.save(target_lap_name, tar_ker)
    

    ############################# COMPUTING MUNK ##############################################
    munk_folder = f"{args.working_folder}/{args.mapping}_lap_ker_{args.laplacian_param}.munk.npy"
    if os.path.exists(munk_folder):
        print(f"MUNK matrix already computed! Ending...")
    else:
        log("Computing MUNK coembedding")
        mapping = read_mapping(f"{args.working_folder}/{args.mapping}.tsv", args.mapping_num_of_pairs, src_map, tar_map, separator=" ")
        munk_mat = coembed_networks(src_ker, tar_ker, mapping, verbose=True)
    
        if args.save_munk_matrix:
            log(f"\t Saving...")
            np.save(munk_folder, munk_mat)


if __name__ == "__main__":
    main(parse_args())
