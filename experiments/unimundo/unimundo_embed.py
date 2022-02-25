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


def main(args):

    def log(strng):
        if args.verbose:
            print(strng)

    log("Reading networks from BIOGRID files")
    g_source = read_network_from_tsv(f"{args.biogrid_tsv_folder}/{args.source_organism_name}.tsv")
    g_target = read_network_from_tsv(f"{args.biogrid_tsv_folder}/{args.target_organism_name}.tsv")

    src_nodelist = list(g_source.nodes())
    tar_nodelist = list(g_target.nodes())

    src_map = {val: i for i, val in enumerate(src_nodelist)}
    tar_map = {val: i for i, val in enumerate(tar_nodelist)}

    source_base_name = f"{args.working_folder}/{args.source_organism_name}"
    target_base_name = f"{args.working_folder}/{args.target_organism_name}"

    with open(f"{source_base_name}.dsd.json", "w") as jf:
        json.dump(src_map, jf)
    
    with open(f"{target_base_name}.dsd.json", "w") as jf:
        json.dump(tar_map, jf)

    log("Computing source and target DSD embeddings")
    src_dsd = compute_dsd_embedding(g_source, src_nodelist)
    tar_dsd = compute_dsd_embedding(g_target, tar_nodelist)
    
    if args.construct_dsd:
        np.save(f"{source_base_name}.dsd.npy", src_dsd)
        np.save(f"{target_base_name}.dsd.npy", tar_dsd)

    log("Converting source and target DSD matrices to square form pairwise distance matrices")
    src_ddist = squareform(pdist(src_dsd))
    tar_ddist = squareform(pdist(tar_dsd))
    
    if args.construct_dsd_dist:
        np.save(f"{source_base_name}.dsd.dist.npy", src_ddist)
        np.save(f"{target_base_name}.dsd.dist.npy", tar_ddist)

    log("Computing source and target laplacian kernel")
    src_ker = laplacian_kernel(src_ddist, gamma=args.laplacian_param)
    tar_ker = laplacian_kernel(tar_ddist, gamma=args.laplacian_param)
    
    if args.construct_kernel:
        np.save(f"{source_base_name}.dsd.rbf_{args.gamma}.npy", src_ker)
        np.save(f"{target_base_name}.dsd.rbf_{args.gamma}.npy", tar_ker)

    log("Computing MUNK coembedding")
    mapping = read_mapping(f"{args.working_folder}/{args.mapping}.tsv", args.mapping_num_of_pairs, src_map, tar_map, separator="\t")
    munk_mat = coembed_networks(src_ker, tar_ker, mapping, verbose=True)
    
    if args.construct_coembed:
        np.save(f"{args.working_folder}/{args.mapping}_lap_ker_{args.laplacian_param}.munk.npy", munk_mat)


if __name__ == "__main__":
    main(parse_args())
