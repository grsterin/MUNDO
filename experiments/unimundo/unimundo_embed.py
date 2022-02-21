import numpy as np
import argparse 
from gmundo.linalg import compute_dsd_embedding
from gmundo.coembed import coembed_networks
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import laplacian_kernel
from unimundo_utils import read_mapping
import json
from gmundo.network_op import read_network_from_biogrid_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_folder", help="Current working folder.")
    parser.add_argument("--biogrid_folder", help="Folder with biogrid files.")
    parser.add_argument("--source_biogrid_file")
    parser.add_argument("--target_biogrid_file")
    parser.add_argument("--src_organism_name", help="Full latin source organism name, e.g., 'homo sapiens'")
    parser.add_argument("--target_organism_name", help="Full latin source organism name, e.g., 'mus musculus'")
    parser.add_argument("--mapping", help="Name of network mapping file without extension, e.g. 'munk'")
    parser.add_argument("--mapping_num_of_pairs", type=int, default=300,
                        help="Number of aligned node pairs to be used for coembedding")
    parser.add_argument("--construct_dsd", action="store_true")
    parser.add_argument("--construct_dsd_dist", action="store_true")
    parser.add_argument("--construct_kernel", action="store_true")
    parser.add_argument("--laplacian_param", default=0.1, type=float)
    parser.add_argument("--save_munk_matrix", action="store_true")
    return parser.parse_args()


def main(args):

    def log(strng):
        if args.verbose:
            print(strng)

    src_file = f"{args.working_folder}/{args.src_organism_name.replace(' ', '-')}"
    target_file = f"{args.working_folder}/{args.target_organism_name.replace(' ', '-')}"

    log("Reading networks from BIOGRID files")
    gsrc = read_network_from_biogrid_file(args.source_biogrid_file, args.src_organism_name)
    gtar = read_network_from_biogrid_file(args.target_biogrid_file, args.target_organism_name)

    src_nodelist = list(gsrc.nodes())
    tar_nodelist = list(gtar.nodes())
    
    src_map = {val: i for i, val in enumerate(src_nodelist)}
    tar_map = {val: i for i, val in enumerate(tar_nodelist)}
    
    with open(f"{src_file}.dsd.json", "w") as jf:
        json.dump(src_map, jf)
    
    with open(f"{target_file}.dsd.json", "w") as jf:
        json.dump(tar_map, jf)

    log("Computing source and target DSD embeddings")
    src_dsd = compute_dsd_embedding(gsrc, src_nodelist)
    tar_dsd = compute_dsd_embedding(gtar, tar_nodelist)
    
    if args.construct_dsd:
        np.save(f"{src_file}.dsd.npy", src_dsd)
        np.save(f"{target_file}.dsd.npy", tar_dsd)

    log("Converting source and target DSD matrices to square form pairwise distance matrices")
    src_ddist = squareform(pdist(src_dsd))
    tar_ddist = squareform(pdist(tar_dsd))
    
    if args.construct_dsd_dist:
        np.save(f"{src_file}.dsd.dist.npy", src_ddist)
        np.save(f"{target_file}.dsd.dist.npy", tar_ddist)

    log("Computing source and target laplacian kernel")
    src_ker = laplacian_kernel(src_ddist, gamma=args.laplacian_param)
    tar_ker = laplacian_kernel(tar_ddist, gamma=args.laplacian_param)
    
    if args.construct_kernel:
        np.save(f"{src_file}.dsd.rbf_{args.gamma}.npy", src_ker)
        np.save(f"{target_file}.dsd.rbf_{args.gamma}.npy", tar_ker)

    log("Computing MUNK coembedding")
    mapping = read_mapping(f"{args.working_folder}/{args.mapping}.tsv", args.mapping_num_of_pairs, src_map, tar_map)
    munk_mat = coembed_networks(src_ker, tar_ker, mapping, verbose=True)
    
    if args.construct_coembed:
        np.save(f"{args.working_folder}/{args.mapping}_lap_ker_{args.laplacian_param}.munk.npy", munk_mat)


if __name__ == "__main__":
    main(parse_args())
