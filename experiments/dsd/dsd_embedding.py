#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import os
import sys
import argparse
import numpy as np
from gmundo.network_op import read_network_from_tsv
from gmundo.linalg import compute_dsd_embedding, squareform, pdist, laplacian_kernel

sys.path.append(os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_folder", help="Working folder containing the files")
    parser.add_argument("--organism_name", help="Name of the organism")
    parser.add_argument("--save_dsd", action="store_true")
    parser.add_argument("--save_dsd_dist", action="store_true")
    parser.add_argument("--save_laplacian", action="store_true")
    parser.add_argument("--laplacian_param", default=0.1, type=float)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main(args):
    def log(strng):
        if args.verbose:
            print(strng)

    base_name = f"{args.working_folder}/{args.organism_name}"
    dsd_name = f"{base_name}.dsd.npy"
    g_network = read_network_from_tsv(f"{base_name}.tsv")

    if os.path.exists(dsd_name):
        log("DSD file already exists! Loading...")
        dsd_matrix = np.load(dsd_name)
    else:
        log("Computing DSD embedding")
        dsd_matrix = compute_dsd_embedding(g_network, list(g_network.nodes()))
        if args.save_dsd:
            log("\tSaving...")
            np.save(dsd_name, dsd_matrix)

    ###################################### COMPUTING DSD DIST ####################################################3#

    dist_name = f"{base_name}.dsd.dist.npy"

    if os.path.exists(dist_name):
        log("Dist file already exists! Loading...")
        dist_matrix = np.load(dist_name)
    else:
        log("Converting DSD matrix to square form pairwise distance matrix")
        dist_matrix = squareform(pdist(dsd_matrix))
        if args.save_dsd_dist:
            log(f"\tSaving...")
            np.save(dist_name, dist_matrix)

    ###################################### COMPUTING LAPLACIAN ######################################################

    lap_name = f"{base_name}.dsd.rbf_{args.laplacian_param}.npy"

    if os.path.exists(lap_name):
        log("Laplacian file already exists! Finished.")
    else:
        log("Computing laplacian kernel")
        src_ker = laplacian_kernel(dist_matrix, gamma=args.laplacian_param)
        if args.save_laplacian:
            log(f"\tSaving...")
            np.save(lap_name, src_ker)


if __name__ == "__main__":
    main(parse_args())
