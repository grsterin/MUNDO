#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/src")
import numpy as np
import argparse
from gmundo.linalg import compute_dsd_embedding
from gmundo.coembed import coembed_networks
from unimundo_utils import read_mapping
import json
from gmundo.network_op import read_network_from_tsv, read_network_from_biogrid_file, write_network_to_tsv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_folder", help="Current working folder.")
    parser.add_argument("--src_biogrid_file", help="Biogrid file with interactions for source organism")
    parser.add_argument("--tgt_biogrid_file", help="Biogrid file with interactions for target organism")
    parser.add_argument("--sc_organism_name", help="Source organism name, e.g. 'Homo sapiens'")
    parser.add_argument("--tgt_organism_name", help="Target organism name, e.g. 'Mus musculus'")
    parser.add_argument("--src_simple_organism_name", help="Simple source organism name, e.g. 'human'")
    parser.add_argument("--tgt_simple_organism_name", help="Simple target organism name, e.g. 'mouse'")
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

    check_all_files([args.src_biogrid_file, args.tgt_biogrid_file])

    log("Preprocessing BIOGRID files")
    src_network_graph, src_biordid_to_entrez_dict = read_network_from_biogrid_file(args.src_biogrid_file,
                                                                                   args.src_organism_name)
    tgt_network_graph, tgt_biordid_to_entrez_dict = read_network_from_biogrid_file(args.tgt_biogrid_file,
                                                                                   args.tgt_organism_name)

    write_network_to_tsv(src_network_graph, f"{args.working_folder}/{args.src_simple_organism_name}.tsv")
    write_network_to_tsv(tgt_network_graph, f"{args.working_folder}/{args.tgt_simple_organism_name}.tsv")

    with open(f"{args.working_folder}/{args.src_simple_organism_name}-biogrid-to-entrez.json", "w") as file:
        json.dump(src_biordid_to_entrez_dict, file)
    with open(f"{args.working_folder}/{args.tgt_simple_organism_name}-biogrid-to-entrez.json", "w") as file:
        json.dump(tgt_biordid_to_entrez_dict, file)

if __name__ == "__main__":
    main(parse_args())
