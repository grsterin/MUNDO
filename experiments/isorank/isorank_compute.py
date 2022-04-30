#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import sys
import json
import networkx as nx

sys.path.append(os.getcwd()) 
sys.path.append(f"{os.getcwd()}")
from gmundo.alignment import isorank
import pandas as pd
import numpy as np
import argparse

def get_params():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--sequence_score")
    parser.add_argument("--sequence_score_src")
    parser.add_argument("--sequence_score_tar")
    parser.add_argument("--output_mapping")
    parser.add_argument("--no_mapping", default = 1000, type = int)
    parser.add_argument("--alpha", default = 0.5, type = float)
    parser.add_argument("--iterations", default = 3, type = int)
    return parser.parser_args()

def main(args):
    def log(strng):
        if args.verbose:
            print(strng)

    log("Reading Networks...")
    g_source = nx.read_edgelist(args.source)
    g_target = nx.read_edgelist(args.target)

    source_nodes = list(g_source.nodes())
    target_nodes = list(g_target.nodes())

    source_map   = {k:i for i, k in enumerate(source_nodes)}
    target_map   = {k:i for i, k in enumerate(target_nodes)}

    ssrc         = args.sequence_score_src
    star         = args.sequence_score_tar

    log("Reading BLAST sequence scores...")
    dfseq   = pd.read_csv(args.sequence_score, delim_whitespace = True).astype({ssrc: 'str',
                                                                                star: 'str'})

    
    # Filter
    dfseq = dfseq.loc[dfseq[ssrc].isin(source_map) & dfseq[star].isin(target_map), :]
    log(f"\tNumber of positive BLAST scores: {len(dfseq)}")
    dfseq = dfseq.replace({ssrc: source_map, star: target_map})
    max_wt = dfseq["weight"].max()
    dfseq["weight"] = dfseq["weight"]/float(max_wt)
    
    # Sequence Matrix
    log("Running Isorank...")
    E       = np.zeros((len(source_nodes), len(target_nodes)))
    for p, q, w in dfseq.values:
        E[p, q] = w

    # Compute and save isorank
    isorank(g_source,
            g_target,
            source_map,
            target_map,
            args.alpha,
            args.no_mapping,
            iterations = args.iterations,
            saveto = args.output_mapping,
            rowname = ssrc,
            colname = star)

    log("Isorank operation completed!!")

if __name__ == "__main__":
    main(get_params())
