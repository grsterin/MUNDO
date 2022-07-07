#!/cluster/tufts/cowenlab/.envs/denoise/bin/python

# import subprocess
# import multiprocessing as mp
# from multiprocessing import Pool
import sys
import os
import time
import shlex
import argparse
import pandas as pd

"""
    Used to store the async output
"""
class BlastOP():
    def __init__(self):
        self.outstring = None
    def update(self, out):
        self.outstring = out

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_forward", help = "forward output location",
                       required = True)
    parser.add_argument("--out_reverse", help = "backward output location",
                       required = True)
    parser.add_argument("--final_op", help = "final output tsv file",
                       required = True)
    parser.add_argument("--verbose", default = False, action = "store_true")
    return parser.parse_args()

    
def main(args):
    """
        Main process
    """
    print(f"[!] Number of available cpus: {mp.cpu_count()}")
    def log(strng):
        if args.verbose:
            print(strng)
    
    fwd = pd.read_csv(args.out_forward, sep = "\t", header = None)
    rev = pd.read_csv(args.out_reverse, sep = "\t", header = None)
    
    # Add headers to forward and reverse results dataframes
    headers = ["query", "subject", "identity", "coverage",
           "qlength", "slength", "alength",
           "bitscore", "E-value"]
    fwd.columns = headers
    rev.columns = headers
    
    
    # Create a new column in both dataframes: normalised bitscore
    fwd['norm_bitscore'] = fwd.bitscore/fwd.qlength
    rev['norm_bitscore'] = rev.bitscore/rev.qlength

    # Create query and subject coverage columns in both dataframes
    fwd['qcov'] = fwd.alength/fwd.qlength
    rev['qcov'] = rev.alength/rev.qlength
    fwd['scov'] = fwd.alength/fwd.slength
    rev['scov'] = rev.alength/rev.slength

    # Clip maximum coverage values at 1.0
    fwd['qcov'] = fwd['qcov'].clip_upper(1)
    rev['qcov'] = rev['qcov'].clip_upper(1)
    fwd['scov'] = fwd['scov'].clip_upper(1)
    rev['scov'] = rev['scov'].clip_upper(1)
    
    ## Merge
    rec = pd.merge(fwd, rev[["query", "subject"]],
                   left_on = "subject",
                   right_on = "query",
                   how = "outer")
    
    rec = rec.loc[rec["query_x"] == rec["subject_y"]]
    rec = rec.groupby(['query_x', 'subject_x'], as_index = False).max()

    log("Saving output...")
    rec.to_csv(args.final_op, sep = "\t", index = False)
    ####
    


        
if __name__ == "__main__":
    main(get_args())
