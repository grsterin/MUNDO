import subprocess
from multiprocessing import Pool
import sys
import os
import time
import shlex
import argparse

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
    parser.add_argument("--blast_loc", 
                        help = "location of the blastp binary. If the blastp program is accessible everywhere, just use `blastp`", 
                        default = "blastp")
    parser.add_argument("--no_threads_for_blast", type = int, default = 5, 
                       help = "The number of threads allocated for blast")
    parser.add_argument("--source_fasta", help = "location of the source fasta file", 
                        required = True)
    parser.add_argument("--target_fasta", help = "location of the target fasta file",
                       required = True)
    parser.add_argument("--out_forward", help = "forward output location",
                       required = True)
    parser.add_argument("--out_reverse", help = "backward output location",
                       required = True)
    parser.add_argument("--final_op", help = "final output tsv file",
                       required = True)
    parser.add_argument("--verbose", default = False, action = "store_true")
    return parser.parse_args()

def run_blast_command(blast_loc, 
                      n_threads,
                      query,
                      subject,
                      out_loc,
                      out_fmt):
    """
        RUNNING THE blastp command
    """
    command = [blast_loc, "-out", out_loc, "-outfmt", str(out_fmt), "-query", query, "-subject", subject,  "-num_threads", str(n_threads)]
    try:
        proc    = subprocess.run(command, check = True, capture_output = True)
    except Exception as e:
        raise Exception(f"Error: {str(e)}")
    return "\t OP: --- " + proc.stdout.decode("utf-8") + "\n Errors: ---" + proc.stderr.decode("utf-8")


    
def main(args):
    """
        Main process
    """
    def log(strng):
        if args.verbose:
            print(strng)
    
    pool = Pool()
    
    rp1, rp2 = False, False
    r1       = BlastOP()
    r2       = BlastOP()
    if not os.path.exists(args.out_forward):
        rp1 = True
        p1 = pool.apply_async(run_blast_command, 
                               args = (args.blast_loc,
                                args.no_threads_for_blast,
                                args.source_fasta,
                                args.target_fasta,
                                args.out_forward,
                                      6), callback = r1.update)
    if not os.path.exists(args.out_reverse):
        rp2 = True
        p2 = pool.apply_async(run_blast_command, 
                              (args.blast_loc,
                               args.no_threads_for_blast,
                               args.target_fasta,
                               args.source_fasta,
                               args.out_reverse,
                              6), callback = r2.update)
    print("Waiting for Reciprocal Blast to complete...")
    
    procs = []
    try:
        procs.append(p1)
    except NameError:
        procs = []
    
    try:
        procs.append(p2)
    except NameError:
        pass
    
    
    # Catch exceptions
    while True:
        # Catch exceptions in results not ready
        try:
            ready = [p.ready() for p in procs]
            successful = [p.successful() for p in procs]
            print(successful)
            print(ready)
        except Exception:
            continue
        # If both returned success exit
        if all(successful):
            break
        # If exception occurs report it
        if all(ready) and not all(successful):
            raise Exception(f"Exceptions Raised {[str(p._value) for p in procs if not p.successful()]}")
            break
    pool.close()
    pool.join()
    
    if rp1:
        log(f"Output from FORWARD: \n\n {r1.outstring} \n\n" + "*" * 100 + "\n")
    if rp2:
        log(f"Output from REVERSE: \n\n {r2.outstring} \n\n" + "*" * 100 + "\n")
    
    print("Reciprocal blast completed! loading the blast results...")
    
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
    
    rec.to_csv(args.final_op, sep = "\t", index = False)
    ####
    


        
if __name__ == "__main__":
    main(get_args())