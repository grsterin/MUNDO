"""
Program that takes in the source DSD embedding, target DSD embedding, 
and MUNK embedding and performs function prediction.
"""
import argparse
from utils import compute_dsd, get_unpickled_networks_url, get_unpickled_network, get_dsd_url, get_dsd_matrices


def get_args():
    """
    Command Line Arguments
    """
    parser = argparse.ArgumentParser()
    
    """
    GO annotation files
    """
    parser.add_argument("-s", 
                        "--source_go_annotation", 
                        help = "source go annotation")
    
    parser.add_argument("-t", 
                        "--target_go_annotation", 
                        help = "target go annotation")
    
    """
    GO-TYPE:
    P = BIOLOGICAL PROCESS
    F = MOLECULAR FUNCTION
    C = CELLULAR COMPONENT
    A = ALL
    """
    parser.add_argument("--go-type", 
                        default = "F", 
                        choices = ["P", "F", "C", "A"])
    
    
    """
    Input and output prefixes and folders
    """
    parser.add_argument("--input_folder")
    parser.add_argument("--inprefix")
    parser.add_argument("--output_folder")
    parser.add_argument("--outprefix")
    
    """
    Print log and debug messages.
    """
    parser.add_argument("--verbose", default = False, action = "store_true")
    return parser.parse_args()
    
    
def main(args):
    
    verbose = args.verbose
    
    def log(strng):
        if verbose:
            print(strng)
    
    log("Loading DSD matrices...")
    source, target = get_dsd_url(args.input_folder, args.inprefix)
    source_dsd, target_dsd = get_dsd_matrices(source, target)
    
    