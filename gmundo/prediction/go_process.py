import numpy as np
import time
import sys
import mygene
from goatools.base import get_godag
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.base import download_ncbi_associations


class GoTool:
    """
    Class used to process GO terms
    """
    def __init__(self, obo_location):
        """
        Location of obo file needed `go-basic.obo` 
        """
        self.godag = get_godag(obo_location, optional_attrs='relationship')
        
    def get_labels(self, filters = None):
        """
        Returns all the GO labels satisfying the certain filter. Filter is a dictionary with terms
        {'max_level' :  int #maximum level of GO terms allowed
        'min_level'  :  int #minimum level of GO terms allowed
        'namespace'  :  Namespace of the GO terms, P, F or C.
        """
        if filters == None:
            return list(self.godag.keys())
        go_terms   = []
        for k in self.godag.keys():
            k_val = self.godag[k]
            if "max_level" in filters:
                if k_val.level > filters["max_level"]:
                    continue
            if "min_level" in filters:
                if k_val.level < filters["min_level"]:
                    continue
            if "namespace" in filters:
                if k_val.namespace != filters["namespace"]:
                    continue
            go_terms.append(k)
        return go_terms

    
    
def get_go_labels(filter_protein, 
                  filter_label, 
                  entrez_labels, 
                  gene_to_go_file,
                  obo_file,
                  species_id,
                  anno_map = lambda x : x,
                  verbose = True):
    """
    Given a list of proteins indexed by their entrez ids, returns the GO terms involved, 
    that satisfies certain conditions outlined by filter_protein and filter_label
    
    """
    
    def log(strng):
        if verbose:
            print(strng)
            
    # Read the gene to go file, for a certain species id
    objanno = Gene2GoReader(gene_to_go_file, taxids=[species_id]) # 9606 for human
    
    # go2geneids essentially is a map that maps GO -> [genes]
    go2geneids = objanno.get_id2gos(namespace=filter_protein["namespace"], 
                                    go2geneids=True)
    
    mg = mygene.MyGeneInfo()
    
    # Use the GoTool described above, to get the filtered labels
    gt          = GoTool(obo_file)
    
    # This is the complete list of GO Labels filtered out only based on thier depth and namespace (P, F or C)
    labels      = gt.get_labels(filter_label)
    labels_dict = {}
    f_labels    = []
    
    for key in labels:
        """
        Check only the filtered labels
        """
        if key not in go2geneids:
            continue
         
        assoc_genes   = go2geneids[key]
        
        # `f_assoc_genes` to filter genes according to the filter_protein
        f_assoc_genes = list(set(assoc_genes).intersection(set(entrez_labels))) 
        
        # Removes the GO terms if it very sparsely annotates the protein list
        
        if len(f_assoc_genes) > filter_protein["lower_bound"]:
            labels_dict[key] = f_assoc_genes
            f_labels.append(key)
    log(f"Number of GO-terms: {len(f_labels)}")
    return f_labels, labels_dict


