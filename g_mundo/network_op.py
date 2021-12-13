from .linalg import turbo_dsd
import numpy as np
import networkx as nx
from typing import Any, Dict, List, NewType, Tuple, Set

## Define New Types
ndarray = NewType("numpy ndarray", np.ndarray)
Graph   = NewType("networkx Graph Object", nx.Graph)

############################### CODE START ###############################

def embed_network(network: Graph,
                  nrw: int)-> Tuple[ndarray, List[str], Dict[str, int]]:
    """
    Function that takes in a graph file `network` and an integer value `nrw`,
    representing a DSD constant (default = 1) and returns the dsd matrix, 
    along with the sorted nodelist.
    
    The nodes are sorted in descending order based on their degree in the network.

    If nrw is set to zero, the dsd_matrix is not computed, and the sorted_nodelist
    is returned. 

    `indexed_nodes` is just sorted_nodelist, with the names edited out.
    """
    sorted_nodelist = [node for node, d in
                       sorted(network.degree, key=lambda x: x[1], reverse=True)]
    indexed_nodes = {node.split('.')[0]:i for i, node
                     in enumerate(sorted_nodelist)}
    if nrw:
	adj_matrix = nx.to_numpy_matrix(network, nodelist=sorted_nodelist)
	dsd_matrix = turbo_dsd(adj_matrix, nrw)
	return dsd_matrix, sorted_nodelist, indexed_nodes
    return sorted_nodelist, indexed_nodes

    
def extract_fields(p1: int,
                   p2: int,
                   interaction: str,
                   outer_delimiter: str,
                   inner_delimiter: str=None,
                   p_in: int=None) -> Tuple[str, str]:
    """
    Given an interaction 
    """
    interaction = interaction.split(outer_delimiter)
    if not inner_delimiter and not p_in:
	return interaction[p1], interaction[p2]
    return interaction[p1].split(inner_delimiter)[p_in], interaction[p2].split(inner_delimiter)[p_in]



def map_db_to_refseq(node_set: Set[str],        # Set of proteins
                     db: str,                   # Database
                     organism: str)-> Tuple[Dict[str, str], Dict[str, str]]:
    input_formats = {'BIOGRID': 'BIOGRID_ID',
                     'HUMANNET': 'P_ENTREZGENEID',
                     'BIOPLEX': 'ID',
                     'LEGACY_BIOGRID': 'GENENAME',
		     'DIP': 'DIP_ID',
                     'REACTOME': 'ID',
                     'STRING': 'STRING_ID',
                     'GIANT': 'P_ENTREZGENEID',
                     'GENEMANIA': 'ENSEMBL_ID'}
	
    uniprot_acc_params = {'from': input_formats[db],
                          'to': 'ACC',
                          'format': 'tab',
                          'query': ' '.join(node_set)}
    uniprot_id_params = {'from': input_formats[db],
                         'to': 'ID',
                         'format': 'tab',
                         'query': ' '.join(node_set)}
	
    acc_request = request.Request('https://www.uniprot.org/uploadlists/',
                                  parse.urlencode(uniprot_acc_params).encode('utf-8'))
    id_request = request.Request('https://www.uniprot.org/uploadlists/',
                                 parse.urlencode(uniprot_id_params).encode('utf-8'))
	
    refseq_to_uniprot_mapping, db_to_refseq_mapping = dict(), dict()

    with request.urlopen(acc_request) as accs, request.urlopen(id_request) as ids:
	next(accs); next(ids)
	valid_uniprot_accs = [acc for
                              acc, pid in zip(accs.readlines(), ids.readlines())
                              if organism.lower() in pid.decode('utf-8').lower()]
	db_to_uniprot_mapping = {db_id:uni_id for db_id, uni_id in
                                 [acc.decode('utf-8').strip().split('\t')
                                  for acc in valid_uniprot_accs]}
	refseq_params = {'from': 'ACC',
                         'to': 'P_REFSEQ_AC',
                         'format': 'tab',
                         'query': ' '.join(db_to_uniprot_mapping.values())}
	ref_request = request.Request('https://www.uniprot.org/uploadlists/',
                                      parse.urlencode(refseq_params).encode('utf-8'))

	with request.urlopen(ref_request) as refs:
	    uniprot_to_refseq_mapping = {uni_id : ref_id
                                         for uni_id, ref_id in
                                         [ref.decode('utf-8').strip().split('\t')
                                          for ref in refs]}
	    db_to_refseq_mapping = {k:v
                                    for k,v in
                                    {(db_id, uniprot_to_refseq_mapping.get(uni_id))
                                     for db_id, uni_id in db_to_uniprot_mapping.items()}
                                    if v}

    refseq_to_uniprot_mapping = {v:k for k, v in uniprot_to_refseq_mapping.items()}
    return refseq_to_uniprot_mapping, db_to_refseq_mapping



def build_network(dbFile: str, db: str, organism: str)-> Tuple[Dict[str, str], Graph]:
	if not file_exists(dbFile, 'NETWORK PROCESSING'): exit()
	try:
		edgeset = set()
		with open(dbFile, 'r') as fptr:
			next(fptr) # get rid of format desciption line
			for line in fptr.readlines():
				interaction = line.strip()
				if db == 'STRING': src, dst = extract_fields(0, 1, interaction, ' ')
				elif db == 'DIP': src, dst = extract_fields(0, 1, interaction, '\t', '|', 0)
				elif db == 'BIOGRID': src, dst = extract_fields(3, 4, interaction, '\t')
				elif db == 'LEGACY_BIOGRID': src, dst = extract_fields(0, 1, interaction, '\t')
				elif db == 'REACTOME': src, dst = extract_fields(0, 3, interaction, '\t', ':', -1)
				elif db == 'BIOPLEX': src, dst = extract_fields(2, 3, interaction, '\t')
				elif db == 'GIANT': src, dst = extract_fields(0, 1, interaction, '\t') if interaction.split('\t')[2] < 0.90 else ('!','!')
				else: src, dst = extract_fields(0, 1, interaction, '\t') # HumanNet, geneMANIA
				if src == dst: continue # ignore self-loops
				edgeset.add((src, dst))
	
	except ValueError: print(f'[NETWORK PROCESSING ERROR] File \"{file}\" is saved in an invalid format: expected <src> <dst> ...'); exit()

	nodeset = set(chain(*edgeset)) # get unique nodes
	refseq_to_uniprot_mapping, db_to_refseq_mapping = map_db_to_refseq(nodeset, db, organism)
	refseq_edgeset = {e for e in {(db_to_refseq_mapping.get(src), db_to_refseq_mapping.get(dst)) for src, dst in edgeset} if all(e)}
	
	G = nx.Graph()
	G.add_edges_from(refseq_edgeset)

	return refseq_to_uniprot_mapping, largest_connected_component(G, organism)



