from gmundo.prediction.go_process import get_go_labels
import networkx as nx
import pandas as pd


def get_go_lab(go_type, min_level, min_prot, org_id, go_folder, entrez_proteins):
    GOT = "biological_process" if go_type == "P" else "molecular_function"
    GOT = "cellular_component" if go_type == "C" else GOT
    
    filter_label = {"namespace": GOT, "min_level":min_level}
    filter_prot  = {"namespace": GOT, "lower_bound":min_prot}
    labels, go_prots = get_go_labels(filter_prot, 
                                     filter_label, 
                                     entrez_proteins, 
                                     f"{go_folder}/gene2go", 
                                     f"{go_folder}/go-basic.obo", 
                                     org_id, 
                                     verbose=True)
    return labels, go_prots

def get_go_lab_src(go_type, org_id, go_folder, target_gos, entrez_proteins):
    GOT = "biological_process" if go_type == "P" else "molecular_function"
    GOT = "cellular_component" if go_type == "C" else GOT
    
    filter_label = {"namespace": GOT, "min_level":0}
    filter_prot  = {"namespace": GOT, "target_gos": target_gos}
    labels, go_prots = get_go_labels(filter_prot, 
                                     filter_label, 
                                     entrez_proteins, 
                                     f"{go_folder}/gene2go", 
                                     f"{go_folder}/go-basic.obo", 
                                     org_id, 
                                     verbose=True)
    return labels, go_prots



def get_prot_go_dict(go_prot_dict, entrez_id_map):
    prot_go = {}
    for l in go_prot_dict:
        for p in go_prot_dict[l]:
            if entrez_id_map[str(p)] not in prot_go:
                prot_go[entrez_id_map[str(p)]] = [l]
            else:
                prot_go[entrez_id_map[str(p)]].append(l)
    return prot_go


def read_network_file(network):
    net = nx.read_weighted_edgelist(network)
    return net


def read_mapping(map_file, number_of_pairs, src_map, tar_map, separator):
    df = pd.read_csv(map_file, sep=separator, nrows=number_of_pairs, header = None)
    df[[0, 1]] = df[[0, 1]].astype(str)
    df = df.replace({0: tar_map, 1: src_map})
    return df[[1, 0]].values.tolist()
