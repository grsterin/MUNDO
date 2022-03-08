from gmundo.prediction.go_process import get_go_labels


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


def get_prot_go_dict(go_prot_dict, entrez_id_map):
    prot_go = {}
    for l in go_prot_dict:
        for p in go_prot_dict[l]:
            if entrez_id_map[str(p)] not in prot_go:
                prot_go[entrez_id_map[str(p)]] = [l]
            else:
                prot_go[entrez_id_map[str(p)]].append(l)
    return prot_go
