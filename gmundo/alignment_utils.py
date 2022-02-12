from urllib import parse, request
from typing import Dict, Tuple, List


def map_refseq_to_entrezgene(node_list: List[str], organism: str) -> Dict[str, int]:
    uniprot_acc_params = {'from': 'P_REFSEQ_AC', 'to': 'ACC', 'format': 'tab', 'query': ' '.join(node_list)}
    uniprot_id_params = {'from': 'P_REFSEQ_AC', 'to': 'ID', 'format': 'tab', 'query': ' '.join(node_list)}

    acc_request = request.Request('https://www.uniprot.org/uploadlists/', parse.urlencode(uniprot_acc_params).encode('utf-8'))
    id_request = request.Request('https://www.uniprot.org/uploadlists/', parse.urlencode(uniprot_id_params).encode('utf-8'))

    with request.urlopen(acc_request) as accs, request.urlopen(id_request) as ids:
        next(accs)
        next(ids)
        valid_uniprot_accs = [acc for acc, pid in zip(accs.readlines(), ids.readlines()) if organism.lower() in pid.decode('utf-8').lower()]
        refseq_to_uniprot_mapping = {refseq_acc: uni_id for refseq_acc, uni_id in [acc.decode('utf-8').strip().split('\t') for acc in valid_uniprot_accs]}
        entrezgene_params = {'from': 'ACC', 'to': 'P_ENTREZGENEID', 'format': 'tab', 'query': ' '.join(refseq_to_uniprot_mapping.values())}
        entrezgene_request = request.Request('https://www.uniprot.org/uploadlists/', parse.urlencode(entrezgene_params).encode('utf-8'))

        with request.urlopen(entrezgene_request) as entrezgenes:
            uniprot_to_entrezgene_mapping = {uni_id: entrezgene_id for uni_id, entrezgene_id in [entrezgene.decode('utf-8').strip().split('\t') for entrezgene in entrezgenes]}
            refseq_to_entrezgene_mapping = {refseq: int(uniprot_to_entrezgene_mapping.get(uni_id)) for refseq, uni_id in refseq_to_uniprot_mapping.items() if uni_id in uniprot_to_entrezgene_mapping}

    return refseq_to_entrezgene_mapping


def read_mapping_from_hubalign_alignment_file(alignment_file_path: str,
                                              source_nodelist: List[int],
                                              target_nodelist: List[int],
                                              num_of_pairs: int,
                                              is_need_to_swap_columns: bool = False) -> (List[Tuple[int, int]], List[Tuple[int, int]]):
    mapping = []
    i_mapping = []
    with open(alignment_file_path, 'r') as alignment_file:
        for line in alignment_file.readlines()[:num_of_pairs]:
            splitted_line = line.split(" ")
            if is_need_to_swap_columns:
                mapping.append((int(splitted_line[1]), int(splitted_line[0])))
                i_mapping.append((source_nodelist.index(int(splitted_line[1])),
                                  target_nodelist.index(int(splitted_line[0]))))
            else:
                mapping.append((int(splitted_line[0]), int(splitted_line[1])))
                i_mapping.append((source_nodelist.index(int(splitted_line[0])),
                                  target_nodelist.index(int(splitted_line[1]))))
    return mapping, i_mapping


def seq_sim_file_from_refseq_to_entrezgene(seq_sim_file_path: str,
                                           updated_seq_sim_file_path: str,
                                           left_organism_name: str,
                                           right_organism_name: str):
    left_organism_proteins = set()
    right_organism_proteins = set()
    with open(seq_sim_file_path, 'r') as seq_sim_file:
        for seq_sim_entry in seq_sim_file.readlines():
            tokens = seq_sim_entry.split('\t')
            left_organism_proteins.add(tokens[0])
            right_organism_proteins.add(tokens[1])

    left_organism_refseq_to_entrezgene_mapping = map_refseq_to_entrezgene(list(left_organism_proteins),
                                                                          left_organism_name)
    right_organism_refseq_to_entrezgene_mapping = map_refseq_to_entrezgene(list(right_organism_proteins),
                                                                           right_organism_name)

    updated_entries = []
    with open(seq_sim_file_path, 'r') as seq_sim_file:
        for seq_sim_entry in seq_sim_file.readlines():
            tokens = seq_sim_entry.split('\t')
            if tokens[0] in left_organism_refseq_to_entrezgene_mapping and tokens[1] in right_organism_refseq_to_entrezgene_mapping:
                updated_entries.append((
                    left_organism_refseq_to_entrezgene_mapping[tokens[0]],
                    right_organism_refseq_to_entrezgene_mapping[tokens[1]],
                    tokens[2]
                ))

    with open(updated_seq_sim_file_path, 'w') as file:
        for prot1, prot2, similarity in updated_entries:
            file.write(f'{prot1}\t{prot2}\t{similarity}')
