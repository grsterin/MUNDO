import json
from typing import List, Dict, Tuple
from urllib import request, parse

import networkx as nx
import pandas as pd


def read_network_from_biogrid_file(db_file: str, organism_name: str) -> Tuple[nx.Graph, Dict[int, int]]:
    edgeset = set()
    biogrid_to_entrez_dict = dict()
    with open(db_file, 'r') as fptr:
        next(fptr)  # skip format description line
        for line in fptr.readlines():
            interaction = line.strip().split('\t')
            organism1 = interaction[len(interaction)-1]
            organism2 = interaction[len(interaction)-2]
            if organism1 != organism2 or organism1.lower() != organism_name.lower():
                continue  # skip interactions with other organisms
            # read entrezgene ids
            src_entrez = int(interaction[1])
            dst_entrez = int(interaction[2])

            src_biogr = int(interaction[3])
            dst_biogr = int(interaction[4])

            if src_biogr == dst_biogr:
                continue  # ignore self-loops

            edgeset.add((src_biogr, dst_biogr))
            biogrid_to_entrez_dict[src_biogr] = src_entrez
            biogrid_to_entrez_dict[dst_biogr] = dst_entrez
    G = nx.Graph()
    G.add_edges_from(edgeset)

    return G.subgraph(max(nx.connected_components(G), key=len)), biogrid_to_entrez_dict


def read_biogrid_ids_list_from_biogrid_file(db_file: str, organism_name: str) -> List[str]:
    edgeset = set()
    with open(db_file, 'r') as fptr:
        next(fptr)  # skip format description line
        for line in fptr.readlines():
            interaction = line.strip().split('\t')
            organism1 = interaction[len(interaction)-1]
            organism2 = interaction[len(interaction)-2]
            if organism1 != organism2 or organism1.lower() != organism_name.lower():
                continue  # skip interactions with other organisms
            # read biogrid ids
            src = interaction[3]
            dst = interaction[4]
            if src == dst:
                continue  # ignore self-loops
            edgeset.add((src, dst))
    G = nx.Graph()
    G.add_edges_from(edgeset)
    return list(G.nodes)


def write_network_to_tsv(graph: nx.Graph, file_path: str):
    pd_edgelist: pd.DataFrame = nx.to_pandas_edgelist(graph)
    pd_edgelist.to_csv(file_path, sep="\t", header=False, index=False)


def read_network_from_tsv(tsv_file: str) -> nx.Graph:
    df_network = pd.read_csv(tsv_file, sep="\t", header=None)
    df_network[[0,1]] = df_network[[0,1]].astype(str)
    return nx.from_pandas_edgelist(df_network, 0, 1)


def create_fasta_file_for_biogrid_ids(biogrid_node_ids: List[str], fasta_file: str):
    uniprot_acc_params = {'from': 'BioGRID', 'to': 'UniProtKB', 'ids': ",".join(biogrid_node_ids)}

    acc_request = request.Request('https://rest.uniprot.org/idmapping/run', parse.urlencode(uniprot_acc_params).encode('utf-8'))

    accs = request.urlopen(acc_request)
    job_id = json.loads(accs.read().decode("utf-8"))["jobId"]

    print(job_id)
    status = None
    while status != "FINISHED":
        print(status)
        status_req = request.Request(f"https://rest.uniprot.org/idmapping/status/{job_id}")
        status_resp = request.urlopen(status_req)
        resp_content = json.loads(status_resp.read().decode("utf-8"))
        if "results" in resp_content:
            break
        status = resp_content["jobStatus"]

    res_req = request.Request(f"https://rest.uniprot.org/idmapping/uniprotkb/results/stream/{job_id}?compressed=false&download=true&format=fasta")
    res_resp = request.urlopen(res_req)
    with open(fasta_file, "w") as file:
        val = res_resp.read().decode("utf-8")
        file.write(val)


if __name__ == "__main__":
    biogrid_ids_human = read_biogrid_ids_list_from_biogrid_file(
        "C:\\university\\MUNDO\\data\\biogrid_files\\BIOGRID-ORGANISM-4.4.206.tab3\\BIOGRID-ORGANISM-Mus_musculus-4.4.206.tab3.txt",
        "Mus musculus"
    )
    print(biogrid_ids_human)
    create_fasta_file_for_biogrid_ids(
        biogrid_ids_human,
        "C:\\university\\MUNDO\\data\\biogrid_files\\human-biogrid.fasta"
    )
