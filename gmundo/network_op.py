import networkx as nx


def read_network_from_biogrid_file(db_file: str, organism_name: str) -> nx.Graph:
    edgeset = set()
    with open(db_file, 'r') as fptr:
        next(fptr)  # skip format description line
        for line in fptr.readlines():
            interaction = line.strip().split('\t')
            organism1 = interaction[len(interaction)-1]
            organism2 = interaction[len(interaction)-2]
            if organism1 != organism2 or organism1.lower() != organism_name.lower():
                continue  # skip interactions with other organisms
            # read entrezgene ids
            src = int(interaction[1])
            dst = int(interaction[2])
            if src == dst:
                continue  # ignore self-loops
            edgeset.add((src, dst))

    G = nx.Graph()
    G.add_edges_from(edgeset)

    # return largest connected component
    return G.subgraph(max(nx.connected_components(G), key=len))
