import networkx as nx
import numpy as np
from g_mundo.linalg import turbo_dsd

def compute_dsd(graph, nodelist):
    """
    Returns the DSD matrix from the networkx graph
    """
    adj_mat = nx.to_numpy_matrix(graph, nodelist = nodelist)
    return turbo_dsd(adj_mat)

def get_unpickled_networks_url(input_folder, input_prefix):
    """
    Generate the source and target network url from the folder url and prefix url.
    """
    source = f"{input_folder}/{input_prefix}.source.gpickle"
    target = f"{input_folder}/{input_prefix}.target.gpickle"
    return source, target


def get_unpickled_networks(source, target):
    """
    Get source and target network in networkx graph format from the file urls
    """
    return nx.read_gpickle(source), nx.read_gpickle(target)


def get_dsd_url(folder, prefix):
    """
    Generate source and target dsd file from url and precomputed prefix.
    """
    source = f"{folder}/{prefix}.source.dsd.npy"
    target = f"{folder}/{prefix}.target.dsd.npy"
    return source, target


def get_dsd_matrices(source, target):
    """
    Source and target DSD matrix if it is already precomputed.
    """
    return np.load(source, allow_pickle = True), np.load(target, allow_pickle = True)


