#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import os
import sys
sys.path.append(os.getcwd())
'''
Reference implementation of node2vec. 
Author: Aditya Grover
For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''
import argparse
import numpy as np
import networkx as nx
import node2vec as nv
from gensim.models import Word2Vec
from datetime import datetime
from gmundo.prediction.go_process import get_go_labels
import pandas as pd

def default_args(args = {}):
	keys   = ["input", "dimensions", "walk-length", "num-walks", "window-size", "iter", "workers", "p", "q", "weighted", "unweighted", "directed", "undirected"]
	values = ["karate.edgelist",
		  50,
		  80,
		  10,
		  10,
		  1,
		  8,
		  1,
		  1,
		  True,
		  False,
		  False,
		  True]
	for i in range(len(keys)):
		key   = keys[i]
		value = values[i]
		if key not in args:
			args[key] = value
	return args
		
def read_graph(args):
	'''
	Reads the input network in networkx.
	'''
	if args["weighted"]:
		G = nx.read_edgelist(args["input"], nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args["input"], nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args["directed"]:
		G = G.to_undirected()

	return G

def learn_embeddings(walks, args):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	args -- A dictionary
	    dimensions - 
	'''
	walks = [list(map(str, walk)) for walk in walks]
	print(args)
	model = Word2Vec(walks, vector_size=args["dimensions"], window=args["window-size"], min_count=0, sg=1, workers=args["workers"]) #, iter=args["iter"])
	print("Here")
	fname =  args["intermediate_file_loc"]# datetime.now().strftime("%d-%m-%Y-%H-%M-%S.txt")
	model.wv.save_word2vec_format(fname)
	with open(fname, "r") as fp:
		emb = []
		id  = 0
		node_map = {}
		for line in fp:
			words                   = line.rstrip().lstrip().split()
			node_map[int(words[0])] = id
			id                     += 1
			vec                     = [float(f) for f in words[1: ]]
			emb.append(vec)
	s_list = []
	for i in range(len(node_map)):
		s_list.append(node_map[i])
	emb = np.array(emb)
	return emb[s_list]


def compute_embedding(edgelist, args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	args = default_args(args)
	nx_G = nx.Graph()
	nx_G.add_weighted_edges_from(edgelist)
	G = nv.Graph(nx_G, args["directed"], args["p"], args["q"])
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args["num-walks"], args["walk-length"])
	return learn_embeddings(walks, args)


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
                                     verbose = True)
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

