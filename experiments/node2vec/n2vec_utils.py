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
from gensim.models import Word2Vec
from datetime import datetime
from gmundo.prediction.go_process import get_go_labels
import pandas as pd
import random


# ----------------------------------------- N2VEC CODE -------------------------------------------------- #



class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]



# ----------------------------------------- N2VEC CODE -------------------------------------------------- #


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
	model = Word2Vec(walks, size=args["dimensions"], window=args["window-size"], min_count=0, sg=1, workers=args["workers"], iter=args["iter"])
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
	G = Graph(nx_G, args["directed"], args["p"], args["q"])
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

