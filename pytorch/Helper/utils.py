# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:48:03 2019

@author: lhuls
"""

import numpy as np

def construct_feed_dict(pkl):
	"""Construct feed dictionary."""
	coord = {'features': pkl[0]}
	pool_idx = {'pool_idx' : pkl[4]}
	faces = {'faces' : pkl[5]}
	lape_idx = {'lape_idx': pkl[7]}
	# laplace = pkl[6]

	edges = []
	for i in range(1,4):
		adj = pkl[i][1]
		edges.append(adj[0])
	
	feed_dict = dict()
	feed_dict.update(coord)
	feed_dict.update(pool_idx)
	feed_dict.update(faces)
	feed_dict.update(lape_idx)
	feed_dict.update({'edges': edges[i] for i in range(len(edges))})
	feed_dict.update({'support1': pkl[1][i] for i in range(len(pkl[1]))})
	feed_dict.update({'support2': pkl[2][i] for i in range(len(pkl[2]))})
	feed_dict.update({'support3': pkl[3][i] for i in range(len(pkl[3]))})

	return feed_dict
#end