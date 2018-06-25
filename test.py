#import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import itertools
import utils
from utils import timeit
from multiprocessing import Pool

I = np.ones((7,7))*20
I[0,0:4] = 16
I[0:3,4:7] = 10
I[1,0:2] = 16
I[1,2:4] = 5
I[2,0:5:3] = 10
I[2,1:3] = 5
I[3,1:5] = 10
I[4,2] = 10
I[5,4:6] = 25
I = np.flipud(I)

#define a NetworkX 4-adj connected graph
#define 'weight' of each edge to be randome integer between 1-9
G = nx.grid_2d_graph(range(3), range(4))
for u, v, d in G.edges(data = True):
    d['weight'] = np.random.randint(-10, 10)

utils.drawing(G)
utils.drawing(utils.watershed_affinity(G.copy()))