import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import itertools#some function definitions

DELTA_TOLERANCE = 1.0e-12


def InitSimple(width, noiseMag = None):
    if noiseMag is None:
        noiseMag = 0.1

    G = nx.grid_2d_graph(range(width), range(width))
    for u, v, d in G.edges(data = True):
        if u[0] < 2 and u[1] < 2 and v[0] < 2 and v[1] < 2:    
            d['weight'] = 1.0 + (np.random.rand()*2.0 - 1.0) * noiseMag
        else:
            d['weight'] = -1.0 + (np.random.rand()*2.0 - 1.0) * noiseMag
                 
    return G


def InitRandom(width, maxValue=None, makeInt=None, makePeriodic=None):
    if maxValue is None:
        maxValue = 1
    if makeInt is None:
        makeInt = False
    if makePeriodic is None:
        makePeriodic = False

    G = nx.grid_2d_graph(range(width), range(width), periodic=makePeriodic)
    for u, v, d in G.edges(data = True):
        if makeInt:
            d['weight'] = (np.random.randint(0, maxValue+1)*2 - maxValue)
        else:
            d['weight'] = (np.random.rand()*2.0 - 1.0) * maxValue
        #if d['weight'] == 0:
        #    d['weight'] = np.random.randint(0, 1) * 2 - 1        
    
    return G
    

def PartitionRecursive(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in PartitionRecursive(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller


def PartitionSet(theSet):
    for n, p in enumerate(PartitionRecursive(theSet), 1):
        yield p
