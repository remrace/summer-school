import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import itertools#some function definitions
from PIL import Image
import matplotlib.image as mpimg


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

#-----------------------------------------------

def image_to_graph(image):
    img = mpimg.pil_to_array(image)
    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap='gray')
    G = nx.grid_2d_graph(img.shape[0], img.shape[1])
    nx.set_node_attributes(G,{u:{'intensity':v} for u,v in np.ndenumerate(img)})
    for u, v, d in G.edges(data = True):
        d['weight'] = int(np.percentile(img,5)) - abs(np.subtract(int(img[u]), int(img[v]))) + 0.5
    return G

def viz_segment(label, size_X = None, size_Y = None):
    #get the size
    if (size_X == None or size_Y == None):
        size_X = 1 + max([coord[0] for coord in label.keys()])
        size_Y = 1 + max([coord[1] for coord in label.keys()])
    I = np.zeros((size_X,size_Y))
    for coord, z in label.items():
        I[coord] = z
    #plt.figure(figsize=(4,4))
    #plt.imshow(I, cmap = 'hot')
    print (I)