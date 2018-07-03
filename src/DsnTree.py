import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import VizGraph as viz
import DsnNode as dsnNode
import matplotlib.pyplot as plt
import itertools

def Minimize(WG, numSteps=None):
    
    if numSteps is None:
        numSteps = 1
    
    print("DsnTree: Minimizing...")
    params = dict()
    labels = dict()
    energy = dict()
    
    labels[0], energy[0], params[0] = dsnNode.Minimize(WG)


    finalLabels = labels[0]
    finalEnergy = energy[0]

    print("DsnTree: Done")
    return finalLabels, finalEnergy, params

def next_WG(WG, labels):
    H = nx.Graph()
    for x, y in itertools.combinations(set(labels.values()),2):
        if segments_is_connected(WG, labels, x, y):



#find out if two labels is connected
#inputs: the graph G, 2 segments (2 integers)
def segments_is_connected(G, labels, x, y):
    nodes_x = [node for node,color in labels.items() if color == x]
    nodes_y = [node for node,color in labels.items() if color == y]
    return any([G.has_edge(u,v) for u,v in itertools.product(nodes_x,nodes_y)])

#find average 
