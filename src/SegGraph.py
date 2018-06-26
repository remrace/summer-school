import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import itertools#some function definitions


def InitRandom(width, intRange, amIperiodic=False):

    G = nx.grid_2d_graph(range(width), range(width), periodic=amIperiodic)
    for u, v, d in G.edges(data = True):
        d['weight'] = (np.random.rand()*2.0 - 1.0) * intRange
        #if d['weight'] == 0:
        #    d['weight'] = np.random.randint(0, 1) * 2 - 1
    
    return G

def GetLabelsAtThreshold(G,theta=0):
    lg = G.copy()    
    lg.remove_edges_from([(u,v) for (u,v,d) in  G.edges(data=True) if d['weight']<=theta])
    L = {node:color for color,comp in enumerate(nx.connected_components(lg)) for node in comp}    
    return L

def GetWatershedGraph(G):
    WG = G.copy()    
    #this function returns the graph WG with new weights of max(min(neighbors))    
    for (u,v,d) in WG.edges(data = True):
        #print(u)        
        uew = [WG[u][ues]['weight'] for ues in WG[u] if ues != v]
        vew = [WG[v][ves]['weight'] for ves in WG[v] if ves != u]
        d['weight'] = d['weight'] - max(min(uew), min(vew))       
    return WG

def GetLabelEnergy(G, L):
    E = 0
    for (u,v,d) in G.edges(data = True):
        if L[u] != L[v]:
            E = E + d['weight']    
    return E


def FindMinEnergyThreshold(WG):
    
    mySets = nx.utils.UnionFind()   
    nextNode = dict()

    for n in WG:
        nextNode[n] = mySets[n]

    #print(rg[(0,0)][(2,2)])
    #if (1,0) in rg[(0,0)]:
    #    print("Yes  way")

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    

    totalPos = sum([d[2] for d in edgeWeights if d[2] > 0])
    totalNeg = sum([d[2] for d in edgeWeights if d[2] < 0])
    accTotal = [0]*len(edgeWeights)

    accTotal[0] = totalPos + totalNeg
    print("Energy 0: " + str(accTotal[0]) + " from Pos: " + str(totalPos) + ", Neg: " + str(totalNeg))

    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    
    ei = 1      # edge index
    lowE = accTotal[0]
    lowT = 0

    for u, v, w in sortedEdges:
        if mySets[u] != mySets[v]:
            accWeight = 0.0
            # traverse nodes in u and look at edges
            # if fully connected we should probably traverse nodes u and v instead
            done = False
            cu = u
            while not done:
                for uev in WG[cu]:                
                    if mySets[uev] == mySets[v]:
                        accWeight = accWeight + WG[cu][uev]['weight']
                cu = nextNode[cu]
                if cu == u:
                    done = True

            # Merge sets
            mySets.union(u, v)
            # Swap next pointers
            tempNext = nextNode[u]
            nextNode[u] = nextNode[v]
            nextNode[v] = tempNext

            accTotal[ei] = accTotal[ei-1] - accWeight            
            print("Energy " + str(ei) + ": " + str(accTotal[ei]) + " from merge " + str(accWeight)) 
            if accTotal[ei] < lowE:
                lowE = accTotal[ei]
                lowT = ei

            ei = ei + 1
        
    # Set threshold half way between
    if lowT > 0:
        lowThreshold = (sortedEdges[lowT][2] + sortedEdges[lowT-1][2]) / 2.0

    print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowThreshold)) 
    return(lowThreshold, lowE)
