import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import VizGraph as viz
import DsnNode as dsnNode
import os


def ExportGraph(G, width, fname = None):

    if fname is None:
        fname = "inputGraph.txt"

    fp = open(fname, "w")

    for (u,v,w) in G.edges(data = 'weight'):
        ind1 = u[0]*width + u[1]
        ind2 = v[0]*width + v[1]
        line = str(ind1) + ' ' + str(ind2) + ' ' + str(w) + '\n'
        fp.write(line)
        
    fp.close()


def ImportEdgeLabels(width, fname = None):

    if fname is None:
        fname = "outputGraph.txt"

    edgeLabels = dict()

    with open(fname) as f:            
        for line in f:
            lparts = line.split(' ')
            if (len(lparts)) == 3:
                ind1 = int(lparts[0].strip())
                ind2 = int(lparts[1].strip())
                el  = float(lparts[2].strip())
                u = ( int(ind1/width), ind1 % width )
                v = ( int(ind2/width), ind2 % width )
                edgeLabels[(u,v)] = el
    
    return edgeLabels
    

def Minimize(G, width, theta = None, minType = None):
        
    if minType is None:
        minType = 'kl'          # kl (heurestic) or lp

    if theta is None:
        theta = 0.5     # Used to round the result (note.. if all results are integer already that tells us its optimal for sure (I think))
    
    ExportGraph(G, width)
    
    cmd = "graphCluster -n " + str(len(G)) + " -c " + minType
    print(cmd)
    os.system(cmd)

    edgeLabels = ImportEdgeLabels(width)
    
    # label the nodes
    lg = G.copy()   
    for (u,v,d) in lg.edges(data = True): 
        d['weight'] = edgeLabels[(u,v)]        
    
    lg.remove_edges_from([(u,v) for (u,v,d) in  lg.edges(data=True) if d['weight']>=theta])
    nodeLabels = {node:color for color,comp in enumerate(nx.connected_components(lg)) for node in comp}    
    
    minE = seg.GetLabelEnergy(G, nodeLabels)

    return nodeLabels, minE 



