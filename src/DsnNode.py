import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import VizGraph as viz
import matplotlib.pyplot as plt


DELTA_TOLERANCE = 1.0e-12

def Minimize(WG, nodeType=None):
    
    if nodeType is None:
        nodeType = 'CC'    
    
    #print("DsnNode: Minimizing...")
    param = dict()
    param['nodeType'] = nodeType

    if nodeType == 'CC':
        thresh, minE = seg.FindMinEnergyThreshold(WG)
        #print("WS Min  Energy: " + str(minE) + "           @ t=" + str(thresh))                    
        param['threshold'] = thresh
        L = seg.GetLabelsAtThreshold(WG, theta=thresh)
        #E2 = seg.GetLabelEnergy(WG, L)
        #print("Energy check: " + str(minE) + " verse " + str(E2))
    
    elif nodeType == 'WC':
        WC = seg.GetWatershedGraph(WG)
        
        thresh, minE = seg.FindMinEnergyThreshold(WC, eval=WG)
        #print("WS Min  Energy: " + str(minE) + "           @ t=" + str(thresh))    
        param['threshold'] = thresh
        L = seg.GetLabelsAtThreshold(WC,theta=thresh)
        #E2 = seg.GetLabelEnergy(WG, L)
        #print("Energy check: " + str(E2))        

    elif nodeType == 'WS':       # fixed threshold WC
        WC = seg.GetWatershedGraph(WG)
        
        param['threshold'] = 0.0
        L = seg.GetLabelsAtThreshold(WC,theta=0.0)
        minE = seg.GetLabelEnergy(WG, L)
        #print("Energy check: " + str(E2))        


    #print("DsnNode: Done")
    return L, minE, param

def FindBestThreshold(NG, RG, WCLabels):

    edgeWeights = [(u,v,w) for (u,v,w) in NG.edges(data = 'weight')]    
    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2])
    
    bestL = WCLabels.copy()
    bestE = seg.GetLabelEnergy(RG, bestL)
    bestT = sortedEdges[0][2] + DELTA_TOLERANCE
    print("E = " + str(bestE) + " from T = " + str(bestT))

    for e in sortedEdges:
        t = e[2] - DELTA_TOLERANCE
        L = seg.GetLabelsAtThreshold(NG,theta=t)
        lowerL = WCLabels.copy()
        for n in lowerL:
            lowerL[n] = L[ lowerL[n] ]
        
        lowerE = seg.GetLabelEnergy(RG, lowerL)
        if lowerE < bestE:
            bestE = lowerE
            bestL = lowerL.copy()
            bestT = t 
            print("E = " + str(bestE) + " from T = " + str(bestT))  

    return bestL, bestE, bestT


def InitGraph(WG, L, initType=None):
    if initType is None:
        initType = 'pool'

    NG = nx.Graph()
    nodes = dict()    
    nodesX = dict()    
    nodesY = dict()    
    nodeMin = dict() 
    nodeMax = dict() 
    nodeCount = dict()
    edgeMin = dict() 
    edgeMax = dict() 
    edgeCount = dict() 

    if initType == 'pool':        
        #print("Pooling graph")
        for n in WG:
            if L[n] in nodes:
                nodes[ L[n] ] = nodes[ L[n] ] + 1
                nodesX[ L[n] ] = nodesX[ L[n] ] + n[0]
                nodesY[ L[n] ] = nodesY[ L[n] ] + n[1]
                
            else:
                nodes[ L[n] ] = 1
                nodesX[ L[n] ] = n[0]
                nodesY[ L[n] ] = n[1]
                #print('Adding Node: ' + str(L[n])) 
                NG.add_node(L[n])
        
        for (u,v,w) in WG.edges(data = 'weight'):
            if ( L[u] != L[v] ):
                # add an edge
                if L[u] < L[v]:
                    a = L[u]
                    b = L[v]
                else:
                    b = L[u]
                    a = L[v]
                    
                if (a, b) in edgeCount:
                    edgeCount[(a, b)] = edgeCount[(a, b)] + 1
                    edgeMin[(a, b)] = min(edgeMin[(a, b)], w)
                    edgeMax[(a, b)] = max(edgeMax[(a, b)], w)
                else:                    
                    NG.add_edge(a, b)                    
                    #print('Adding Edge: ' + str((a,b))) 
                    edgeCount[(a, b)] = 1
                    edgeMin[(a, b)] = w
                    edgeMax[(a, b)] = w                    
            else:
                if L[u] in nodeCount:
                    nodeCount[ L[u] ] = nodeCount[ L[u] ] + 1
                    nodeMin[ L[u] ] = min(nodeMin[ L[u] ], w)
                    nodeMax[ L[u] ] = max(nodeMax[ L[u] ], w)
                else:
                    nodeCount[ L[u] ] = 1
                    nodeMin[ L[u] ] = w
                    nodeMax[ L[u] ] = w

        for (u,v,d) in NG.edges(data = True):
            #print('Edge: ' + str((u,v)))
            # this is to handle nodes that are isolated
            if (u in nodeMax) and (v in nodeMax):
                highest = max(nodeMax[u], nodeMax[v])
            if (u in nodeMax) and (v not in nodeMax):
                highest = nodeMax[u]
            if (u not in nodeMax) and (v in nodeMax):
                highest = nodeMax[v]
            if (u not in nodeMax) and (v not in nodeMax):
                highest = 0

            d['weight'] = edgeMax[(u,v)]  # - highest

        pos = dict()

        for n in NG:
            nodesX[n] = nodesX[n] / nodes[n]
            nodesY[n] = nodesY[n] / nodes[n]
            pos[n] = (nodesX[n], nodesY[n])

    return NG, pos