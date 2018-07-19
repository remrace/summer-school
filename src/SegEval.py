import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import itertools#some function definitions

DELTA_TOLERANCE = 1.0e-12



def FindMinEnergyThreshold(WG, eval=None):
    
    mySets = nx.utils.UnionFind()   
    nextNode = dict()

    for n in WG:
        nextNode[n] = mySets[n]

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    

    if eval is None:
        evalWeights = edgeWeights
        eval = WG
    else:
        evalWeights = [(u,v,w) for (u,v,w) in eval.edges(data = 'weight')]    

    totalPos = sum([d[2] for d in evalWeights if d[2] > 0])
    totalNeg = sum([d[2] for d in evalWeights if d[2] < 0])
    accTotal = [0]*len(evalWeights)

    accTotal[0] = totalPos + totalNeg
    #print("Energy 0: " + str(accTotal[0]) + " from Pos: " + str(totalPos) + ", Neg: " + str(totalNeg))

    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    
    ei = 1      # edge index
    lowE = accTotal[0]
    lowT = sortedEdges[0][2] + DELTA_TOLERANCE

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
                        accWeight = accWeight + eval[cu][uev]['weight']
                cu = nextNode[cu]
                if cu == u:
                    done = True

            # Merge sets
            mySets.union(u, v)
            # Swap next pointers... this incrementally builds a pointer cycle around all the nodes in the component
            tempNext = nextNode[u]
            nextNode[u] = nextNode[v]
            nextNode[v] = tempNext

            accTotal[ei] = accTotal[ei-1] - accWeight            
            #print("Energy at threshold " + str(w) + ": " + str(accTotal[ei]))
            if accTotal[ei] < lowE:
                lowE = accTotal[ei]
                lowT = w - DELTA_TOLERANCE


            ei = ei + 1
        
    #print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowThreshold)) 
    return(lowT, lowE)


def DotProductLabels(a, b) {
    sum = 0.0    
    for key in a: 
        if key in b: 
            sum = sum + a[key]*b[key]
            
    return sum

def GetNumberLabels(a) {
    sum = 0.0    
    for key in a:         
        sum = sum + a[key]
            
    return sum

def CombineLabels(a, b) {
    c = a.copy()
    
    for key in b:         
        if key in c:
            c[key] = c[key] + b[key]
        else:
            c[key] = b[key]
            
    return c

def FindRandCounts(WG, gtLabels):
    
    mySets = nx.utils.UnionFind()   
    nextNode = dict()
    labelCount = dict()

    for n in WG:
        nextNode[n] = mySets[n]
        labelCount[n] = dict()
        labelCount[n][ gtLabels[n] ] = 1.0


    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    
    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 

    mstEdges = list()
    posCounts = list()
    negCounts = list()
    totalPos = 0.0
    totalNeg = 0.0    
    

    for u, v, w in sortedEdges:
        if mySets[u] != mySets[v]:
            
            labelAgreement = DotProductLabels( labelCount[u], labelCount[v] )
            mstEdges.append( (u,v) )
            posCounts.append(labelAgreement)
            negCounts.append( GetNumberLabels( labelCount[u] ) * GetNumberLabels( labelCount[v] ) - labelAgreement)
            totalPos = totalPos + posCounts[-1] 
            totalNeg = totalNeg + negCounts[-1] 

            allLabels = CombineLabels(labelCount[u], labelCount[v])
            print("Prior to: " + str(mySets[u]))
            mySets.union(u, v)
            print("And after: " + str(mySets[u]))
            # Swap next pointers... this incrementally builds a pointer cycle around all the nodes in the component
            #tempNext = nextNode[u]
            #nextNode[u] = nextNode[v]
            #nextNode[v] = tempNext

            #accTotal[ei] = accTotal[ei-1] - accWeight            
            #print("Energy at threshold " + str(w) + ": " + str(accTotal[ei]))
            #if accTotal[ei] < lowE:
            #    lowE = accTotal[ei]
            #    lowT = w - DELTA_TOLERANCE


            #ei = ei + 1
        
    #print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowThreshold)) 
    return(totalPos, totalNeg)

