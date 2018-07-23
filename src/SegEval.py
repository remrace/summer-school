import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import itertools#some function definitions

DELTA_TOLERANCE = 1.0e-12


def GetNodeEdgeLabels(seg):
    
    G = nx.grid_2d_graph(seg.shape[0], seg.shape[1])
    nlabels = dict()
    elabels = dict()
    for (u,v,d) in G.edges(data = True):
        labelu = seg[u[0], u[1]]
        labelv = seg[v[0], v[1]]
        nlabels[u] = seg[u[0], u[1]]
        nlabels[v] = seg[v[0], v[1]]

        if abs(nlabels[u] - nlabels[v]) < 1.0:
            elabels[(u,v)] = 1.0
        else:
            elabels[(u,v)] = -1.0

    return (nlabels, elabels) 


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


def DotProductLabels(a, b):
    ssum = 0.0    
    for key in a: 
        if key in b: 
            ssum = ssum + a[key]*b[key]
            
    return ssum

def GetNumberLabels(a):
    ssum = 0.0    
    for key in a:         
        ssum = ssum + a[key]
            
    return ssum

def CombineLabels(a, b):
    c = a.copy()
    
    for key in b:         
        if key in c:
            c[key] = c[key] + b[key]
        else:
            c[key] = b[key]
            
    return c

def FindRandCounts(WG, gtLabels):
    
    mySets = nx.utils.UnionFind()       
    labelCount = dict()

    for n in WG:        
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
        su = mySets[u]
        sv = mySets[v]
        if su != sv:
            
            labelAgreement = DotProductLabels( labelCount[su], labelCount[sv] )
            numLabelsU = GetNumberLabels( labelCount[su] )
            numLabelsV = GetNumberLabels( labelCount[sv] )
            labelDisagreement = numLabelsU * numLabelsV - labelAgreement
            mstEdges.append( (u,v) )
            posCounts.append(labelAgreement)
            negCounts.append( labelDisagreement)

            #print(str(u) + " has " str(numLabelsU) + ", " + str(v)  and " + str(numLabelsV) + " got to " + str(labelAgreement) + " and " + str(labelDisagreement))    
            totalPos = totalPos + posCounts[-1] 
            totalNeg = totalNeg + negCounts[-1] 
                        
            allLabels = CombineLabels(labelCount[su], labelCount[sv])
            #numAll = GetNumberLabels(allLabels)
            mySets.union(u, v)
            labelCount[ mySets[u] ] = allLabels.copy()

    #print(mstEdges)    
    print("Total Pos: " + str(totalPos) + " Neg: " + str(totalNeg)) 
    return(posCounts, negCounts, mstEdges)



def FindMinEnergyAndRandCounts(WG, gtLabels, evalw=None):
    
    mySets = nx.utils.UnionFind()   
    nextNode = dict()
    labelCount = dict()

    for n in WG:
        nextNode[n] = mySets[n]
        labelCount[n] = dict()
        labelCount[n][ gtLabels[n] ] = 1.0

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    

    if evalw is None:
        evalWeights = edgeWeights
        evalw = WG
    else:
        evalWeights = [(u,v,w) for (u,v,w) in evalw.edges(data = 'weight')]    

    mstEdges = list()
    mstEdgeWeights = list()
    posCountsRand = list()
    negCountsRand = list()
    totalPosRand = 0.0
    totalNegRand = 0.0    
    

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
        su = mySets[u]
        sv = mySets[v]
        if su != sv:
            accWeight = 0.0
            # traverse nodes in u and look at edges
            # if fully connected we should probably traverse nodes u and v instead
            done = False
            cu = u
            while not done:
                for uev in WG[cu]:                
                    if mySets[uev] == mySets[v]:
                        accWeight = accWeight + evalw[cu][uev]['weight']
                cu = nextNode[cu]
                if cu == u:
                    done = True

            labelAgreement = DotProductLabels( labelCount[su], labelCount[sv] )
            numLabelsU = GetNumberLabels( labelCount[su] )
            numLabelsV = GetNumberLabels( labelCount[sv] )
            labelDisagreement = numLabelsU * numLabelsV - labelAgreement
            posCountsRand.append(labelAgreement)
            negCountsRand.append( labelDisagreement)

            #print(str(u) + " has " str(numLabelsU) + ", " + str(v)  and " + str(numLabelsV) + " got to " + str(labelAgreement) + " and " + str(labelDisagreement))    
            totalPosRand = totalPosRand + posCountsRand[-1] 
            totalNegRand = totalNegRand + negCountsRand[-1] 

            mstEdges.append( (u,v) )
            mstEdgeWeights.append( w )
                        
            allLabels = CombineLabels(labelCount[su], labelCount[sv])
            #numAll = GetNumberLabels(allLabels)            

            # Merge sets
            mySets.union(u, v)
            # Merge counts
            labelCount[ mySets[u] ] = allLabels.copy()            
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
    return(lowT, lowE, posCountsRand, negCountsRand, mstEdges, mstEdgeWeights)

def FindBestRandThreshold(posCounts, negCounts, mstEdges, mstEdgeWeights):
    
    sumPosCount = 0.0
    sumNegCount = 0.0
	
    totalPos = sum(posCounts) 
    totalNeg = sum(negCounts)
    localPos = totalPos
    localNeg = 0.0

    # start off with every point in own cluster
    lowE = (localPos + localNeg) / (totalPos + totalNeg)
    lowT = mstEdgeWeights[0] + DELTA_TOLERANCE
    
    for i in range(len(posCounts)):
        localPos = localPos - posCounts[i]
        localNeg = localNeg + negCounts[i]

        newError = (localPos + localNeg) / (totalPos + totalNeg)
		
        if newError < lowE:
            lowE = newError
            lowT = mstEdgeWeights[i] - DELTA_TOLERANCE        

    return(lowT, lowE)
