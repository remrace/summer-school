import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import VizGraph as viz
import DsnTree as dsnTree
import matplotlib.pyplot as plt


def TestDSN():
    #np.random.seed(1)
    WG = seg.InitRandom(10, 1, False)
    viz.DrawGraph(WG, title='original')    

    # This is not a valid partition
    maxNeg = seg.GetLowerBound(WG)
    print("Maximum negative " + str(maxNeg))

    # Run our minimizer
    L, params, E = dsnTree.Minimize(WG)

    print("Energy was " + str(E))
    viz.DrawGraph(WG, labels=L, title='dsn')    

    plt.show() 


def TestEnumerate():

    allNodes = list(range(1,10))
    print(allNodes)
    print(allNodes[1:])

    result = seg.PartitionSet(allNodes)
    ri = 0

    for r in result:         
        ri = ri + 1

    print(str(ri))


if __name__ == '__main__':
    print("Init")
    #TestDSN()
    #TestEnumerate()
    print("Exit")
