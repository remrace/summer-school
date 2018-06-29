import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import SynGraph as syn
import VizGraph as viz
import DsnTree as dsnTree
import matplotlib.pyplot as plt
import DsnNode as dsnNode
import GraphCluster as gc

def TestGraphCluster():
                
        mySeed = 37
        width = 10
        
        np.random.seed(mySeed)
        RG = syn.InitRandom(width)
        #RG = syn.InitSimple(width)
        viz.DrawGraph(RG, title='original')    

        CCLabels, CCE, param = dsnNode.Minimize(RG, nodeType='CC') 
        viz.DrawGraph(RG, labels=CCLabels, title='CC labels')         

        WCLabels, WCE, param = dsnNode.Minimize(RG, nodeType='WC') 
        viz.DrawGraph(RG, labels=WCLabels, title='WC labels')         

        KLLabels, KLE = gc.Minimize(RG, width, minType='kl')
        viz.DrawGraph(RG, labels=KLLabels, title='KL labels')         

        LPLabels, LPE = gc.Minimize(RG, width, minType='lp')
        viz.DrawGraph(RG, labels=LPLabels, title='LP labels')                 
        
        
        print("CC: " + str(CCE))
        print("WC: " + str(WCE))
        print("KL: " + str(KLE))
        print("LP: " + str(LPE))
        
        plt.show() 
                        


def TestDSN():
    #np.random.seed(1)
    WG = syn.InitRandom(10, 1, False)
    viz.DrawGraph(WG, title='original')    

    # This is not a valid partition
    maxNeg = seg.GetLowerBound(WG)
    print("Maximum negative " + str(maxNeg))

    # Run our minimizer
    L, params, E = dsnTree.Minimize(WG)

    print("Energy was " + str(E))
    viz.DrawGraph(WG, labels=L, title='dsn')    

    plt.show() 

if __name__ == '__main__':
    print("Init")
    #TestDSN()
    TestGraphCluster()
    print("Exit")
