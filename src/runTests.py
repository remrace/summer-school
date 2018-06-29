import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import SynGraph as syn
import VizGraph as viz
import matplotlib.pyplot as plt
import DsnNode as dsnNode
import GraphCluster as gc

def TestGraphCluster():
                
        mySeed = 37
        width = 3
        
        np.random.seed(mySeed)
        #RG = syn.InitRandom(width)
        RG = syn.InitSimple(width)
        viz.DrawGraph(RG, title='original')    

        CCLabels, CCE, param = dsnNode.Minimize(RG, nodeType='CC') 
        viz.DrawGraph(RG, labels=CCLabels, title='CC labels')         

        KLLabels, KLE = gc.Minimize(RG, width, minType='kl')
        viz.DrawGraph(RG, labels=KLLabels, title='KL labels')         

        LPLabels, LPE = gc.Minimize(RG, width, minType='lp')
        viz.DrawGraph(RG, labels=LPLabels, title='LP labels')                 
        
        
        print("CC: " + str(CCE))
        print("KL: " + str(KLE))
        print("LP: " + str(LPE))
        
        plt.show() 
                        


def TestWatershed():        
        mySeed = 0
        #for mySeed in range(100):
        np.random.seed(mySeed)
        rg = syn.InitRandom(3)    
        viz.DrawGraph(rg, title='original')    

        orig = seg.FindFrustrated(rg)
        
        ##############################
        ## WC Labeling         
        wg = seg.GetWatershedGraph(rg)
        viz.DrawGraph(wg, title='watershed')    
        
        ws = seg.FindFrustrated(wg)
        
        print("Seed " + str(mySeed) + " Orig : " + str(orig) + " and WS : " + str(ws)) 
        #wcLabels = seg.GetLabelsAtThreshold(wg,0)
        #wcEnergy = seg.GetLabelEnergy(rg, wcLabels)
        #print("WC Labeling: " + str(wcEnergy))    
        #viz.DrawGraph(wg, labels=wcLabels, title='WC labels')

        plt.show() 

def TestKruskal():
    #np.random.seed(1)
    WG = syn.InitRandom(10, 1, False)        
    viz.DrawGraph(WG, title='original')    

    # Get the labeling at threshold 0 and calculate energy
    L = seg.GetLabelsAtThreshold(WG,theta=0)
    E = seg.GetLabelEnergy(WG, L)
    print("Energy at zero is " + str(E))
    viz.DrawGraph(WG, labels=LG, title='original')    
    
    
    # Find the best threshold and check it 
    thresh, minE = seg.FindMinEnergyThreshold(WG)
    print("Minimum energy at " + str(thresh) + " is " + str(minE))    
    # check to see if that's correct
    L = seg.GetLabelsAtThreshold(WG,theta=thresh)
    E2 = seg.GetLabelEnergy(WG, L)
    print("Energy check: " + str(E2))
    viz.DrawGraph(WG, labels=LG, title='bestThresh')    

    plt.show() 

def TestBansal():
    G = syn.InitRandom(10, makeInt=True)        

    WG = G.copy()    
    WG.remove_edges_from([(u,v) for (u,v,d) in  WG.edges(data=True) if d['weight'] < 0.0])
    
    viz.DrawGraph(WG, title='original')    
    
    print("Trying Bansal")
    ban = bansal.BansalSolver(WG)
    components = ban.run()
    labels = dict()
    ci = 1
    for component in components:
        for v in component:
            labels[v] = ci
        ci = ci + 1

    E = seg.GetLabelEnergy(G, labels)
    print("Bansal energy is " + str(E))

    L = seg.GetLabelsAtThreshold(G,theta=0)
    E = seg.GetLabelEnergy(G, L)
    print("CC at zero is " + str(E))

    thresh, minE = seg.FindMinEnergyThreshold(G)
    print("CC min is " + str(minE))    
    
    plt.show() 
    print("Done")

if __name__ == '__main__':
        
    #TestWatershed()    
    #TestKruskal()    
    #TestBansal()
    #TestBest()
    TestGraphCluster()
    
