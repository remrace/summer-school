import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import VizGraph as viz
import BansalSolver as bansal
import matplotlib.pyplot as plt
import DsnNode as dsnNode
import CVXSolver as cvx

def TestBest():
        #for mySeed in range(100):
        mySeed = 37
        np.random.seed(mySeed)
        rg = seg.InitRandom(3)

        #viz.DrawGraph(rg, title='original')    
        print("Seed " + str(mySeed))
                
        bestEnergy, BL = cvx.Minimize(rg)

        ## Best Labeling 
        #bestEnergy, BL = seg.minimum_energy(rg)
        #print("  Best Energy: " + str(bestEnergy))    
        #viz.DrawGraph(BL, title='Best Labeling')      

        #print("  Connected Components")
        #labelsCC, paramsCC, energyCC = dsnNode.Minimize(rg, nodeType='CC')
        #viz.DrawGraph(rg, labels=labelsCC, title='CC labels')
        
        #print("  WatershedCut")
        #labelsWC, paramsWC, energyWC = dsnNode.Minimize(rg, nodeType='WC')                
        #viz.DrawGraph(rg, labels=labelsWC, title='WC labels')
        #print("  BE: " + str(bestEnergy))    
        #print("  CC: " + str(energyCC))
        #print(" WC: " + str(energyWC))
        #plt.show()    



def TestWatershed():

    rg = seg.InitRandom(3, 1, False)    
    viz.DrawGraph(rg, title='original')    

    ##############################
    ## CC Labeling 
    ccLabels = seg.GetLabelsAtThreshold(rg,0)
    ccEnergy = seg.GetLabelEnergy(rg, ccLabels)
    print("CC Labeling: " + str(ccEnergy))    

    viz.DrawGraph(rg, labels=ccLabels, title='CC labels')


    ##############################
    ## WC Labeling         
    wg = seg.GetWatershedGraph(rg)
    viz.DrawGraph(wg, title='watershed')    
    
    wcLabels = seg.GetLabelsAtThreshold(wg,0)
    wcEnergy = seg.GetLabelEnergy(rg, wcLabels)
    print("WC Labeling: " + str(wcEnergy))    

    viz.DrawGraph(wg, labels=wcLabels, title='WC labels')

    plt.show() 

def TestKruskal():
    #np.random.seed(1)
    WG = seg.InitRandom(10, 1, False)        
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
    G = seg.InitRandom(10, makeInt=True)        

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
    print("Init")
    
    #TestWatershed()    
    #TestKruskal()    
    #TestBansal()
    TestBest()
    print("Exit")
