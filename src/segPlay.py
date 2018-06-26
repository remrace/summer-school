import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import VizGraph as viz
import matplotlib.pyplot as plt


def FixedThresholdExp():

    rg = seg.InitRandom(3, 1, False)    
    viz.DrawGraph(rg, 'original')    

    ##############################
    ## CC Labeling 
    ccLabels = seg.GetLabelsAtThreshold(rg,0)
    ccEnergy = seg.GetLabelEnergy(rg, ccLabels)
    print("CC Labeling: " + str(ccEnergy))    

    viz.DrawLabeledGraph(rg, ccLabels, 'CC labels')


    ##############################
    ## WC Labeling         
    wg = seg.GetWatershedGraph(rg)
    viz.DrawGraph(wg, 'watershed')    
    
    wcLabels = seg.GetLabelsAtThreshold(wg,0)
    wcEnergy = seg.GetLabelEnergy(rg, wcLabels)
    print("WC Labeling: " + str(wcEnergy))    

    viz.DrawLabeledGraph(wg, wcLabels, 'WC labels')

    ##############################
    ## Best Labeling 

    bestEnergy, BL = seg.minimum_energy(rg)
    print("Min Energy: " + str(bestEnergy))    
    viz.DrawGraph(BL, 'Best Labeling')      

    plt.show() 

def TestKruskal():
    #np.random.seed(1)
    WG = seg.InitRandom(3, 1, False)    
    
    viz.DrawGraph(WG, 'original')    

    thresh, energy = seg.FindMinEnergyThreshold(WG)

    plt.show() 


if __name__ == '__main__':
    print("Init")
    TestKruskal()    
    #FixedThresholdExp()    
    print("Exit")
