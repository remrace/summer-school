import pickle 
import numpy as np
import networkx as nx
import SegLib as seg
import matplotlib.pyplot as plt


def FixedThresholdExp():

    rg = seg.random_graph(3, 5, False)    
    seg.DrawGraph(rg, 'original')    

    ##############################
    ## CC Labeling 
    ccLabels = seg.GetLabelsAtThreshold(rg,0)
    ccEnergy = seg.GetLabelEnergy(rg, ccLabels)
    print("CC Labeling: " + str(ccEnergy))    

    seg.DrawLabeledGraph(rg, ccLabels, 'CC labels')


    ##############################
    ## WC Labeling         
    wg = seg.GetWatershedGraph(rg)
    seg.DrawGraph(wg, 'watershed')    
    
    wcLabels = seg.GetLabelsAtThreshold(wg,0)
    wcEnergy = seg.GetLabelEnergy(rg, wcLabels)
    print("WC Labeling: " + str(wcEnergy))    

    seg.DrawLabeledGraph(wg, wcLabels, 'WC labels')

    ##############################
    ## Best Labeling 

    bestEnergy, BL = seg.minimum_energy(rg)
    print("Min Energy: " + str(bestEnergy))    
    seg.DrawGraph(BL, 'Best Labeling')      

    plt.show() 


if __name__ == '__main__':
    

    FixedThresholdExp()    
