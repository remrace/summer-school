import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import VizGraph as viz
import matplotlib.pyplot as plt


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


    #print("DsnNode: Done")
    return L, minE, param

