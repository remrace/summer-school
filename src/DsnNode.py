import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import VizGraph as viz
import matplotlib.pyplot as plt


def Minimize(WG, nodeType=None):
    
    if nodeType is None:
        nodeType = 'CC'    
    
    print("DsnNode: Minimizing...")
    param = dict()
    param['nodeType'] = nodeType

    thresh, minE = seg.FindMinEnergyThreshold(WG)
    param['threshold'] = thresh

    LG = seg.GetLabelsAtThreshold(WG, theta=thresh)

    print("DsnNode: Done")
    return LG, param, minE 

