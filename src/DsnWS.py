import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import VizGraph as viz
import DsnNode as dsnNode
import matplotlib.pyplot as plt



def Minimize(WG, numSteps=None):
    
    if numSteps is None:
        numSteps = 1
    
    print("DsnWS: Minimizing...")
    params = dict()
    labels = dict()
    energy = dict()
    
    labels[0], energy[0], params[0] = dsnNode.Minimize(WG, 'WS')

    NG = dsnNode.InitGraph(WG, labels[0])
    

    finalLabels = labels[0]
    finalEnergy = energy[0]

    print("DsnTree: Done")
    return finalLabels, finalEnergy, params



