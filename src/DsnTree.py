import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import VizGraph as viz
import DsnNode as dsnNode
import matplotlib.pyplot as plt


def Minimize(RG, treeType=None):
    
    if treeType is None:
        treeType = 'WS'
    

    if treeType == 'WS':
        WCLabels, WCE, param = dsnNode.Minimize(RG, nodeType='WS') 
        #viz.DrawGraph(RG, labels=WCLabels, title='WC labels')         

        NG, pos = dsnNode.InitGraph(RG, WCLabels)
        #viz.Draw2ndGraph(NG, pos, title='2nd Layer')         

        finalLabels, finalE, finalP = dsnNode.FindBestThreshold(NG, RG, WCLabels)
        
    
    return finalLabels, finalE, finalP



