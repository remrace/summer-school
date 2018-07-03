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
import csv
import pandas as pd
from time import time
from PIL import Image
from PIL import ImageOps
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

def DataGen(num_iter, width):
    for i in range(num_iter):    
        RG = syn.InitRandom(width)
        CCLabels, CCE, param = dsnNode.Minimize(RG, nodeType='CC') 
        WCLabels, WCE, param = dsnNode.Minimize(RG, nodeType='WC') 
        KLLabels, KLE = gc.Minimize(RG, width, minType='kl')
        LPLabels, LPE = gc.Minimize(RG, width, minType='lp')
        yield {'CCE': CCE, 'WCE': WCE, 'KLE': KLE, 'LPE': LPE}

def WriteData(datagen, filename = None):
    if filename == None:
        filename = 'tempdata.csv'
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['CCE', 'WCE', 'KLE', 'LPE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        num=0
        for data in datagen:
            num+=1
            writer.writerow(data)
    print(num)
    


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
    #TestGraphCluster()
    """
    WriteData(DataGen(num_iter = 100, width = 10))
    data = 'tempdata.csv'
    df = pd.read_csv(data)
    df.plot.box()
    plt.show()
    """
    im = Image.open('image.jpg').convert("L").crop((150,150,200,200))
    #op = ImageOps.autocontrast(im, cutoff = 30)

    print('reading image and convert to graph...')
    G = syn.image_to_graph(im)
    print('done')  
    
    print('CCLabels')
    CCLabels, CCE, param = dsnNode.Minimize(G, nodeType='CC')
    print('done')
    print('WCLabels')
    WCLabels, WCE, param = dsnNode.Minimize(G, nodeType='WC')
    
    print('done')
    #print('KLLabels')
    #KLLabels, KLE = gc.Minimize(G, 20, minType='kl')
    #print('done')
    print('LPLabels')
    LPLabels, LPE = gc.Minimize(G, 50, minType='lp')
    print('done')
    
    viz.viz_segment(CCLabels, title = 'CCLabels')
    viz.viz_segment(WCLabels, title = 'WCLabels')
    #viz.viz_segment(KLLabels, title = 'KLLabels')
    viz.viz_segment(LPLabels, title = 'LPLabels')

    print("CC: " + str(CCE))
    print("WC: " + str(WCE))
    #print("KL: " + str(KLE))
    print("LP: " + str(LPE))

    plt.show() 
    print("Exit")
