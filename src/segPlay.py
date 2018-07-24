import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import SegEval as ev
import SynGraph as syn
import VizGraph as viz
import DsnTree as dsnTree
import matplotlib.pyplot as plt
import DsnNode as dsnNode
import GraphCluster as gc
import csv
import pandas as pd
import PIL.Image as Image
import os


def TestRand():
                
        mySeed = 37
        width = 3
        
        np.random.seed(mySeed)
        RG = syn.InitRandom(width)
        #RG = syn.InitSimple(width)
        viz.DrawGraph(RG, title='original')    
        
        myLabels = dict()
        i = 1
        for n in RG:        
            myLabels[n] = i
            i = i + 1        
        
        viz.DrawGraph(RG, labels=myLabels, title='Ind labels')         

        #CCLabels, CCE, param = dsnNode.Minimize(RG, nodeType='CC') 
        #viz.DrawGraph(RG, labels=CCLabels, title='CC labels')         

        (posCounts, negCounts, mstEdges) = ev.FindRandCounts(RG, myLabels)
        #print(posCounts)
        #print(negCounts)
        plt.show() 


def TestEnergyRand():
                
        mySeed = 37
        width = 3
        
        np.random.seed(mySeed)
        RG = syn.InitRandom(width)
        #RG = syn.InitSimple(width)
        viz.DrawGraph(RG, title='original')    

        CCLabels, CCE, param = dsnNode.Minimize(RG, nodeType='CC') 
        viz.DrawGraph(RG, labels=CCLabels, title='CC labels')         
        print(param)
        
        (thresh, bestE, posCounts, negCounts, mstEdges, mstEdgeWeights) = ev.FindMinEnergyAndRandCounts(RG, CCLabels)
        print("Energy: " + str(bestE) + " at threshold " + str(thresh))         
        plt.show() 


def TestGraphCluster():
                
        #mySeed = 37
        width = 10
        
        #np.random.seed(mySeed)
        RG = syn.InitRandom(width)
        #RG = syn.InitSimple(width)
        viz.DrawGraph(RG, title='original')    


        WCLabels, WCE, param = dsnNode.Minimize(RG, nodeType='WC') 
        viz.DrawGraph(RG, labels=WCLabels, title='WC labels')         

        WSLabels, WSE, param = dsnTree.Minimize(RG, treeType='WS') 
        viz.DrawGraph(RG, labels=WSLabels, title='WS labels')         
   
        print("CC: " + str(CCE))
        print("WC: " + str(WCE))
        print("WS: " + str(WSE))

        #KLLabels, KLE = gc.Minimize(RG, width, minType='kl')
        #viz.DrawGraph(RG, labels=KLLabels, title='KL labels')         

        #LPLabels, LPE = gc.Minimize(RG, width, minType='lp')
        #viz.DrawGraph(RG, labels=LPLabels, title='LP labels')                         
     
        #print("KL: " + str(KLE))
        #print("LP: " + str(LPE))
        
        #plt.show() 

def TestWS():

    mySeed = 37
    width = 10
        
    np.random.seed(mySeed)
    RG = syn.InitRandom(width)
    #RG = syn.InitSimple(width)
    viz.DrawGraph(RG, title='original')    

    WCLabels, WCE, param = dsnNode.Minimize(RG, nodeType='WS') 
    viz.DrawGraph(RG, labels=WCLabels, title='WC labels')         

    NG, pos = dsnNode.InitGraph(RG, WCLabels)
    viz.Draw2ndGraph(NG, pos, title='2nd Layer')         

    CCLabels, CCE, param = dsnNode.FindBestThreshold(NG, RG, WCLabels)
    viz.DrawGraph(RG, labels=CCLabels, title='CC labels')         

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
    
def RunExp():
    WriteData(DataGen(num_iter = 100, width = 28))
    data = 'tempdata.csv'
    df = pd.read_csv(data)
    df.plot.box()
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

def getGroundTruths():
    #Ground truth got from using LP on simple original images
    #labels is a dict of 'image':label
    labels = pickle.load( open( "./synimage/groundtruth/save.p", "rb" ) )
    #TestExample
    #viz.viz_segment(label=labels['s10.bmp'])
    #plt.show()
    return labels

def LP_GT():
    image = Image.open('./synimage/noise/s1.png').convert("L")
    print('converting to graph...')
    G = syn.InitImage(image)
    print('done')

    LPLabels, LPE = gc.Minimize(G, 100, minType='lp')
    viz.viz_segment(LPLabels)
    
    plt.show()
   

if __name__ == '__main__':
    print("Init")
    #TestDSN()
    #TestGraphCluster()
    #TestGraphCluster()
    #TestWS()
    #RunExp()
    TestRand()
    #TestEnergyRand()
    print("Exit")
