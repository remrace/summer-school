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
import PIL.Image as Image
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

if __name__ == '__main__':
    print("Init")
    #TestDSN()
    #TestGraphCluster()
    #TestWS()
    #RunExp()
    '''
    image = Image.open('image.jpg').convert("L")
    print('converting to graph...')
    G = syn.InitImage(image)
    print('done')

    print('getting label...')
    labels = seg.multi_spanning(G,steps=200)
    print('done')
    for label in labels[-10:]:
        viz.viz_segment(label)
    
    pickle.dump( labels, open( "save.p", "wb" ) )

    with open('tempdata1.csv', 'r', newline='') as csvfile:
        fieldnames = ['nClasses', 'energy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for label in labels:
            data = {'nClasses': max(label.values())+1, 'energy': seg.GetLabelEnergy(G,label)}
            writer.writerow(data)
    plt.show()
    '''
    image = Image.open('image.jpg').convert("L")
    print('converting to graph...')
    G = syn.InitImage(image)
    print('done')
    labels = pickle.load( open( "save.p", "rb" ) )
    df = pd.DataFrame({'num' : [max(label.values())+1 for label in labels],
                        'energy' : [seg.GetLabelEnergy(G,label) for label in labels]})
    print(df)
    df.plot()
    plt.show()
    print("Exit")
