import pickle 
import numpy as np
import networkx as nx
import SegGraph as seglib
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
import SegBSDS as bsds 
import SegTF as segTrain
from sklearn.decomposition import PCA

def TestExp1():
    print("Loading")
    # This gets 100 by 100
    (img, seg) = bsds.LoadTrain(0)    
    (img1, seg1, img2, seg2) = bsds.ScaleAndCropData(img, seg)    
    
    bsds.VizTrainTest(img1, seg1, img2, seg2)
    plt.show()

    print("Training")
    segTrain.Train(img2, seg2)    




NUM_OUTPUTS = 1
PATCH_SIZE = 21
KERNEL_SIZE = 11
N = (PATCH_SIZE-1) * PATCH_SIZE * 2
D = KERNEL_SIZE * KERNEL_SIZE * 3
    

def ShowPatch(img, seg):
    
    fig=plt.figure(figsize=(4, 2))

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.imshow(seg)
    plt.show()    
    return    
    
def UnrollData(img, seg, G, nlabels, elabels):

    X_train = np.zeros( (N, D), np.float32)
    Y_train = np.zeros( (N, 1), np.float32)
    upto = 0
    for (u,v) in G.edges():
        uimg = img[u[0]:(u[0]+KERNEL_SIZE), u[1]:(u[1]+KERNEL_SIZE), :]
        vimg = img[v[0]:(v[0]+KERNEL_SIZE), v[1]:(v[1]+KERNEL_SIZE), :]
        aimg = (uimg + vimg) / 2.0
        X_train[upto,:] = np.reshape(aimg, (1, D))
        Y_train[upto, 0] = elabels[(u,v)]
        upto = upto + 1

    return (X_train, Y_train)
    
def TestMSTSwap():
    print("Loading")
    # This gets 100 by 100
    (imgOrig, segOrig) = bsds.LoadTrain(0)    
    (img1, seg1, img, seg) = bsds.ScaleAndCropData(imgOrig, segOrig)    
    
    bsds.VizTrainTest(img1, seg1, img, seg)
    
    imgn = (img / img.max()) * 2.0 - 1.0
    
    mySeed = 37
    np.random.seed(mySeed)

    (patchImg, patchSeg, G, nlabels, elabels) = ev.SamplePatch(imgn, seg, PATCH_SIZE, KERNEL_SIZE)
    
    ShowPatch(patchImg, patchSeg)    
    
    (X_train, Y_train) = UnrollData(patchImg, patchSeg, G, nlabels, elabels)
    
    pca = PCA(n_components=D, whiten=True)    
    X_train = pca.fit_transform(X_train)
    
    W = np.ones((D,1), np.float32)
    print(X_train.shape)
    Y_pred = np.matmul(X_train, W)
    print(Y_pred.shape)
    
    upto = 0
    for u, v, d in G.edges(data = True):
        d['weight'] = Y_pred[upto]
        upto = upto + 1

    [posCounts, negCounts, mstEdges, edgeInd, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels)
    mstW = np.zeros( (len(posCounts), 1))

    posError = totalPos
    negError = 0.0
    swapToPos = 0.0
    swapToNeg = 0.0
    for i in range(len(posCounts)):

        posError = posError - posCounts[i]
        negError = negError + posCounts[i]                                        
        mstW[i] = posError - negError
        
        if mstW[i] > 0.0:
            if Y_train[ edgeInd[i] ] < 0.0:
                swapToPos = swapToPos + 1.0
        else:
            if Y_train[ edgeInd[i] ] > 0.0:
                swapToNeg = swapToNeg + 1.0

    print("MST has " + str(len(posCounts)) + " edges")
    print("MST had " + str(swapToPos) + " swap to pos")
    print("MST had " + str(swapToNeg) + " swap to neg")
    plt.show()
    

if __name__ == '__main__':
    print("Init")
    TestExp1()
    #TestMSTSwap()
    print("Exit")
