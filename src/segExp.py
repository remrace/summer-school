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
import SegBSDS as bsds 
import SegTF as segTrain

def TestExp1():
    print("Loading")
    # This gets 100 by 100
    (img, seg) = bsds.LoadTrain(0)    
    (img1, seg1, img2, seg2) = bsds.ScaleAndCropData(img, seg)    
    
    bsds.VizTrainTest(img1, seg1, img2, seg2)
    plt.show()

    print("Training")
    segTrain.Train(img2, seg2)    

    
    

if __name__ == '__main__':
    print("Init")
    TestExp1()
    print("Exit")
