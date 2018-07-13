import os
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


labels = dict()
for filename in os.listdir('./synimage/original'):
    if filename.endswith(".bmp"): 
        image = Image.open(os.path.join('./synimage/original', filename)).convert("L")
        print('converting' + filename + 'to graph...')
        G = syn.InitImage(image)
        print('done')
        LPLabels, LPE = gc.Minimize(G, 100, minType='lp')
        labels.update({filename: LPLabels})

pickle.dump(labels, open( "./synimage/groundtruth/save.p", "wb" ) )

labels = pickle.load( open( "./synimage/groundtruth/save.p", "rb" ) )

#test
viz.viz_segment(label=labels['s10.bmp'])
plt.show()