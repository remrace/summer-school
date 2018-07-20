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
import numpy as np

labels = dict()
gt_affs = dict()
for filename in os.listdir('./synimage/original'):
    if filename.endswith(".bmp"): 
        image = Image.open(os.path.join('./synimage/original', filename)).convert("L")
        print('converting' + filename + 'to graph...')
        G = syn.InitImage(image)
        print('done')
        LPLabels, LPE = gc.Minimize(G, 100, minType='lp')
        labels.update({filename: LPLabels})

        #binary affinity labels
        #for 2D image, there's 1 plane for vertical and 1 plane for horizontal
        xedges = np.zeros(shape=(100,99), dtype=int)
        yedges = np.zeros(shape=(100,99), dtype=int)
        for u,v,w in G.edges(data='weight'):
            if LPLabels[u] == LPLabels[v]:
                if u[0] == v[0]:
                    xedges[u[0],u[1]] = 1
                elif u[1] == v[1]:
                    yedges[u[1],u[0]] = 1
                else:
                    print('something is wrong!!!!!')
        gt_affs.update({filename: np.dstack((xedges,yedges))}) 

target = {'labels': labels, 'gt_affs': gt_affs}
#pickle.dump(labels, open( "./synimage/groundtruth/save.p", "wb" ) )
pickle.dump(target, open( "./synimage/groundtruth/target.p", "wb" ) )
#labels = pickle.load( open( "./synimage/groundtruth/save.p", "rb" ) )

target = pickle.load( open( "./synimage/groundtruth/target.p", "rb" ) )
print(target['labels']['s1.bmp'])
print(target['gt_affs']['s1.bmp'])
print(target['gt_affs']['s1.bmp'].shape)