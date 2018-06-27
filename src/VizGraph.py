import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import itertools#some function definitions


def DrawGraph(G, labels=None, title=None, figSize=None, nodeSize=None):

    if labels is None:
        colordict = {node:color for color,comp in enumerate(nx.connected_components(G)) for node in comp}
        nodecolor = [colordict[key] for key in sorted(colordict.keys())]
    else:        
        nodecolor = [labels[key] for key in sorted(labels.keys())]        

    if figSize is None:
        figSize = 10

    if nodeSize is None:
        nodeSize = 100
            
    #pos specifies the location of each vertex to be drawn
    pos = {x: x for x in list(G)}
    #this is to make the drawing region bigger
    plt.figure(figsize=(figSize,figSize)) 
        
    nx.draw_networkx_nodes(G, pos, node_size = nodeSize, node_color = nodecolor, cmap = plt.cm.tab20c)
    #drawing labels (locations) of the nodes
    #nx.draw_networkx_labels(G, pos, nx.get_node_attributes(G,'intensity'))
    if labels is not None:
        nx.draw_networkx_labels(G, pos, labels, font_size=6) 
    #drawing edges and theirs labels ('weight')
    nx.draw_networkx_edges(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    for edge in labels:
        labels[edge] = "{0:.2f}".format(labels[edge])

    nx.draw_networkx_edge_labels(G, pos, labels, font_size=8)
    
    plt.axis('off')    
    if title is not None:
        fig = plt.gcf()
        fig.canvas.set_window_title(title)
