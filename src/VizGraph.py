import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import itertools#some function definitions


def DrawLabeledGraph(G, L, title=''):
    #pos specifies the location of each vertex to be drawn
    pos = {x: x for x in list(G)}
    #this is to make the drawing region bigger
    plt.figure(figsize=(4,4)) 
    #drawing nodes    
    nodecolor = [L[key] for key in sorted(L.keys())]
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color = nodecolor, cmap = plt.cm.tab20c)
    #drawing labels (locations) of the nodes
    nx.draw_networkx_labels(G, pos, nx.get_node_attributes(G,'intensity'))
    #drawing edges and theirs labels ('weight')
    nx.draw_networkx_edges(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, labels)
    #show the drawing
    plt.axis('off')    
    if title != '':
        fig = plt.gcf()
        fig.canvas.set_window_title(title)

def DrawGraph(G, title=''):
    #pos specifies the location of each vertex to be drawn
    pos = {x: x for x in list(G)}
    #this is to make the drawing region bigger
    plt.figure(figsize=(4,4)) 
    #drawing nodes
    colordict = {node:color for color,comp in enumerate(nx.connected_components(G)) for node in comp}
    nodecolor = [colordict[key] for key in sorted(colordict.keys())]
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color = nodecolor, cmap = plt.cm.tab20c)
    #drawing labels (locations) of the nodes
    nx.draw_networkx_labels(G, pos, nx.get_node_attributes(G,'intensity'))
    #drawing edges and theirs labels ('weight')
    nx.draw_networkx_edges(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, labels)
    #show the drawing
    plt.axis('off')            

    if title != '':
        fig = plt.gcf()
        fig.canvas.set_window_title(title)

