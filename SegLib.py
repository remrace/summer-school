#import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import itertools#some function definitions

def drawing(G):
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
    plt.show()
    
    
def minimum_edges(G):
    #this function returns all minimum edges of a graph
    min_edge = min(list(G.edges.data('weight')), key = lambda x : x[2])
    min_edges = [i for i in list(G.edges.data('weight')) if i[2] == min_edge[2]]
    return min_edges

def adjacency_edges_of_subgraph(S,G):
    #this function returns all adjacency edges of a subgragh S of G
    lst = []
    for (u,v,w) in G.edges.data('weight'):
        if S.has_node(u) ^ S.has_node(v):
            lst.append((u,v,w))
    return lst

def average_weight(G):
    #this returns the avarage 'weight' of a graph G
    return np.mean([w for (u,v,w) in G.edges.data('weight')])

from itertools import islice
def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight), k))

from itertools import tee
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def maximin(G):
    #this function returns the graph G with new maximin weights of each adjacent pair
    min_edges = []
    for u, v, d in G.edges(data = True):
        min_edges.clear()
        for path in k_shortest_paths(G,u,v,20):
            min_edges.append(min([G.get_edge_data(*edge)['weight'] for edge in pairwise(path)]))
        d['weight'] = max(min_edges)
    return G

def minmax(G):
    #this function returns the graph G with new weights of min(max(neighbors))
    for (u,v,d) in G.edges(data = True):
        maxu = max(G[u].values())
        maxv = max(G[v].values())
        d['weight'] = min(maxu,maxv)
    return G

def threshold(G,theta=0):
    #this function return a graph that is thresholded by theta,
    #deleting all edges that is below the threshold
    G.remove_edges_from([(u,v) for (u,v,d) in  G.edges(data=True) if d['weight']<=theta])
    return G

def segmentation_out(G,theta=0):
    nx.set_node_attributes(G, 0, 'intensity')
    components = nx.connected_components(G,theta)
    color_scale = len(list(components))
    attrs = {node: {'intensity' : index*255/color_scale} for index, comp in enumerate(components) for node in comp}
    print(attrs)
    nx.set_node_attributes(G, attrs)
    return G


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def minimum_energy(G,print_all_graphs = False):
    min_energy = 0
    M = G.copy()
    for cut_edges in powerset(G.edges):
        H = G.copy()
        H.remove_edges_from(cut_edges)
        components = [[x for x in comp] for comp in nx.connected_components(H)]
        energy = 0
        for edge in cut_edges:
            if not any(edge[0] in comp and edge[1] in comp for comp in components):
                energy+=G.edges[edge]['weight']
        
        if print_all_graphs:
            drawing(H)
            print('energy:', energy)
            
        if energy < min_energy:
            min_energy = energy
            M = H.copy()
    print('min energy: ', min_energy)
    drawing(M)

#----------------------------------

def image_to_graph(img):
    G = nx.grid_2d_graph(img.shape[0], img.shape[1])
    G.graph['X'] = img.shape[0]
    G.graph['Y'] = img.shape[1]
    nx.set_node_attributes(G,{u:{'intensity':v} for u,v in np.ndenumerate(img)})
    for u, v, d in G.edges(data = True):
        d['weight'] =  10 - abs(np.subtract(img[u], img[v]))
    return G

def graph_to_image(G):
    intensity = nx.get_node_attributes(G,'intensity')
    I = np.zeros((G.graph['X'], G.graph['Y']), dtype=np.int8)
    for node,value in intensity.items():
        I[node[0]][node[1]] = value
    return Image.fromarray(I,mode='L')


def random_graph(width, intRange):

    G = nx.grid_2d_graph(range(width), range(width))
    for u, v, d in G.edges(data = True):
        d['weight'] = np.random.randint(-intRange, intRange)
    
    return G