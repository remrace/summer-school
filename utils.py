#import necessary libraries
import networkx as nx
import numpy as np
from PIL import Image

#some function definitions
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

def maximin(G):
    #this function returns the graph G with new maximin weights of each adjacent pair
    for u, v, d in G.edges(data = True):
        min_edges = []
        for path in nx.all_simple_paths(G, source=u, target=v, cutoff = 5):
            min_edges.append(min([z['weight'] for x,y,z in G.edges(path,data=True)]))
        d['weight'] = max(min_edges)
    return G

def threshold(G,theta=0):
    #this function return a graph that is thresholded by theta,
    #deleting all edges that is below the threshold
    G.copy().remove_edges_from([(u,v,d) for (u,v,d) in  G.edges(data=True) if d['weight']<theta])
    return G

def segmentation_out(G,theta=0):
    nx.set_node_attributes(G, 0, 'intensity')
    components = nx.connected_components(G)
    color_scale = len(list(components))
    print(color_scale)
    attrs = {node: {'intensity' : index*255/color_scale} for index, comp in enumerate(components) for node in comp}
    print(attrs)
    nx.set_node_attributes(G, attrs)
    return G

#----------------------------------

def image_to_graph(img):
    G = nx.grid_2d_graph(img.shape[0], img.shape[1])
    G.graph['X'] = img.shape[0]
    G.graph['Y'] = img.shape[1]
    nx.set_node_attributes(G,{u:{'intensity':v} for u,v in np.ndenumerate(img)})
    for u, v, d in G.edges(data = True):
        d['weight'] = abs(np.subtract(img[u], img[v]))
    return G

def graph_to_image(G):
    intensity = nx.get_node_attributes(G,'intensity')
    I = np.zeros((G.graph['X'], G.graph['Y']), dtype=np.int8)
    for node,value in intensity.items():
        I[node[0]][node[1]] = value
    return Image.fromarray(I,mode='L')