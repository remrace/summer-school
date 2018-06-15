
# coding: utf-8

# In[1]:


#import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#define size of the grid
SIZE_X = 3
SIZE_Y = 3


# In[3]:


#define a NetworkX 4-adj connected graph
#define 'weight' of each edge to be randome integer between 1-9
G = nx.grid_2d_graph(range(SIZE_X), range(SIZE_Y))
for u, v, d in G.edges(data = True):
    d['weight'] = np.random.randint(1, 10)


# In[4]:


#this whole cell is to draw the graph G nicely
#pos specifies the location of each vertex to be drawn
pos = {x: x for x in list(G)}

#this is to make the drawing region bigger
plt.figure(figsize=(12,12)) 

#drawing nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

#drawing labels (locations) of the nodes
nx.draw_networkx_labels(G, pos)

#drawing edges and theirs labels ('weight')
nx.draw_networkx_edges(G, pos)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, labels)

#show the drawing
plt.axis('off')
plt.show()


# In[40]:


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
        for path in nx.all_simple_paths(G, source=u, target=v):
            min_edges.append(min([G[path[i]][path[i+1]]['weight'] for i, node in enumerate(path) if i < (len(path)-1)]))
        d['weight'] = max(min_edges)
    return G

def threshold(G,theta=0):
    #this function return a graph that is thresholded by theta,
    #deleting all edges that is below the threshold
    G.remove_edges_from([(u,v,d) for (u,v,d) in  G.edges(data=True) if d['weight']<theta])
    return G

def segmentation_out(G,theta=0):
    nx.set_node_attributes(G, 0, 'intensity')
    attrs = {node: {'intensity' : index} for index, comp in enumerate(nx.connected_components(threshold(maximin(G),theta)))
            for node in comp}
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
    return Image.fromarray(I,mode='P')


# In[38]:


#drawing graph S
S = segmentation_out(G,7)
plt.figure(figsize=(12,12)) 
nx.draw_networkx_nodes(S, pos, node_size=700)
nx.draw_networkx_labels(S, pos)
nx.draw_networkx_edges(S, pos)
labels = nx.get_edge_attributes(S, 'weight')
nx.draw_networkx_edge_labels(S, pos, labels)
plt.axis('off')
plt.show()


# In[42]:


from PIL import Image
im = Image.open('image1.jpg').convert('L')
img = np.array(im)


# In[ ]:


G = image_to_graph(img)
#this whole cell is to draw the graph G nicely
#pos specifies the location of each vertex to be drawn
pos = {x: x for x in list(G)}

#this is to make the drawing region bigger
plt.figure(figsize=(12,12)) 

#drawing nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

#drawing labels (locations) of the nodes
nx.draw_networkx_labels(G, pos)

#drawing edges and theirs labels ('weight')
nx.draw_networkx_edges(G, pos)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, labels)

#show the drawing
plt.axis('off')
plt.show()

