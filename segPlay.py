from __future__ import division, print_function
import pickle 
import numpy as np
import networkx as nx
import SegLib as seg



if __name__ == '__main__':

    rg = seg.random_graph(3, 5)
    drawing(rg)
    
