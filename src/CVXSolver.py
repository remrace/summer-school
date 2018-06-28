import pickle 
import numpy as np
import networkx as nx
import SegGraph as seg
import VizGraph as viz
import matplotlib.pyplot as plt
#import cvxopt as cvx
#from cvxopt import matrix

def Minimize(G):

    n = len(G)
    nv = n*n

    Q = np.zeros((nv, nv))

    #P = cvx.matrix(numpy.diag([1,0]), tc=’d’)
    #q = cvx.matrix(numpy.array([3,4]), tc=’d’)
    #G = cvx.matrix(numpy.array([[-1,0],[0,-1],[-1,-3],[2,5],[3,4]]), tc=’d’)
    #h = cvx.matrix(numpy.array([0,0,-15,100,80]), tc=’d’)
    
    
    #Q = [ [0] * nv for i in range(nv)] 
    
    #expr = quicksum(Q[i,j]*x[i]*x[j] for i in range(nv) for j in range(nv))
    
    #cons = expr <= 1.0
    #                              upper triangle,     diagonal
    #assert len(cons.expr.terms) == dim * (dim-1) / 2 + dim
    #m.addCons(cons)


    #x =  [None] * nv
    #for i in range(nv):
    #    x[i] = s.addVar(vtype="BINARY", name = "x(%s)" % (i))

    #m = Model("dense_quadratic")
    #dim = 200


    #for n in G:
        
    #x = s.addVar("x")
    #y = s.addVar("y", vtype="INTEGER")
    #s.setObjective(x + y)
    #s.addCons(2*x - y*y >= 0)

    #s.setMaximize()
    #s.optimize()

    #s.freeProb()

    E = 0.0
    L = dict()

    return E, L



