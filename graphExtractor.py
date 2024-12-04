##
# @file graphExtractor.py
# @author Keren Zhu
# @date 11/16/2019
# @brief The functions and classes for processing the graph
#

import abc_py as abcPy
import numpy as np
from numpy import linalg as LA
import numpy as np
import dgl
import torch

def symmetricLaplacian(abc):
    numNodes = abc.numNodes()
    L = np.zeros((numNodes, numNodes))
    print("numNodes", numNodes)
    for nodeIdx in range(numNodes):
        aigNode = abc.aigNode(nodeIdx)
        degree = float(aigNode.numFanouts())
        if (aigNode.hasFanin0()):
            degree += 1.0
            fanin = aigNode.fanin0()
            L[nodeIdx][fanin] = -1.0
            L[fanin][nodeIdx] = -1.0
        if (aigNode.hasFanin1()):
            degree += 1.0
            fanin = aigNode.fanin1()
            L[nodeIdx][fanin] = -1.0
            L[fanin][nodeIdx] = -1.0
        L[nodeIdx][nodeIdx] = degree
    return L

def symmetricLapalacianEigenValues(abc):
    L = symmetricLaplacian(abc)
    print("L", L)
    eigVals = np.real(LA.eigvals(L))
    print("eigVals", eigVals)
    return eigVals

"""
def extract_dgl_graph(abc):
    numNodes = abc.numNodes()

    # Create a graph with a fixed number of nodes
    G = dgl.graph(([], []), num_nodes = numNodes)

    # Prepare edge lists and feature tensors
    edge_src = []
    edge_dst = []
    features = torch.zeros(numNodes, 6)

    for nodeIdx in range(numNodes):
        aigNode = abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()

        features[nodeIdx][nodeType] = 1.0

        if aigNode.hasFanin0():
            edge_src.append(aigNode.fanin0())
            edge_dst.append(nodeIdx)

        if aigNode.hasFanin1():
            edge_src.append(aigNode.fanin1())
            edge_dst.append(nodeIdx)

    # Batch-add edges
    G.add_edges(edge_src, edge_dst)

    # Assign features to graph nodes
    G.ndata['feat'] = features

    return G
"""

def extract_dgl_graph(abc, upperbound):
    numNodes = abc.numNodes()

    # Create a graph with a fixed number of nodes
    G = dgl.graph(([], []), num_nodes=upperbound)

    # Prepare edge lists and feature tensors
    edge_src = []
    edge_dst = []
    features = torch.zeros(upperbound, 7)

    list_fin = []  # Collect indices of nodes of type 2
    for nodeIdx in range(numNodes):
        aigNode = abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()

        if nodeType == 2:
            list_fin.append(nodeIdx)

        features[nodeIdx][nodeType] = 1.0

        if aigNode.hasFanin0():
            edge_src.append(aigNode.fanin0())
            edge_dst.append(nodeIdx)

        if aigNode.hasFanin1():
            edge_src.append(aigNode.fanin1())
            edge_dst.append(nodeIdx)

    # Add additional features for nodes between numNodes and upperbound
    for nodeIdx in range(numNodes, upperbound):
        features[nodeIdx][6] = 1.0
        edge_src.extend([nodeIdx] * len(list_fin))
        edge_dst.extend(list_fin)

    # Batch-add edges
    G.add_edges(edge_src, edge_dst)

    # Assign features to graph nodes
    G.ndata['feat'] = features

    return G
