from search import *
import numpy as np

def solve(tnList, max_legs):
    edges = get_all_edge(tnList)
    scores = [priority(edge) for edge in edges]
    tmp = np.argsort(scores, kind='stable')[::-1]
    edges = list(edges[tmp])
    tn = GetTensorNetworkFromEdges(edges, tnList, max_legs)
    d,error,K = eval_TN(tn, 0.01, 0.05)
    return error

import random

def priority(edge: [int, int, int, int]) -> float:
    """Returns the priority with the new edge which we want to add to the tensor network."""
    return -np.abs(edge[0] - edge[2])

if __name__ == "__main__":
    tnList = ['code804','code603', 'codeH', 'codeS', 'code604', 'codeGHZ']
    result = solve(tnList, 8)
    print(result)