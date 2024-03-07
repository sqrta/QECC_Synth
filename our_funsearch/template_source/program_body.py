from search import *
import numpy as np

def evaluate(tnList, max_legs):
    edges = get_all_edge(tnList)
    scores = [priority(edge) for edge in edges]
    tmp = np.argsort(scores, kind='stable')[::-1]
    edges = list(edges[tmp])
    tn = GetTensorNetworkFromEdges(edges, tnList, max_legs)
    _, error, _ = eval_TN(tn, 0.01, 0.05)
    return error
