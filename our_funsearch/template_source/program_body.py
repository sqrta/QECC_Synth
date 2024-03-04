from search import *
import numpy as np


@funsearch.run
def evaluate(tnList, max_legs) -> float:
  """Returns the size of an `n`-dimensional cap set."""
  error = solve(tnList, max_legs)
  return error


def solve(tnList, max_legs) -> float:
    tracted = set()
    tracted_leg = set()
    edges = get_all_edge(tnList)
    scores = [priority(edge) for edge in edges]
    tmp = np.argsort(scores, kind='stable')[::-1]
    edges = list(edges[tmp])
    
    new_edges = []
    insList = []
    
    def insert_edge(name, edge):
        insList.append([name] + list(edge))
        tracted.add(edge[0])
        tracted.add(edge[2])
        tracted_leg.add((edge[0], edge[1]))
        tracted_leg.add((edge[2], edge[3]))
    
    insert_edge('trace', edges.pop(0))
    index = 0
    while len(insList)<len(tnList)-1:
        edge = edges[index]
        t1 = edge[0]
        t2 = edge[2]
        if (t1, edge[1]) in tracted_leg or (t2, edge[3]) in tracted_leg:
            index += 1
            continue
        if (t1 in tracted and t2 in tracted) or (t1 not in tracted and t2 not in tracted):
            new_edges.append(edge)
        else:
            insert_edge('trace', edge)
        index += 1
    new_edges += edges[index:]
    for edge in new_edges:
        if len(insList) >= max_legs:
            break
        if (edge[0], edge[1]) not in tracted_leg and (edge[2], edge[3]) not in tracted_leg:
           insert_edge('self', edge)
    
    insList, tnList = normal_prog(insList, tnList)
    tn = GenProg2TNN(insList, tnList)
    logicalLeg = tn.equiv_trace_leg()[0]
    tn.setLogical(logicalLeg[0], logicalLeg[1])
    px = 0.01
    pz = 0.05
    _, error, _ = eval_TN(tn, px, pz)
    return error
