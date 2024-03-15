from search import *
import numpy as np

def evaluate(tnList, max_legs):
    edges = get_all_edge(tnList)
    scores = [priority(edge) for edge in edges]
    tmp = np.argsort(scores, kind='stable')[::-1]
    edges = list(edges[tmp])
    tn = GetTensorNetworkFromEdges(edges, tnList, max_legs)
    d,error,K = eval_TN(tn, 0.01, 0.05)
    print(d, error)
    return error

import random

# gen_round1521_n2
def priority1521_n2(edge) -> float:
    if edge[1] == 0 and edge[3] == 0:
        score = 1.0
    elif edge[0] == 0 and edge[1] == 0 and edge[2] == 1:
        score = 0.9
    elif edge[0] == 1 and edge[1] == 0 and edge[2] == 0:
        score = 0.9
    elif edge[0] % 2 == 0 and edge[2] % 2 == 1:
        if edge[1] * edge[3] == 0:
            score = 1.0
        else:
            score = 0.9
    elif edge[0] % 2 == 0 and edge[2] % 2 == 0:
        if edge[1] * edge[3] == 0:
            score = 0.9
        else:
            score = 0.6
    else:
        score = 0.98 * edge[0] + 1.25 * edge[1] - (edge[2] + edge[3]) / 2 - edge[3] ** 0.5

    return score

def priority(edge) -> float:
    score = 0.0

    if edge[0] % 2 == 0 and edge[2] % 2 == 1:
        if edge[1] * edge[3] == 0:
            score = min(score, 1.0)
        elif edge[1] * edge[3] % 2 == 0:
            score = min(score, 0.9)
        else:
            score = min(score, 0.6)
    elif edge[0] % 2 == 1 and edge[2] % 2 == 0:
        if edge[1] * edge[3] == 0:
            score = min(score, 0.4)
        elif edge[1] * edge[3] % 2 == 0:
            score = min(score, 0.1)
        else:
            score = min(score, 0.2)
    elif edge[0] % 2 == 0 and edge[2] % 2 == 0:
        if (edge[1] * edge[3]) == 0:
            score += 0.5
        else:
            score += 0.3
    else:
        even_sum = sum([edge[i] for i in range(4) if edge[i] % 2 == 0])
        score += even_sum * 0.001

    return score

def priority_8760_n1(edge) -> float:
    score = 0
    if (edge[0] in [0, 1] and edge[2] in [0, 1]) or \
            ((edge[0] == 2 and edge[2] == 3) or (edge[0] == 3 and edge[2] == 2)) or \
            ((edge[0] == 4 and edge[2] == 5) or (edge[0] == 5 and edge[2] == 4)):
        score = -1.5 if edge[1] * edge[3] == 0 else -1.3
    else:
        score = edge[0] * 0.01 + edge[1] * 0.02 - edge[2] * 0.002 - edge[3] ** 0.5
    
    if edge[1] - edge[3] == 0:
        score += 1.2
    elif edge[1] * edge[3] == 0:
        score += 0.6
    elif edge[0] in [0, 1]:
        score += 0.2
    elif edge[0] in [2, 3]:
        score += 0.4
    else:
        score += 0.6
    
    return score




if __name__ == "__main__":
    tnList = ['code804','code603', 'codeH', 'codeS', 'code604', 'codeGHZ']
    result = evaluate(tnList, 8)
    print(result)