def priority(edge: [int, int, int, int]) -> float:
    score_v0 = 0.0
    even_sum = sum([edge[i] for i in range(4) if edge[i] % 2 == 0])
    score_v0 += even_sum * 0.75
    
    if edge[1] == 0 and edge[3] == 0:
        return sum(edge) * 0.1

    if edge[0] == edge[2] and edge[1] * edge[3] == 0:
        score_v0 += 0.9

    if edge[0] + edge[2] == 0:
        score_v0 += 1.0

    if edge[1] + edge[3] == 0:
        score_v0 += 1.0

    prec_edge_conditions = [(0, 1, 0.9, 0.8), (2, 3, 0.7, 0.6), (4, 5, 0.5, 0.4)]
    for (c1, c2, s1, s2) in prec_edge_conditions:
        if (edge[0] == c1 and edge[2] == c2) or (edge[0] == c2 and edge[2] == c1):
            if edge[1] == 0 or edge[3] == 0:
                score_v0 += s1
            else:
                score_v0 += s2

    score_v1 = 0.0
    even_sum = sum([edge[i] for i in range(4) if edge[i] % 2 == 0])
    score_v1 += even_sum * 0.75

    if ((edge[0] in [0, 1] and edge[2] in [0, 1]) or
        (edge[0] in [2, 3] and edge[2] in [2, 3]) or
        (edge[0] in [4, 5] and edge[2] in [4, 5])):
        if edge[1] * edge[3] == 0:
            score_v1 += 1.0
        elif edge[1] * edge[3] % 2 == 0:
            score_v1 += 0.9
        else:
            score_v1 += 0.7
    else:
        if edge[1] + edge[3] == 0:
            score_v1 += 1.0
        else:
            score_v1 += 0.1

    return min(score_v0, score_v1)