def priority(edge: [int, int, int, int]) -> float:
    score = 0.0
    if ((edge[0] % 2 == edge[2] % 2) and (edge[0] in [1, 3]) and (edge[1] + edge[3] == 0)):
        score = 0.0
    elif ((edge[0] % 2 == edge[2] % 2) and (edge[0] in [0, 2, 4, 5]) and (edge[1] + edge[3] != 0)):
        score = 0.1
    else:
        if (edge[0] + edge[2]) % 3 == 0:
            if edge[1] * edge[3] == 0:
                score += 0.7
            else:
                score += 0.6
        elif (edge[0] + edge[2]) % 3 == 1:
            if edge[1] * edge[3] == 0:
                score += 0.5
            else:
                score += 0.4
        elif (edge[0] + edge[2]) % 3 == 2:
            if edge[1] * edge[3] == 0:
                score += 0.9
            else:
                score += 0.8
    return score