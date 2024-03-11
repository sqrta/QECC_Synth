def priority(edge: [int, int, int, int]) -> float:
    score = 0.0

    # calculate even_sum at the beginning
    even_sum = sum([edge[i] for i in range(4) if edge[i] % 2 == 0])
    score += even_sum * 0.5  # adjust weight

    # update the original conditions and increase weights
    if ((edge[0] in [0, 1] and edge[2] in [0, 1]) or
        (edge[0] in [2, 3] and edge[2] in [2, 3]) or
        (edge[0] in [4, 5] and edge[2] in [4, 5])):
        if edge[1] * edge[3] == 0:
            score += 1.5  # increase weight
        elif edge[1] * edge[3] % 2 == 0:
            score += 1  # increase weight
        else:
            score += 0.8  # increase weight
    else:
        if edge[1] + edge[3] == 0:
            score += 1.0
        else:
            score += 0.5  # increase weight

    return score