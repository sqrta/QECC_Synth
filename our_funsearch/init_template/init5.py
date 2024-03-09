def priority(edge: [int, int, int, int]) -> float:
    """Returns the priority with the new edge which we want to add to the tensor network."""
    score = 0

    if (edge[0] == 0 and edge[2] == 1) or (edge[0] == 1 and edge[2] == 0):
        if edge[1] == 0 or edge[3] == 0:
            score += 0.9
        else:
            score += 0.8
    elif (edge[0] == 2 and edge[2] == 3) or (edge[0] == 3 and edge[2] == 2):
        if edge[1] == 0 or edge[3] == 0:
            score += 0.7
        else:
            score += 0.6
    elif (edge[0] == 4 and edge[2] == 5) or (edge[0] == 5 and edge[2] == 4):
        if edge[1] == 0 or edge[3] == 0:
            score += 0.5
        else:
            score += 0.4

    if edge[0] % 2 == 0:
        score -= edge[0] * 0.1
    else:
        score += edge[0] * 0.2

    if edge[1] < 5:
        score -= edge[1] * 0.2
    else:
        score += edge[1] * 0.1

    if edge[2] >= 0:
        score -= edge[2] * 0.1
    else:
        score += edge[2] * 0.2

    if edge[3] > 1:
        score -= edge[3] * 0.1
    else:
        score += edge[3] * 0.2

    return score