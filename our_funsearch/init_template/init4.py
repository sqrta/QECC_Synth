def priority(edge: [int, int, int, int]) -> float:
    """Returns the priority with the new edge which we want to add to the tensor network."""
    score = 0

    if edge[0] % 2 == 0:
        score += edge[0] * 0.98
    else:
        score += edge[0] * 1.25

    if edge[1] < 5:
        score += edge[1] * 1.25
    else:
        score += edge[1]

    if edge[2] >= 0:
        score -= edge[2] / 2
    else:
        score -= edge[2]

    if edge[3] > 1:
        score -= edge[3] * 0.5
    else:
        score -= edge[3]

    return score