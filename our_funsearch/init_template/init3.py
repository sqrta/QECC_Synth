def priority(edge: [int, int, int, int]) -> float:
    """Returns the priority with the new edge which we want to add to the tensor network."""
    score = 0
    score += 0.98*edge[0] + 1.25*edge[1] - edge[2]/2 - edge[3]**0.5
    return score