def priority(edge: [int, int, int, int]) -> float:
    """Returns the priority with the new edge which we want to add to the tensor network."""
    score = 0
    score += sum(edge)
    return score
