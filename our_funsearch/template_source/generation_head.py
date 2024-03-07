def priority(edge: [int, int, int, int]) -> float:
  """Returns the priority with the new edge which we want to add to the tensor network."""
  score = 0.0
  match edge[0],edge[2]:
    case 0,1:
      score += 0.5
    case 2,3:
      score += 0.4
    case 4,5:
      score += 0.3
  return score
