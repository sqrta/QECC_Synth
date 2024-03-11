def priority(edge: [int, int, int, int]) -> float:
    even_sum = sum([e for e in edge if e % 2 == 0])
    score = even_sum * 0.65
    
    pairs = [(0, 1), (2, 3), (4, 5)]
    bonuses = {0: 0.8, 1: 0.6, 2: 0.4}
    reductions = {0: 0.7, 1: 0.5, 2: 0.3}
    for i, pair in enumerate(pairs):
        if edge[0] in pair and edge[2] in pair:
            score += bonuses[i] if edge[1] * edge[3] == 0 else reductions[i]
            
    if edge[1] == edge[3]:
        score += 0.5
        
    return min(score, sum(edge) * 0.1)