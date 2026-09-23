import math

def softmax(scores: list[float]) -> list[float]:
    result = []
    for z_i in scores:
        result.append((math.exp(z_i - max(scores))) / (sum([math.exp(scores[j] - max(scores)) for j in range(len(scores))])))
    return result