import math

def softmax(input: list[float]) -> list[float]:
    result = []
    bottom = sum((math.exp(z_j- max(input))) for z_j in input)
    for z_i in input:
        result.append((math.exp(z_i - max(input))) / bottom)
    return result