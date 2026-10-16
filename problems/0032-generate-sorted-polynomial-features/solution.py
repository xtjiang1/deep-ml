import numpy as np
import math
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    # âï¸  Your code here
    outputs = []
    for i, row in enumerate(X):
        output = []
        for j in range(degree):
            idx = j + 1
            output.extend(list(combinations_with_replacement(row, idx)))
        output.append((1, ))
        output = sorted([np.prod(x) for x in output])
        outputs.append(output)

    return np.array(outputs, dtype = float)