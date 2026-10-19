import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    # Your code here
    subset_size = X.shape[0] if replacements else round(X.shape[0] / 2)

    result = []
    for i in range(n_subsets):
        index = np.random.choice(X.shape[0], size=subset_size, replace = replacements)
        X_subset = []
        y_subset = []
        for idx in index:
            X_subset.append(X[idx].tolist())
            y_subset.append(y[idx].tolist())
        result.append(tuple([X_subset, y_subset]))


    return result