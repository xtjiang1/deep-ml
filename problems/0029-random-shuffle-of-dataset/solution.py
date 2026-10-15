import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
	# rng = np.random.default_rng(seed=seed)
	# shuffled_indices = rng.permutation(len(X))

	# X_shuffled = X[shuffled_indices]
	# y_shuffled = y[shuffled_indices]

	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)
	return X, y