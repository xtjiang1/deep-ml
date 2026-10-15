import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	n_samples = X.shape[0]
	for start in range(0, n_samples, batch_size):
		X_batch = X[start : start + batch_size]
		if y is not None:
			y_batch = y[start : start + batch_size]
			yield [X_batch.tolist(), y_batch.tolist()]
		else:
			yield X_batch.tolist()