import numpy as np

def divide_on_feature(X, feature_i, threshold):
	# Your code here
	deleted_index = []
	X = X.tolist()
	for i, row in enumerate(X):
		if row[feature_i] < threshold:
			deleted_index.append(i)

	deleted = []
	for idx in sorted(deleted_index, reverse=True):
		deleted.append(X.pop(idx))
	
	deleted = sorted(deleted, key= lambda x: x[0])

	return [np.array(X), np.array(deleted)]