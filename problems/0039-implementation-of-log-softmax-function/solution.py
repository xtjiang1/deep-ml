import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	result = []
	log = np.log(sum(np.exp(np.array(scores) - max(scores))))
	for x_i in scores:
		result.append(x_i - max(scores) - log)
	return result