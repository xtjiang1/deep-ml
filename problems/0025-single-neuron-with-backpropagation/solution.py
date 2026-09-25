import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	mse = []
	for epoch in range(epochs):
		sigma = []
		for i, row in enumerate(features):
			z = (row @ initial_weights) + initial_bias
			sigma.append(1 / (1 + np.exp(-z)))
		sigma = np.array(sigma)
		mse.append((((sigma - labels) ** 2).sum()) / sigma.shape[0])
		delta = (sigma - labels) * (sigma * (1 - sigma))
		partical_w = 2/delta.shape[0] * (delta @ features)
		partical_b = 2/delta.shape[0] * sum(delta)

		initial_weights = initial_weights - (learning_rate * partical_w)
		initial_bias = initial_bias - (learning_rate * partical_b)
		
	return initial_weights.round(4).tolist(), initial_bias.round(4).tolist(), [round(x, 4) for x in mse]
