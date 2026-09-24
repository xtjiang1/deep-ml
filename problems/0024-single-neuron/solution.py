import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	E = []
	p = []
	for i, row in enumerate(features):
		z = sum([column * weights[i] for i, column in enumerate(row)]) + bias
		sigma = 1 / (1 + math.exp(-z))
		p.append(round(sigma, 4))
		se = (sigma - labels[i]) ** 2
		E.append(se)
		
	MSE = np.mean(E)
	
	return p, round(MSE, 4)