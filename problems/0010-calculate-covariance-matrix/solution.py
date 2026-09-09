def cov(n1, n2, n3, vectors):
	a1 = vectors[n1][n3] - (sum(vectors[n1]) / len(vectors[n1]))
	a2 = vectors[n2][n3] - (sum(vectors[n2]) / len(vectors[n2]))
	result = (a1 * a2) / (len(vectors[0]) - 1)
	return result

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	result = [[0] * len(vectors) for _ in range(len(vectors))]
	for n1 in range(len(vectors)):
		for n2 in range(len(vectors)):
			result[n1][n2] = sum(cov(n1, n2, n3, vectors) for n3 in range(len(vectors[0])))
			

	return result