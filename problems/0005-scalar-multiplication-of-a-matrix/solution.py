def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	result = [[0] * len(matrix[0]) for _ in range(len(matrix))]
	for i, row in enumerate(matrix):
		for j, column in enumerate(row):
			result[i][j] = matrix[i][j] * scalar

	return result