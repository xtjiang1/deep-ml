def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	
	if mode == 'column':
		means = []
		for i, column in enumerate(zip(*matrix)):
			means.append(sum(column)/len(column))
	elif mode == 'row':
		means = []
		for i, row in enumerate(matrix):
			means.append(sum(row)/len(row))
	else:
		return []

	return means