def tr(matrix: list[list[float|int]]) -> float:
	tr = 0
	for i, row in enumerate(matrix):
		for j, column in enumerate(matrix):
			if i == j:
				tr += matrix[i][j]
	return tr

def det(matrix: list[list[float|int]]) -> float:
	det2 = det1 = 1
	for i, row in enumerate(matrix):
		for j, column in enumerate(matrix):
			if i == j:
				det1 = det1 * matrix[i][j]
			else:
				det2 = det2 * matrix[i][j]
	return det1 - det2

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	eigenvalues = [0, 0]
	Acoef, Bcoef, Ccoef = 1, -1 * tr(matrix), det(matrix)
	delta = complex(Bcoef**2 - 4 * Acoef * Ccoef, 0)
	sqrt_delta = delta ** 0.5
	eigenvalues[0] = ((-Bcoef + sqrt_delta) / (2*Acoef)).real
	eigenvalues[1] = ((-Bcoef - sqrt_delta) / (2*Acoef)).real
	

			
	return eigenvalues