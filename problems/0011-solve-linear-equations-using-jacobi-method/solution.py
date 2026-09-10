import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	x_old = [0] * len(b)
	x_new = [0] * len(b)
	a_ii = np.diag(A)
	for _ in range(n):
		for i, (a_i, b_i) in enumerate(zip(A, b)):
			x_new[i] = (1 / a_ii[i]) * (b_i - sum([(a_i[j] * x_old[j]) for j, a_ij in enumerate(a_i) if i != j]))
		x_old = x_new[:]
	return x_old