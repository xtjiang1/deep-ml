import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	C = np.array(C); B = np.array(B)

	C = np.linalg.inv(C)
	P = C @ B
	return P.tolist()