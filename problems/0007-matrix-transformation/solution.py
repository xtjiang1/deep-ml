import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	t_numpy = np.array(T)
	s_numpy = np.array(S)
	a_numpy = np.array(A)
	transformed_matrix = np.zeros_like(a_numpy)

	det_t = np.linalg.det(t_numpy)
	det_s = np.linalg.det(s_numpy)

	if det_s == 0 or det_t == 0:
		return -1
	else:
		transformed_matrix = np.linalg.inv(t_numpy) @ a_numpy @ s_numpy

	
	return transformed_matrix