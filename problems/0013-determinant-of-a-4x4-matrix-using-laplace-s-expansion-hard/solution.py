def remove_row_col(matrix, row_idx, col_idx):
	new_matrix = []
	for i, row in enumerate(matrix):
		if i == row_idx:
			continue
		new_row = []
		for j, val in enumerate(row):
			if j == col_idx:
				continue
			new_row.append(val)
		new_matrix.append(new_row)
	return new_matrix
			

def cal_det(matrix: list[list[int|float]]):
	c_ij_temp = [(-1) ** (j) for j in range(len([0] * len(matrix[0])))]
	new_matrix = [[0] * (len(matrix[0]) - 1) for _ in range(len(matrix) - 1)]
	if len(matrix[0]) == 3:
		det = 0
		for j in range(len(matrix[0])):
			new_matrix = remove_row_col(matrix, 0, j)
			det += matrix[0][j] * c_ij_temp[j] * (new_matrix[0][0]*new_matrix[1][1]-new_matrix[0][1]*new_matrix[1][0])
		return det
	if len(matrix[0]) == 4:
		det = 0
		for j in range(len(matrix[0])):
			new_matrix = remove_row_col(matrix, 0, j)
			det += matrix[0][j] * c_ij_temp[j] * cal_det(new_matrix)
		return det
	

def determinant_4x4(matrix: list[list[int|float]]) -> float:
	return cal_det(matrix)