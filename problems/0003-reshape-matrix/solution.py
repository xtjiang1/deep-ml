import numpy as np

def flater(a):
	flated_a = []
	for row in a:
		for items in row:
			flated_a.append(items)
	return flated_a


def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	
	row, column = len(a), len(a[-1])
	aim_row, aim_column = new_shape
	reshaped_matrix = [[0] * aim_column for _ in range(aim_row)]
	flat_list = flater(a)
	if aim_row * aim_column != row * column:
		return []
	else:
		idx = 0
		for i in range(aim_row):
			for j in range(aim_column):
				reshaped_matrix[i][j] = flat_list[idx]
				idx +=1
	return reshaped_matrix