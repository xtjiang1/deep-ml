def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	result = []
	if len(a[0]) != len(b):
		return -1
	else:
		for row in a:
			total = 0
			for i in range(len(row)):
				total += row[i] * b[i]
			result.append(total)
		return result

