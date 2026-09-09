def matrixmul(a:list[list[int|float]], b:list[list[int|float]])-> list[list[int|float]]:
	if len(a[0]) != len(b):
		return -1
	else:
		c = [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
		return c