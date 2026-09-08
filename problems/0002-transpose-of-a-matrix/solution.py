def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    result = [[0] * len(a) for _ in range(len(a[0]))]
    for i, row in enumerate(a):
        for j in range(len(row)):
            result[j][i] = a[i][j]

    return result