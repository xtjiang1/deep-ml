def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    det_val = a * d - b * c
    if det_val == 0:
        return None
    else:
        inv_det = 1 / det_val
        result = [[d*inv_det, -b*inv_det], [-c*inv_det, a*inv_det]]
        return result
                    
