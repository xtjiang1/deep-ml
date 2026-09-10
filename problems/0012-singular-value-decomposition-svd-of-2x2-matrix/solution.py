import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    B = A.T @ A
    theta = 0.5 * np.arctan2(2 * B[0, 1], B[0, 0] - B[1, 1])
    V = R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    D = R.T @ B @ R

    sigma = np.array([
        [max(np.sqrt(D[1, 1]), np.sqrt(D[0, 0])), 0],
        [0, min(np.sqrt(D[1, 1]), np.sqrt(D[0, 0]))]
    ])

    U = A @ V @ np.linalg.inv(sigma)
    return U, np.array([sigma[0, 0], sigma[1, 1]]), np.linalg.inv(V)
