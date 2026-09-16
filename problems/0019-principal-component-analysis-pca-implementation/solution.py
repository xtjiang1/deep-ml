import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """


    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardize = (data - mean) / std
    C = (1 / (data.shape[0] - 1)) * (standardize.T @ standardize)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    idx = np.argsort(eigenvalues)[::-1][:k]
    result = eigenvectors[:, idx]

    for j in range(result.shape[1]):
        col = result[:, j]
        nz = np.flatnonzero(np.abs(col) > 1e-10)
        if nz.size > 0 and col[nz[0]] < 0:
            result[:, j] = -col

    return np.round(result, 4)
