import numpy as np

def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Apply RMSNorm to the input array.
    
    Parameters:
        x   : np.ndarray of shape (batch_size, features)
        g   : np.ndarray of shape (features,) - gain parameter
        eps : float - small constant for numerical stability
    
    Returns:
        np.ndarray of same shape as x
    """
    results = []
    for row in x:
        result = []
        for i, x_i in enumerate(row):
            result.append(x_i / np.sqrt(np.mean(np.square(row))+ eps) * g[i])
        results.append(result)
    return results