import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
    
    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """
    # get Kernel Matrix
    k_ij = np.zeros([data.shape[0], data.shape[0]])
    for i, row in enumerate(k_ij):
        for j, column in enumerate(row):
            k_ij[i, j] = np.dot(data[i], data[j]) if kernel == 'linear' else np.exp(-(np.linalg.norm(data[i]-data[j]) ** 2) / (2 * sigma ** 2))
    a = np.zeros(data.shape[0])
    b = 0

    for iter in range(iterations):
        n_t = 1 / (lambda_val * (iter + 1))
        for i, row in enumerate(data):
            f = sum(a[j] * labels[j] * k_ij[j, i] for j in range(data.shape[0])) + b
            if labels[i] * f < 1:
                a[i] = (1 - n_t * lambda_val) * a[i] + n_t
                b = b + n_t * labels[i]
            else:
                a[i] = (1 - n_t * lambda_val) * a[i]
    return (a, b)