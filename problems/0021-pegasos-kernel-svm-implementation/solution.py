import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    """
    n_samples = data.shape[0]
    
    # 1. 计算核矩阵 K (向量化加速)
    if kernel == 'linear':
        k_ij = np.dot(data, data.T)
    elif kernel == 'rbf':
        # 计算两两样本之间的欧式距离平方: ||x_i - x_j||^2
        sq_dists = np.sum(data**2, axis=1, keepdims=True) + np.sum(data**2, axis=1) - 2 * np.dot(data, data.T)
        k_ij = np.exp(-sq_dists / (2 * (sigma ** 2)))
    else:
        raise ValueError("Unsupported kernel type")

    a = np.zeros(n_samples)
    b = 0.0

    # 2. 主迭代循环
    for t in range(1, iterations + 1):
        n_t = 1.0 / (lambda_val * t)
        decay = 1.0 - n_t * lambda_val  # 实际上等于 (1 - 1/t)
        
        for i in range(n_samples):
            # 计算决策值 f(x_i) = sum_j (a_j * y_j * K_{j, i}) + b
            f_i = np.dot(a * labels, k_ij[:, i]) + b
            
            # 检查间隔约束
            if labels[i] * f_i < 1:
                a[i] = decay * a[i] + n_t
                b = b + n_t * labels[i]
            else:
                a[i] = decay * a[i]

    return a.tolist(), float(b)