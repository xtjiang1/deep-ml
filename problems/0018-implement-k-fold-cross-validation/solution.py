import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """

    data = np.arange(n_samples)
    k_quotient = n_samples // k
    k_remainder = n_samples % k
    k_arry = np.zeros(k , dtype = int)
    for i in range(k):
        if k_remainder != 0: 
            k_arry[i] = k_quotient + 1
            k_remainder -= 1
        else:
            k_arry[i] = k_quotient
    if shuffle: np.random.shuffle(data)

    cusum_k_arry = np.cumsum(k_arry, axis=0)[:-1]
    test_idx = np.split(data, cusum_k_arry)
    test_idx = [x.tolist() for x in test_idx]
    
    result = []
    for i in range(k):
        test = test_idx[i]
        train = np.concatenate([test_idx[j] for j in range(k) if j != i]).tolist()
        result.append(tuple([train, test]))
    

        
    return result