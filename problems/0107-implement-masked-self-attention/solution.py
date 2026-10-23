import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def softmax(x, axis = -1):
	x = x - np.max(x, axis=axis, keepdims=True)
	e = np.exp(x)
	return e / np.sum(e, axis=axis, keepdims=True)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	# Your code here
	s = (Q @ K.swapaxes(-1, -2)) / (K.shape[-1] ** 0.5)
	# mask = np.triu(np.ones_like(s), k=1).astype(bool)
	# s[mask] = -np.inf
	s = s + mask
	attention = softmax(s) @ V
	return attention