import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def softmax(x : torch.Tensor, dim = -1) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e / torch.sum(e, dim=dim, keepdim=True)

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    attention = softmax(Q @ K.swapaxes(-1, -2) / (K.shape[-1] ** 0.5)) @ V
    return attention
