import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _softmax(x, dim=-1):
        x = x - torch.max(x, dim=dim)
        e = torch.exp(x)
        return e / torch.sum(e, dim=dim)

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        # d_model = 8; num_heads = 2; d_head = 4;
        B, T, d_model = x.shape
        H, d_head = self.num_heads, self.d_head
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # X(B, T, d_model) -> X(B, T, H, d_head) -> X(B, H, T, d_head)
        Q = Q.view(B, T, H, d_head).swapaxes(1, 2)
        K = K.view(B, T, H, d_head).swapaxes(1, 2)
        V = V.view(B, T, H, d_head).swapaxes(1, 2)

        S = Q @ K.swapaxes(-1, -2) / Q.shape[-1] ** 0.5

        if mask is not None:
            S = S + mask

        attention = F.softmax(S, dim=-1) @ V

        # attention(B, H, T, d_head) -> attention(B, T, H, d_head) -> attention(B, T, d_model)
        attention = attention.swapaxes(1, 2).contiguous().view(B, T, d_model)


        return self.out_proj(attention)
