import torch
import math
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,Q,K,V,mask=None):
        d_k=Q.size(-1)
        scores=torch.matmul(Q,K.transpose(-2,-1))
        scores=scores/math.sqrt(d_k)
        if mask is not None:
            scores=scores.masked_fill(mask==False,torch.finfo(scores.dtype).min)
        weights=torch.softmax(scores,dim=-1)
        output=torch.matmul(weights,V)
        return output,weights

if __name__ == "__main__":
    torch.manual_seed(0)
    attn = ScaledDotProductAttention()

    Q = torch.randn(2, 4, 3, 64)
    K = torch.randn(2, 4, 5, 64)
    V = torch.randn(2, 4, 5, 64)

    # Test 1: no mask
    out, w = attn(Q, K, V)
    print("output shape :", out.shape)      # expect (2, 4, 3, 64)
    print("weights shape:", w.shape)        # expect (2, 4, 3, 5)
    print("row sums     :", w.sum(-1)[0, 0])  # expect all 1.0

    # Test 2: hide the last 2 keys (True = keep, False = hide)
    mask = torch.ones(2, 1, 1, 5, dtype=torch.bool)
    mask[..., 3:] = False
    out, w = attn(Q, K, V, mask)
    print("masked weights:\n", w[0, 0])     # last 2 columns must be 0
    print("row sums     :", w.sum(-1)[0, 0])  # still all 1.0