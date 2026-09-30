import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__ (self,d_model=256,d_ff=1024):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(d_ff,d_model)
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(0)
    ffn = PositionwiseFeedForward(d_model=256, d_ff=1024)

    x = torch.randn(2, 5, 256)
    out = ffn(x)
    print("output shape:", out.shape)     # (2, 5, 256)

    hidden = ffn.relu(ffn.linear1(x))
    print("hidden shape:", hidden.shape)  # (2, 5, 1024)
    print("min after ReLU:", hidden.min().item())  # 0.0, no negatives

    # position-wise: each position processed independently
    print("position-wise:", torch.allclose(ffn(x[:, 2:3]), out[:, 2:3], atol=1e-6))  # True

    print("FFN parameters:", sum(p.numel() for p in ffn.parameters()))  # 525568