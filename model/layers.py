import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

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

class EncoderLayer(nn.Module):
    def __init__ (self,d_model=256,num_heads=4,d_ff=1024,dropout=0.1):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model,num_heads)
        self.ffn=PositionwiseFeedForward(d_model,d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)

    def forward(self,x,mask):
         output,_=self.self_attn(x,x,x,mask)
         x=self.norm1(x+self.dropout1(output))
         f=self.ffn(x)
         x=self.norm2(x+self.dropout2(f))
         return x

class Encoder(nn.Module):
     def __init__ (self,num_layers,d_model=256,num_heads=4,d_ff=1024,dropout=0.1):
        super().__init__()
        self.layers=nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout)for _ in range(num_layers)])
     def forward(self,x,mask):
        for  layer in self.layers:
               x=layer(x,mask)
        return x

class DecoderLayer(nn.Module):
    def __init__ (self,d_model=256,num_heads=4,d_ff=1024,dropout=0.1):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model,num_heads)
        self.cross_attn=MultiHeadAttention(d_model,num_heads)
        self.ffn=PositionwiseFeedForward(d_model,d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
        self.dropout3=nn.Dropout(dropout)

    def forward(self,x,enc_out,src_mask,tgt_mask):
        # 1. masked self-attention (tgt_mask = padding mask AND causal mask)
        output,_=self.self_attn(x,x,x,tgt_mask)
        x=self.norm1(x+self.dropout1(output))
        # 2. cross-attention: queries from decoder, keys/values from encoder output
        output,_=self.cross_attn(x,enc_out,enc_out,src_mask)
        x=self.norm2(x+self.dropout2(output))
        # 3. feed-forward
        f=self.ffn(x)
        x=self.norm3(x+self.dropout3(f))
        return x

class Decoder(nn.Module):
     def __init__ (self,num_layers,d_model=256,num_heads=4,d_ff=1024,dropout=0.1):
        super().__init__()
        self.layers=nn.ModuleList([DecoderLayer(d_model,num_heads,d_ff,dropout)for _ in range(num_layers)])
     def forward(self,x,enc_out,src_mask,tgt_mask):
        for layer in self.layers:
               x=layer(x,enc_out,src_mask,tgt_mask)
        return x

if __name__ == "__main__":
    torch.manual_seed(0)

    # --- FFN ---
    ffn = PositionwiseFeedForward(256, 1024)
    x = torch.randn(2, 5, 256)
    print("FFN out:", ffn(x).shape, " params:", sum(p.numel() for p in ffn.parameters()))  # (2,5,256) 525568

    # --- Encoder layer ---
    layer = EncoderLayer(256, 4, 1024, 0.1)
    mask = torch.ones(2, 1, 1, 5, dtype=torch.bool)
    print("EncoderLayer out:", layer(x, mask).shape)                    # (2,5,256)
    print("EncoderLayer params:", sum(p.numel() for p in layer.parameters()))  # 789760

    # --- Encoder stack ---
    enc = Encoder(3, 256, 4, 1024, 0.1)
    print("Encoder out:", enc(x, mask).shape)                           # (2,5,256)
    print("Encoder params:", sum(p.numel() for p in enc.parameters()))  # 2369280
    print("layers have separate weights:",
          enc.layers[0].ffn.linear1.weight is not enc.layers[1].ffn.linear1.weight)  # True

    # --- Padding must not change real positions ---
    enc.eval()                                   # turn dropout off for a fair comparison
    x1 = torch.randn(1, 5, 256)
    m1 = torch.ones(1, 1, 1, 5, dtype=torch.bool)
    x2 = torch.cat([x1, torch.randn(1, 3, 256)], dim=1)   # add 3 junk "pad" positions
    m2 = torch.ones(1, 1, 1, 8, dtype=torch.bool)
    m2[..., 5:] = False                                    # hide them
    same = torch.allclose(enc(x1, m1), enc(x2, m2)[:, :5], atol=1e-5)
    print("padding-invariant:", same)                                   # True

    # --- Decoder layer ---
    T, S = 4, 5
    y = torch.randn(2, T, 256)
    enc_out = torch.randn(2, S, 256)
    src_mask = torch.ones(2, 1, 1, S, dtype=torch.bool)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
    dlayer = DecoderLayer(256, 4, 1024, 0.1)
    print("DecoderLayer out:", dlayer(y, enc_out, src_mask, causal).shape)   # (2,4,256)
    print("DecoderLayer params:", sum(p.numel() for p in dlayer.parameters()))  # 1053440

    # --- Decoder stack ---
    dec = Decoder(3, 256, 4, 1024, 0.1)
    print("Decoder out:", dec(y, enc_out, src_mask, causal).shape)          # (2,4,256)
    print("Decoder params:", sum(p.numel() for p in dec.parameters()))      # 3160320

    # --- Causality: changing a future token must not affect earlier positions ---
    dec.eval()
    y2 = y.clone()
    y2[:, -1] = torch.randn(2, 256)            # change only the last position
    o1 = dec(y, enc_out, src_mask, causal)
    o2 = dec(y2, enc_out, src_mask, causal)
    print("causal:", torch.allclose(o1[:, :-1], o2[:, :-1], atol=1e-5))     # True