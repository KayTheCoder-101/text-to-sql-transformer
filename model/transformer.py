import torch
import os
import sys
import torch.nn as nn
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "starter"))
)
from embeddings import TokenEmbedding, InputLayer
from model.layers import Encoder, Decoder

def make_pad_mask(ids,pad_id=0):
    mask=ids!=pad_id
    mask=mask.unsqueeze(1).unsqueeze(1)
    return mask

def make_causal_mask(size,device):
    causal_mask=torch.ones(size,size,dtype=torch.bool,device=device)
    causal_mask=torch.tril(causal_mask)
    causal_mask=causal_mask.unsqueeze(0).unsqueeze(0)
    return causal_mask

def make_decoder_mask(tgt_ids,pad_id=0):
    tgt_mask=make_pad_mask(tgt_ids,pad_id)
    causal_mask=make_causal_mask(tgt_ids.size(1),tgt_ids.device)
    tgt_mask=tgt_mask & causal_mask
    return tgt_mask

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=3, d_ff=1024, dropout=0.1, pad_id=0, max_len=512):
        super().__init__()
        self.d_model=d_model
        self.pad_id=pad_id
        self.embedding=TokenEmbedding(vocab_size,d_model,pad_id)
        self.src_input=InputLayer(self.embedding,d_model,max_len,dropout)
        self.tgt_input=InputLayer(self.embedding,d_model,max_len,dropout)
        self.encoder=Encoder(num_layers,d_model,num_heads,d_ff,dropout)
        self.decoder=Decoder(num_layers,d_model,num_heads,d_ff,dropout)
        self.out_proj=nn.Linear(d_model,vocab_size,bias=False)
        self.out_proj.weight=self.embedding.emb.weight
        self._init_weights()

    def _init_weights(self):
        emb_weight = self.embedding.emb.weight
        for p in self.parameters():
            if p.dim() > 1 and p is not emb_weight:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(emb_weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            emb_weight[self.pad_id].zero_()

    def encode(self,src):
        src_mask=make_pad_mask(src, self.pad_id)
        x=self.src_input(src)
        enc_out=self.encoder(x, src_mask)
        return enc_out,src_mask

    def decode(self,tgt_in,enc_out,src_mask):
        tgt_mask=make_decoder_mask(tgt_in,self.pad_id)
        y=self.tgt_input(tgt_in)
        y,cross_weights=self.decoder(y,enc_out,src_mask,tgt_mask)
        logits=self.out_proj(y)
        return logits,cross_weights

    def forward(self, src, tgt_in):
        enc_out,src_mask=self.encode(src)
        logits,_=self.decode(tgt_in, enc_out, src_mask)
        return logits
if __name__ == "__main__":
    # --- Masks ---
    src = torch.tensor([[5, 6, 7, 0, 0], [8, 9, 4, 3, 0]])
    tgt = torch.tensor([[2, 11, 12, 0], [2, 13, 0, 0]])
    print("pad mask:", make_pad_mask(src).shape, " decoder mask:", make_decoder_mask(tgt).shape)

    # --- Full model ---
    torch.manual_seed(0)
    V = 8000
    model = Transformer(V)
    model.eval()                                       # dropout off for fair comparisons

    src = torch.randint(4, V, (2, 10))
    src[1, 7:] = 0                                     # 2nd example padded
    tgt_in = torch.randint(4, V, (2, 6))
    tgt_in[:, 0] = 2                                   # starts with <s>

    logits = model(src, tgt_in)
    print("logits:", logits.shape)                     # (2, 6, 8000)

    # Weight sharing (manual check)
    print("weight sharing:", model.out_proj.weight is model.embedding.emb.weight)   # True

    # Parameter count (Task 2.7)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameters:", n)                  # 7577600

    # Pad row is zero after initialization
    print("pad row zero:", model.embedding.emb.weight[0].abs().max().item())        # 0.0

    # Causal check (manual): change the last target token, earlier logits must not change
    t2 = tgt_in.clone()
    t2[:, -1] = 99
    same = torch.allclose(model(src, tgt_in)[:, :-1], model(src, t2)[:, :-1], atol=1e-5)
    print("causal check:", same)                       # True

    # Padding check (manual): add extra <pad> to the source, output must not change
    src_padded = torch.cat([src, torch.zeros(2, 5, dtype=torch.long)], dim=1)
    same = torch.allclose(model(src, tgt_in), model(src_padded, tgt_in), atol=1e-5)
    print("padding check:", same)                      # True