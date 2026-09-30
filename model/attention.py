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

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model=256,num_heads=4): 
        super().__init__() 
        assert d_model%num_heads==0
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
        self.attention=ScaledDotProductAttention()


    def forward(self,querey,key,value,mask=None):  
         B_size=querey.size(0)
         q=self.w_q(querey)
         k=self.w_k(key)
         v=self.w_v(value)
         q=q.view(q.size(0),-1,self.num_heads,self.d_k).transpose(1,2)
         k=k.view(k.size(0),-1,self.num_heads,self.d_k).transpose(1,2)
         v=v.view(v.size(0),-1,self.num_heads,self.d_k).transpose(1,2)
         out,weights=self.attention(q,k,v,mask)
         out=out.transpose(1,2).contiguous().view(out.size(0),-1,self.num_heads*self.d_k)
         out=self.w_o(out)
         return out,weights


