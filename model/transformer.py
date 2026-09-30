import torch

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
