import torch
import torch.nn as nn

class PostSmoothLayer(nn.Module):

    def __init__(self, hidden_dim, top_k, dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer('smooth',torch.ones(2*hidden_dim*top_k).to(dtype))
        self.register_buffer('top_k_indices',torch.ones(2*hidden_dim*top_k).to(torch.int32))
        self.top_k = top_k
        self.dtype = dtype

    def forward(self, x):
        B,H,S,D = x.shape
        x=x.transpose(1,2).reshape(B,S,-1)
        
        x[:,:,self.top_k_indices] = x[:,:,self.top_k_indices]*self.smooth
        x = x.reshape(B, S, H, D).transpose(1,2)        
        
        return x
    
