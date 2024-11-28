import torch.nn as nn
import torch

class Self_Attention(nn.Module):

    def __init__(self, d_in, d_out, qkv=False):
        super().__init__()
        self.w_q = nn.Linear(d_in, d_out, bias=qkv)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv)

    def forward(self, x):
        Query = self.w_q(x)
        Keys = self.w_k(x)
        Values = self.w_v(x)
        att_scores = Query @ Keys.T
        att_weights = torch.softmax(
            att_scores / Keys.shape[-1]**0.5, dim=-1
        )
        context_values = att_weights @ Values
        return context_values
    